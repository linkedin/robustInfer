import jax.numpy as jnp
import jax
import numpy as np
from sklearn.linear_model import LogisticRegression

def make_Xg(a,b):
    return jnp.concatenate([jnp.ones_like(a), a, b], axis=1)  # [1, w_i, w_j]

def data_pairwise(y, z, w):
    n = y.size
    Wt  = jnp.concatenate([jnp.ones((n,1)), w], axis=1)

    tri_u, tri_v = jnp.triu_indices(n, k=1)                   # i<j indices
    m            = tri_u.size                                 # #pairs

    wi, wj    = w[tri_u],            w[tri_v]                 # (m,p)
    zi, zj    = z[tri_u],            z[tri_v]
    yi, yj    = y[tri_u],            y[tri_v]

    Wt_i, Wt_j = Wt[tri_u],          Wt[tri_v]                # (m,p+1)
    Xg_ij, Xg_ji = make_Xg(wi,wj), make_Xg(wj,wi)               # (m,2p+1)

    return {
        'Wt': Wt,
        'Xg_ij': Xg_ij,
        'Xg_ji': Xg_ji,
        'Wt_i': Wt_i,
        'Wt_j': Wt_j,
        'yi': yi,
        'yj': yj,
        'zi': zi,
        'zj': zj,
        'wi': wi,
        'wj': wj,
        'i': tri_u,
        'j': tri_v
    }

def safe_sigmoid(x):
    return jax.nn.sigmoid(jnp.clip(x, -10.0, 10.0))

def compute_h_f_fisher(theta, data):
    # All inputs
    Wt_i, Wt_j = data['Wt_i'], data['Wt_j']
    Xg_ij, Xg_ji = data['Xg_ij'], data['Xg_ji']
    yi, yj = data['yi'], data['yj']
    zi, zj = data['zi'], data['zj']
    m = zi.size

    delta, beta, gamma = theta["delta"], theta["beta"], theta["gamma"]

    # predictions
    # p_delta = safe_sigmoid(delta)
    pi_i  = safe_sigmoid(jnp.sum(Wt_i * beta, axis=1))
    pi_j  = safe_sigmoid(jnp.sum(Wt_j * beta, axis=1))
    g_ij = safe_sigmoid(jnp.sum(Xg_ij * gamma, axis=1))
    g_ji = safe_sigmoid(jnp.sum(Xg_ji * gamma, axis=1))

    # indicators
    I_ij = (yi >= yj).astype(jnp.float32)
    I_ji = 1. - I_ij

    # h vector (3‑component) for all pairs
    num1 = zi*(1-zj)/(2*pi_i*(1-pi_j)) * (I_ij - g_ij)
    num2 = zj*(1-zi)/(2*pi_j*(1-pi_i)) * (I_ji - g_ji)
    h1   = num1 + num2 + 0.5*(g_ij + g_ji)
    h2   = 0.5*(zi + zj)
    h3   = 0.5*(zi*(1-zj)*I_ij + zj*(1-zi)*I_ji)
    h    = jnp.stack([h1,h2,h3], axis=1)                  # (m,3)

    # f vector
    f1   = jnp.full_like(h1, delta)
    f2   = 0.5*(pi_i + pi_j)
    f3   = 0.5*(pi_i*(1-pi_j)*g_ij + pi_j*(1-pi_i)*g_ji)
    f    = jnp.stack([f1,f2,f3], axis=1)
    return h, f

def _compute_h_fisher(theta, data):
    h, _ = compute_h_f_fisher(theta, data)
    return h

def _compute_f_fisher(theta, data):
    _, f = compute_h_f_fisher(theta, data)
    return f

@jax.jit
def _compute_B_u_ij(theta, V_inv, data):
    h, f = compute_h_f_fisher(theta, data)
    D = jax.jacfwd(_compute_f_fisher, argnums=0)(theta, data)
    D_ij = jnp.concatenate([D['delta'], D['beta'], D['gamma']], axis=2)

    M0 = jax.jacfwd(_compute_h_fisher, argnums=0)(theta, data)
    M0_ij = jnp.concatenate([M0['delta'], M0['beta'], M0['gamma']], axis=2)
    M_ij = D_ij - M0_ij

    G_ij = jnp.transpose(D_ij, (0, 2, 1)) @ V_inv
    B_ij = jnp.einsum('npq,nqc->npc', G_ij, M_ij)
    B = jnp.mean(B_ij, axis=0)

    S_ij = h - f
    u_ij = jnp.einsum('npc,nc->np', G_ij, S_ij)

    return B, u_ij

def _compute_B_U(theta, V_inv, data):
    B, u_ij = _compute_B_u_ij(theta, V_inv, data)
    U = jnp.mean(u_ij, axis=0)
    return B, U

def compute_B_U_Sig(theta, V_inv, data):
    B, u_ij = _compute_B_u_ij(theta, V_inv, data)
    U = jnp.mean(u_ij, axis=0)

    n = jnp.maximum(jnp.max(data['i']), jnp.max(data['j'])) + 1
    d = u_ij.shape[1]
    u_i = jnp.zeros((n,d)).at[data['i']].add(u_ij).at[data['j']].add(u_ij)/n
    sig_i = jnp.einsum('np,nq->npq', u_i, u_i)

    # Sig_ij = jnp.einsum('np,nq->npq', u_ij, u_ij)
    Sig = jnp.mean(sig_i, axis=0)

    return B, U, Sig

def compute_delta(theta, V_inv, data, lamb=0.0, option="fisher"):
    if option == "fisher":
        B, U = _compute_B_U(theta, V_inv, data)
        J = -B
    else:
        raise ValueError(f"Unknown option {option}")
    step = jnp.linalg.solve(J+ lamb * jnp.eye(J.shape[0]), -U)
    return step, J

def update_theta(theta, step):
    start = 0
    for k,v in theta.items():
        theta[k] += step[start:start+v.size]
        start += v.size
    return theta

def get_theta_init(data, z):
    yi, yj = data['yi'], data['yj']
    zi, zj = data['zi'], data['zj']
    # wi, wj = data['wi'], data['wj']
    Wt = data['Wt']
    Xg_ij, Xg_ji = data['Xg_ij'], data['Xg_ji']
    Wt_i, Wt_j = data['Wt_i'], data['Wt_j']

    I_ij = (yi >= yj).astype(jnp.float32)
    I_ji = 1. - I_ij
    h3   = zi*(1-zj)*I_ij + zj*(1-zi)*I_ji
    z_logistic = LogisticRegression(random_state=0, fit_intercept=False).fit(Wt, z)
    u_logistic = LogisticRegression(random_state=0, fit_intercept=False).fit((zi*(1-zj))[:,None]*Xg_ij + (zj*(1-zi))[:,None]*Xg_ji, h3)

    beta = jnp.array(z_logistic.coef_[0])
    gamma = jnp.array(u_logistic.coef_[0])
    # u_ij = u_logistic.predict_proba(Xg_ij)[:,1]
    # u_ji = u_logistic.predict_proba(Xg_ji)[:,1]
    # delta_reg = 0.5 * np.mean(zi*(1-zj)*(I_ij - u_ij) + zj*(1-zi)*(I_ji - u_ji) + (u_ij + u_ji))
    bi = z_logistic.predict_proba(Wt_i)[:,1]
    bj = z_logistic.predict_proba(Wt_j)[:,1]
    delta_ipw = 0.5*jnp.mean(zi*(1-zj)/(bi*(1-bj))*I_ij + zj*(1-zi)/(bj*(1-bi))*I_ji)

    return {
        "delta": jnp.array([delta_ipw]),
        "beta": beta,
        "gamma": gamma,
    }



