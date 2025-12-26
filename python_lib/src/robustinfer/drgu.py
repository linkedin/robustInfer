import pandas as pd
import torch
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from torch.func import jacfwd, jacrev

from .ee import EstimatingEquation


def make_Xg(a, b):
    """Create expanded feature matrix [1, w_i, w_j]"""
    ones = torch.ones(a.shape[0], 1, dtype=a.dtype, device=a.device)
    return torch.cat([ones, a, b], dim=1)


def data_pairwise(y, z, w):
    """Convert data to pairwise format for U-statistics computation"""
    n = y.size(0)
    Wt = torch.cat([torch.ones(n, 1, dtype=w.dtype, device=w.device), w], dim=1)

    # Get upper triangular indices (i < j)
    tri_u, tri_v = torch.triu_indices(n, n, offset=1)

    wi, wj = w[tri_u], w[tri_v]  # (m,p)
    zi, zj = z[tri_u], z[tri_v]
    yi, yj = y[tri_u], y[tri_v]

    Wt_i, Wt_j = Wt[tri_u], Wt[tri_v]  # (m,p+1)
    Xg_ij, Xg_ji = make_Xg(wi, wj), make_Xg(wj, wi)  # (m,2p+1)

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
        'j': tri_v,
    }


def safe_sigmoid(x):
    """Safe sigmoid function with clipping to prevent overflow"""
    return torch.sigmoid(torch.clamp(x, -15.0, 15.0))


@torch.compile
def compute_h_f_fisher(theta, data):
    """Compute h and f vectors for Fisher scoring"""
    # Extract data
    Wt_i, Wt_j = data['Wt_i'], data['Wt_j']
    Xg_ij, Xg_ji = data['Xg_ij'], data['Xg_ji']
    yi, yj = data['yi'], data['yj']
    zi, zj = data['zi'], data['zj']

    delta, beta, gamma = theta['delta'], theta['beta'], theta['gamma']

    # Predictions
    pi_i = safe_sigmoid(torch.sum(Wt_i * beta, dim=1))
    pi_j = safe_sigmoid(torch.sum(Wt_j * beta, dim=1))
    g_ij = safe_sigmoid(torch.sum(Xg_ij * gamma, dim=1))
    g_ji = safe_sigmoid(torch.sum(Xg_ji * gamma, dim=1))
    d_ij = (Xg_ij[:, 0:1] * delta).sum(dim=1)

    # Indicators
    I_ij = (yi >= yj).float()
    I_ji = 1.0 - I_ij

    # h vector (3-component) for all pairs
    num1 = zi * (1 - zj) / (2 * pi_i * (1 - pi_j)) * (I_ij - g_ij)
    num2 = zj * (1 - zi) / (2 * pi_j * (1 - pi_i)) * (I_ji - g_ji)
    h1 = num1 + num2 + 0.5 * (g_ij + g_ji) - 0.5
    h2 = 0.5 * (zi + zj)
    h3 = 0.5 * (zi * (1 - zj) * I_ij + zj * (1 - zi) * I_ji)
    h = torch.stack([h1, h2, h3], dim=1)  # (m,3)

    # f vector
    f1 = d_ij
    f2 = 0.5 * (pi_i + pi_j)
    f3 = 0.5 * (pi_i * (1 - pi_j) * g_ij + pi_j * (1 - pi_i) * g_ji)
    f = torch.stack([f1, f2, f3], dim=1)

    return h, f


def _compute_h_fisher(theta, data):
    """Compute h vector only"""
    h, _ = compute_h_f_fisher(theta, data)
    return h


def _compute_f_fisher(theta, data):
    """Compute f vector only"""
    _, f = compute_h_f_fisher(theta, data)
    return f


def compute_h_f_jacobians_pytorch(theta, data, method='jacfwd'):
    """Compute both h and f jacobians efficiently using jacrev with has_aux"""

    # Create function for jacrev with has_aux that works directly with dict
    def hf_for_jac(theta_dict):
        h, f = compute_h_f_fisher(theta_dict, data)
        # Return (h, f) as the main output (jacobian computed w.r.t. this)
        # and (h, f) as auxiliary output (returned as-is)
        return (h, f), (h, f)

    # Compute jacobians - jacrev works directly with dictionary inputs
    if method == 'jacrev':
        (h_jac, f_jac), (h, f) = jacrev(hf_for_jac, has_aux=True)(theta)
    elif method == 'jacfwd':
        (h_jac, f_jac), (h, f) = jacfwd(hf_for_jac, has_aux=True)(theta)
    elif method == 'manual':
        h, f = compute_h_f_fisher(theta, data)
        h_jac = compute_jacobian_manual(_compute_h_fisher, theta, data)
        f_jac = compute_jacobian_manual(_compute_f_fisher, theta, data)
    else:
        raise ValueError(f'Unknown method {method}')

    return h_jac, f_jac, h, f


def compute_jacobian_manual(func, theta, data):
    """Manual jacobian computation as fallback"""
    jacobian_dict = {}

    for param_name, param_value in theta.items():
        param_shape = param_value.shape

        # Create fresh parameters for each computation to avoid gradient accumulation issues
        theta_for_param = {k: v.clone().detach().requires_grad_(True) for k, v in theta.items()}

        # Compute function output
        output = func(theta_for_param, data)
        batch_size, output_dim = output.shape

        # Initialize jacobian tensor
        jac = torch.zeros(
            batch_size, output_dim, *param_shape, device=output.device, dtype=output.dtype
        )

        # Compute jacobian for each output element
        for batch_idx in range(batch_size):
            for out_idx in range(output_dim):
                # Compute gradient for this specific output element
                grad_outputs = torch.zeros_like(output)
                grad_outputs[batch_idx, out_idx] = 1.0

                # Compute gradients
                grads = torch.autograd.grad(
                    outputs=output,
                    inputs=theta_for_param[param_name],
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )

                if grads[0] is not None:
                    jac[batch_idx, out_idx] = grads[0]

        jacobian_dict[param_name] = jac

    return jacobian_dict


def _compute_B_u_ij(theta, V_inv, data):
    """Compute B matrix and u_ij vectors"""
    # Use efficient jacobian computation that computes both jacobians in one pass
    h_jac, f_jac, h, f = compute_h_f_jacobians_pytorch(theta, data)

    # Concatenate jacobian components
    D_ij = torch.cat([f_jac['delta'], f_jac['beta'], f_jac['gamma']], dim=2)
    M0_ij = torch.cat([h_jac['delta'], h_jac['beta'], h_jac['gamma']], dim=2)
    M_ij = D_ij - M0_ij

    # Compute B and u_ij
    G_ij = torch.transpose(D_ij, 1, 2) @ V_inv
    B_ij = torch.bmm(G_ij, M_ij)
    B = torch.mean(B_ij, dim=0)

    S_ij = h - f
    u_ij = torch.bmm(G_ij, S_ij.unsqueeze(-1)).squeeze(-1)

    return B, u_ij


def compute_B_U(theta, V_inv, data):
    """Compute B matrix and U vector"""
    B, u_ij = _compute_B_u_ij(theta, V_inv, data)
    U = torch.mean(u_ij, dim=0)
    return B, U


def compute_B_U_Sig(theta, V_inv, data):
    """Compute B matrix, U vector, and Sigma matrix"""
    B, u_ij = _compute_B_u_ij(theta, V_inv, data)
    U = torch.mean(u_ij, dim=0)

    n = max(torch.max(data['i']).item(), torch.max(data['j']).item()) + 1
    d = u_ij.shape[1]

    # Initialize u_i matrix
    device = u_ij.device
    u_i = torch.zeros(n, d, device=device)

    # Add contributions from pairs (i,j) and (j,i)
    u_i.index_add_(0, data['i'], u_ij)
    u_i.index_add_(0, data['j'], u_ij)
    u_i = u_i / (n - 1)

    # Compute sigma matrix
    sig_i = torch.bmm(u_i.unsqueeze(-1), u_i.unsqueeze(-2))
    Sig = torch.mean(sig_i, dim=0)

    return B, U, Sig


def compute_delta(theta, V_inv, data, lamb=0.0, option='fisher'):
    """Compute parameter update step"""
    if option == 'fisher':
        B, U = compute_B_U(theta, V_inv, data)
        J = -B
    else:
        raise ValueError(f'Unknown option {option}')

    # Add regularization to handle numerical issues
    # Note: Don't penalize delta (first parameter), matching Scala implementation
    if lamb > 0:
        vectorized_theta = torch.cat([v.flatten() for v in theta.values()])
        d = vectorized_theta.shape[0]
        # Create penalty mask: don't penalize delta (first parameter)
        penalty_mask = torch.ones(d, device=J.device)
        delta_size = theta['delta'].numel()
        penalty_mask[:delta_size] = 0.0
        penalty_diag = lamb * torch.diag(penalty_mask)

        regularized_J = J - penalty_diag
        regularized_U = U - penalty_diag @ vectorized_theta
    else:
        regularized_J = J
        regularized_U = U

    # Try to solve the linear system
    try:
        step = torch.linalg.solve(regularized_J, -regularized_U)
    except torch._C._LinAlgError:
        # If still singular, use pseudo-inverse as fallback
        step = torch.linalg.pinv(regularized_J) @ (-regularized_U)

    return step, J


def update_theta(theta, step):
    """Update theta parameters with step"""
    theta_new = {}
    start = 0
    for k, v in theta.items():
        size = v.numel()
        theta_new[k] = v + step[start : start + size].reshape(v.shape)
        start += size
    return theta_new


def get_theta_init(data, z):
    """Initialize theta parameters using logistic regression"""
    yi, yj = data['yi'], data['yj']
    zi, zj = data['zi'], data['zj']
    Wt = data['Wt']
    Xg_ij, Xg_ji = data['Xg_ij'], data['Xg_ji']
    Wt_i, Wt_j = data['Wt_i'], data['Wt_j']

    # Convert to numpy for sklearn
    I_ij = (yi >= yj).float()
    I_ji = 1.0 - I_ij
    h3 = zi * (1 - zj) * I_ij + zj * (1 - zi) * I_ji

    # Fit logistic regression models
    z_logistic = LogisticRegression(random_state=0, fit_intercept=False).fit(
        Wt.cpu().numpy(), z.cpu().numpy()
    )

    feature_matrix = (zi * (1 - zj)).unsqueeze(1) * Xg_ij + (zj * (1 - zi)).unsqueeze(1) * Xg_ji
    u_logistic = LogisticRegression(random_state=0, fit_intercept=False).fit(
        feature_matrix.cpu().numpy(), h3.cpu().numpy()
    )

    # Convert back to torch tensors
    device = yi.device
    beta = torch.tensor(z_logistic.coef_[0], dtype=torch.float32, device=device)
    gamma = torch.tensor(u_logistic.coef_[0], dtype=torch.float32, device=device)

    # Compute initial delta
    bi = torch.tensor(
        z_logistic.predict_proba(Wt_i.cpu().numpy())[:, 1], dtype=torch.float32, device=device
    )
    bj = torch.tensor(
        z_logistic.predict_proba(Wt_j.cpu().numpy())[:, 1], dtype=torch.float32, device=device
    )

    delta_ipw = 0.5 * torch.mean(
        zi * (1 - zj) / (bi * (1 - bj)) * I_ij + zj * (1 - zi) / (bj * (1 - bi)) * I_ji
    )

    return {
        'delta': torch.tensor([delta_ipw - 0.5], device=device),
        'beta': beta,
        'gamma': gamma,
    }


class DRGU(EstimatingEquation):
    """
    PyTorch implementation of Doubly Robust Generalized U model.
    This class extends the EstimatingEquation class to implement a doubly robust estimator
    for Doubly Robust U using PyTorch tensors and automatic differentiation.
    """

    def __init__(self, data, covariates, treatment, response, device='cpu'):
        """
        Initialize the DRGU model with data, covariates, treatment, and response.

        :param data: np.ndarray or pandas.DataFrame, the dataset
        :param covariates: list, names of covariate columns
        :param treatment: str, name of the treatment variable
        :param response: str, name of the response variable
        :param device: str, device to use for PyTorch tensors ('cpu' or 'cuda')
        """
        super().__init__(data, covariates, treatment, response)
        self.device = device

        # Convert data to PyTorch tensors
        self.w = torch.tensor(self.data[self.covariates].values, dtype=torch.float32, device=device)
        self.z = torch.tensor(self.data[self.treatment].values, dtype=torch.float32, device=device)
        self.y = torch.tensor(self.data[self.response].values, dtype=torch.float32, device=device)

        # Initialize parameters
        self.theta = {
            'delta': torch.tensor([0.0], dtype=torch.float32, device=device),
            'beta': torch.zeros(len(self.covariates) + 1, dtype=torch.float32, device=device),
            'gamma': torch.zeros(2 * len(self.covariates) + 1, dtype=torch.float32, device=device),
        }
        self._theta_initialized = False  # Track if theta was manually set

    def set_theta(self, theta_dict):
        """
        Set theta parameters explicitly. Useful for testing and custom initialization.

        Args:
            theta_dict: Dictionary with 'delta', 'beta', 'gamma' parameters
        """
        # Validate input
        required_keys = {'delta', 'beta', 'gamma'}
        if not all(key in theta_dict for key in required_keys):
            raise ValueError(f'theta_dict must contain keys: {required_keys}')

        # Set theta with proper device and dtype
        self.theta = {
            'delta': theta_dict['delta'].to(device=self.device, dtype=torch.float32).clone(),
            'beta': theta_dict['beta'].to(device=self.device, dtype=torch.float32).clone(),
            'gamma': theta_dict['gamma'].to(device=self.device, dtype=torch.float32).clone(),
        }
        self._theta_initialized = True

    def fit(self, max_iter=20, tol=1e-6, lamb=0.0, option='fisher', verbose=True):
        """
        Fit the DRGU model to the data.
        This method implements the logic for fitting the model using PyTorch.

        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            lamb: Regularization parameter (L2 penalty)
            option: Optimization method ('fisher' or other)
            verbose: Whether to print convergence information
        """
        # Prepare data for pairwise computation
        data = data_pairwise(self.y, self.z, self.w)

        # Initialize parameters (use existing if manually set)
        theta_init = get_theta_init(data, self.z) if not self._theta_initialized else self.theta

        # Solve the estimating equation
        theta, J, Var = self._solve_ugee(data, theta_init, max_iter, tol, lamb, option, verbose)

        # Store results
        self.theta = theta
        self.theta['delta'] = self.theta['delta'] + 0.5
        self.coefficients = torch.cat([v.flatten() for v in theta.values()])
        self.variance_matrix = Var * (1.0 / self.w.shape[0])

    def _solve_ugee(
        self,
        data,
        theta_init,
        max_iter=20,
        tol=1e-6,
        lamb=0.0,
        option='fisher',
        verbose=True,
    ):
        """Solve the U-statistic generalized estimating equation"""
        V_inv = torch.eye(3, device=self.device)
        theta = {k: v.clone() for k, v in theta_init.items()}

        for i in range(max_iter):
            step, J = compute_delta(theta, V_inv, data, lamb, option)

            if i % 10 == 0 and verbose:
                print(f'Step {i} gradient norm: {torch.norm(step):.6f}')

            theta = update_theta(theta, step)

            if torch.norm(step) < tol:
                if verbose:
                    print(f'Converged after {i} iterations')
                break

        if i == max_iter - 1 and verbose:
            print(f'Did not converge, norm step = {torch.norm(step)}')

        # Compute variance
        B, U, Sig = compute_B_U_Sig(theta, V_inv, data)

        # Use pseudo-inverse for numerical stability
        try:
            B_inv = torch.linalg.inv(B)
        except torch._C._LinAlgError:
            # Use pseudo-inverse if B is singular
            B_inv = torch.linalg.pinv(B)

        Var = 4 * B_inv @ Sig @ B_inv.T

        return theta, J, Var

    def summary(self):
        """
        Generate a summary of the model fit, including coefficients, standard errors,
        z-scores, and p-values.
        """
        # Compute standard errors
        standard_errors = torch.sqrt(torch.diag(self.variance_matrix))

        # Compute z-scores
        null_hypothesis = torch.zeros_like(self.coefficients)
        null_hypothesis[0] = 0.5  # delta null hypothesis
        z_scores = (self.coefficients - null_hypothesis) / standard_errors

        # Compute p-values
        p_values = 2 * (1 - norm.cdf(torch.abs(z_scores).cpu().numpy()))

        # Create a summary table
        row_names = (
            ['delta']
            + [f'beta_{i}' for i in range(len(self.theta['beta']))]
            + [f'gamma_{i}' for i in range(len(self.theta['gamma']))]
        )

        summary = pd.DataFrame(
            {
                'Names': row_names,
                'Coefficient': self.coefficients.cpu().numpy(),
                'Null_Hypothesis': null_hypothesis.cpu().numpy(),
                'Std_Error': standard_errors.cpu().numpy(),
                'Z_Score': z_scores.cpu().numpy(),
                'P_Value': p_values,
            }
        )

        return summary
