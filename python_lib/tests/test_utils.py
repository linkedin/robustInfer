import pytest
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LogisticRegression
from robustinfer.utils import make_Xg, data_pairwise, get_theta_init, safe_sigmoid, compute_h_f_fisher, compute_B_U_Sig, compute_delta, update_theta    

@pytest.fixture
def mock_data():
    return {
        "Wt_i": jnp.array([[1.0, 0.5], [1.0, 1.5]]),
        "Wt_j": jnp.array([[1.0, 1.0], [1.0, 2.0]]),
        "Xg_ij": jnp.array([[1.0, 0.5, 1.0], [1.0, 1.5, 0.5]]),
        "Xg_ji": jnp.array([[1.0, 1.0, 0.5], [1.0, 0.5, 1.5]]),
        "yi": jnp.array([1.0, 2.0]),
        "yj": jnp.array([0.5, 1.5]),
        "zi": jnp.array([0, 1]),
        "zj": jnp.array([1, 0]),
        "i": jnp.array([0, 1]),
        "j": jnp.array([1, 0])
    }

def test_make_Xg():
    # Test the make_Xg function
    a = jnp.array([1.0, 2.0, 3.0])[:,None]
    b = jnp.array([4.0, 5.0, 6.0])[:,None]
    result = make_Xg(a, b)
    expected = jnp.array([
        [1.0, 1.0, 4.0],
        [1.0, 2.0, 5.0],
        [1.0, 3.0, 6.0]
    ])
    assert jnp.allclose(result, expected), "make_Xg did not return the expected result"

def test_data_pairwise():
    # Test the data_pairwise function
    y = jnp.array([1.0, 2.0, 3.0])
    z = jnp.array([0, 1, 0])
    w = jnp.array([[0.5], [1.5], [2.5]])
    result = data_pairwise(y, z, w)

    # Check the keys in the result
    expected_keys = {'Wt', 'Xg_ij', 'Xg_ji', 'Wt_i', 'Wt_j', 'yi', 'yj', 'zi', 'zj', 'wi', 'wj', 'i', 'j'}
    assert set(result.keys()) == expected_keys, "data_pairwise did not return the expected keys"

    # Check the shapes of the outputs
    assert result['Wt'].shape == (3, 2), "Wt shape is incorrect"
    assert result['Xg_ij'].shape == (3, 3), "Xg_ij shape is incorrect"
    assert result['Xg_ji'].shape == (3, 3), "Xg_ji shape is incorrect"

def test_get_theta_init():
    # Test the get_theta_init function
    data = {
        'yi': jnp.array([1.0, 2.0, 3.0]),
        'yj': jnp.array([0.5, 1.5, 2.5]),
        'zi': jnp.array([0, 1, 0]),
        'zj': jnp.array([1, 0, 1]),
        'Wt': jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]]),
        'Xg_ij': jnp.array([[1.0, 0.5, 1.0], [1.0, 1.5, 0.5], [1.0, 2.5, 1.5]]),
        'Xg_ji': jnp.array([[1.0, 1.0, 0.5], [1.0, 0.5, 1.5], [1.0, 1.5, 2.5]]),
        'Wt_i': jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]]),
        'Wt_j': jnp.array([[1.0, 1.5], [1.0, 0.5], [1.0, 1.5]])
    }
    z = jnp.array([0, 1, 0])

    result = get_theta_init(data, z)

    # Check the keys in the result
    expected_keys = {'delta', 'beta', 'gamma'}
    assert set(result.keys()) == expected_keys, "get_theta_init did not return the expected keys"

    # Check the shapes of the outputs
    assert result['delta'].shape == (1,), "delta shape is incorrect"
    assert result['beta'].shape == (2,), "beta shape is incorrect"
    assert result['gamma'].shape == (3,), "gamma shape is incorrect"

def test_safe_sigmoid():
    # Test the _safe_sigmoid function
    x = jnp.array([-100.0, 0.0, 100.0])
    result = safe_sigmoid(x)
    expected = jnp.array([0.0, 0.5, 1.0])  # Clipped sigmoid values
    assert jnp.allclose(result, expected, rtol=1e-03, atol=1e-04), "_safe_sigmoid did not return the expected result"

def test_compute_h_f_fisher():
    # Mock theta
    theta = {
        "delta": jnp.array([0.5]),
        "beta": jnp.array([0.1, 0.2]),
        "gamma": jnp.array([0.3, 0.4, 0.5])
    }

    # Mock data
    data = {
        "Wt_i": jnp.array([[1.0, 0.5], [1.0, 1.5]]),
        "Wt_j": jnp.array([[1.0, 1.0], [1.0, 2.0]]),
        "Xg_ij": jnp.array([[1.0, 0.5, 1.0], [1.0, 1.5, 0.5]]),
        "Xg_ji": jnp.array([[1.0, 1.0, 0.5], [1.0, 0.5, 1.5]]),
        "yi": jnp.array([1.0, 2.0]),
        "yj": jnp.array([0.5, 1.5]),
        "zi": jnp.array([0, 1]),
        "zj": jnp.array([1, 0])
    }

    # Call the function
    h, f = compute_h_f_fisher(theta, data)

    expected_h = jnp.array(
        [[-0.66821027,  0.5       ,  0.        ],
        [ 1.3003929 ,  0.5       ,  0.5       ]])

    expected_f = jnp.array(
        [[0.5       , 0.5621382 , 0.17876694],
        [0.5       , 0.61057353, 0.18292071]])

    # Assertions
    assert jnp.allclose(h, expected_h), "h vector is incorrect"
    assert jnp.allclose(f, expected_f), "f vector is incorrect"

def test_compute_B_U_Sig():
    # Mock theta
    theta = {
        "delta": jnp.array([0.5]),
        "beta": jnp.array([0.1, 0.2]),
        "gamma": jnp.array([0.3, 0.4, 0.5])
    }

    # Mock V_inv (inverse of variance matrix)
    V_inv = jnp.eye(3)

    # Mock data
    data = {
        "Wt_i": jnp.array([[1.0, 0.5], [1.0, 1.5]]),
        "Wt_j": jnp.array([[1.0, 1.0], [1.0, 2.0]]),
        "Xg_ij": jnp.array([[1.0, 0.5, 1.0], [1.0, 1.5, 0.5]]),
        "Xg_ji": jnp.array([[1.0, 1.0, 0.5], [1.0, 0.5, 1.5]]),
        "yi": jnp.array([1.0, 2.0]),
        "yj": jnp.array([0.5, 1.5]),
        "zi": jnp.array([0, 1]),
        "zj": jnp.array([1, 0]),
        "i": jnp.array([0, 1]),
        "j": jnp.array([1, 0])
    }

    # Call the function
    B, U, Sig = compute_B_U_Sig(theta, V_inv, data)

    # Expected results based on the mock data
    expected_B = jnp.array(
        [[ 1.        ,  0.02781541, -0.27603954,  0.20807958,  0.33247495,
         0.0361871 ],
       [ 0.        ,  0.05955444,  0.07354937, -0.00139919, -0.00126154,
        -0.00126466],
       [ 0.        ,  0.07354937,  0.10565361, -0.00184748, -0.00173953,
        -0.00176144],
       [ 0.        , -0.00139919, -0.00184748,  0.00209384,  0.0018017 ,
         0.00178561],
       [ 0.        , -0.00126154, -0.00173953,  0.0018017 ,  0.00157581,
         0.00156811],
       [ 0.        , -0.00126466, -0.00176144,  0.00178561,  0.00156811,
         0.00156202]]
    )

    expected_Sig = jnp.array(
        [[ 3.3822410e-02,  4.6359780e-03,  7.0270584e-03, -4.2670363e-04,
        -6.0150150e-04, -6.5468712e-04],
       [ 4.6359780e-03,  6.3544535e-04,  9.6318650e-04, -5.8487512e-05,
        -8.2446742e-05, -8.9736808e-05],
       [ 7.0270584e-03,  9.6318650e-04,  1.4599655e-03, -8.8653389e-05,
        -1.2496999e-04, -1.3602001e-04],
       [-4.2670363e-04, -5.8487512e-05, -8.8653389e-05,  5.3832946e-06,
         7.5885446e-06,  8.2595352e-06],
       [-6.0150150e-04, -8.2446742e-05, -1.2496999e-04,  7.5885446e-06,
         1.0697168e-05,  1.1643028e-05],
       [-6.5468712e-04, -8.9736808e-05, -1.3602001e-04,  8.2595352e-06,
         1.1643028e-05,  1.2672522e-05]]
    )

    expected_U = jnp.array(
        [-0.1839087 , -0.02520804, -0.03820949,  0.00232019,  0.00327065,
        0.00355985]
    )

    # Assertions for shapes
    assert B.shape == (6, 6), "B matrix shape is incorrect"
    assert U.shape == (6,), "U vector shape is incorrect"
    assert Sig.shape == (6, 6), "Sig matrix shape is incorrect"
    # Assertions for values
    assert jnp.allclose(B, expected_B), "B matrix values are incorrect"
    assert jnp.allclose(U, expected_U), "U vector values are incorrect"
    assert jnp.allclose(Sig, expected_Sig), "Sig matrix values are incorrect"

def test_compute_delta(mock_data):
    # Mock theta
    theta = {
        "delta": jnp.array([0.5]),
        "beta": jnp.array([0.1, 0.2]),
        "gamma": jnp.array([0.3, 0.4, 0.5])
    }

    # Mock V_inv (inverse of variance matrix)
    V_inv = jnp.eye(3)

    # Call the function
    step, J = compute_delta(theta, V_inv, mock_data, lamb=0.0, option="fisher")

    # Assertions for shapes
    assert step.shape == (6,), "Step vector shape is incorrect"
    assert J.shape == (6, 6), "Jacobian matrix shape is incorrect"

    # Additional checks (optional, based on expected values)
    assert jnp.all(jnp.diag(J) < 0), "Jacobian matrix diagonal should be negative"

def test_update_theta():
    # Mock theta
    theta = {
        "delta": jnp.array([0.5]),
        "beta": jnp.array([0.1, 0.2]),
        "gamma": jnp.array([0.3, 0.4, 0.5])
    }

    # Mock step vector
    step = jnp.array([0.1, -0.05, 0.02, -0.01, 0.03, -0.02])

    # Call the function
    updated_theta = update_theta(theta, step)

    # Expected updated theta
    expected_theta = {
        "delta": jnp.array([0.6]),  # 0.5 + 0.1
        "beta": jnp.array([0.05, 0.22]),  # [0.1 - 0.05, 0.2 + 0.02]
        "gamma": jnp.array([0.29, 0.43, 0.48])  # [0.3 - 0.01, 0.4 + 0.03, 0.5 - 0.02]
    }

    # Assertions
    assert jnp.allclose(updated_theta["delta"], expected_theta["delta"]), "delta update is incorrect"
    assert jnp.allclose(updated_theta["beta"], expected_theta["beta"]), "beta update is incorrect"
    assert jnp.allclose(updated_theta["gamma"], expected_theta["gamma"]), "gamma update is incorrect"
