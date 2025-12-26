import pandas as pd
import pytest

try:
    from robustinfer.jax import DRGUJax as DRGU

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    DRGU = None


@pytest.fixture
def mock_data():
    # Create mock data as a pandas DataFrame
    return pd.DataFrame(
        {
            'y': [1.0, 2.0, 3.0],
            'z': [0, 1, 0],
            'w1': [0.5, 1.5, 2.5],
            'w2': [1.0, 2.0, 3.0],
        }
    )


@pytest.mark.skipif(not JAX_AVAILABLE, reason='JAX not available')
def test_initialization(mock_data):
    # Test the initialization of the DRGU class
    covariates = ['w1', 'w2']
    treatment = 'z'
    response = 'y'
    model = DRGU(mock_data, covariates, treatment, response)

    # Assertions
    assert model.w.shape == (3, 2), 'Covariates matrix shape is incorrect'
    assert model.z.shape == (3,), 'Treatment vector shape is incorrect'
    assert model.y.shape == (3,), 'Response vector shape is incorrect'
    assert 'delta' in model.theta, "Theta does not contain 'delta'"
    assert 'beta' in model.theta, "Theta does not contain 'beta'"
    assert 'gamma' in model.theta, "Theta does not contain 'gamma'"


@pytest.mark.skipif(not JAX_AVAILABLE, reason='JAX not available')
def test_fit(mock_data):
    # Test the fit method
    covariates = ['w1', 'w2']
    treatment = 'z'
    response = 'y'
    model = DRGU(mock_data, covariates, treatment, response)

    # Call the fit method
    model.fit()

    # Assertions
    assert hasattr(model, 'coefficients'), 'Model coefficients were not set'
    assert hasattr(model, 'variance_matrix'), 'Variance matrix was not set'
    assert model.coefficients.shape[0] == len(model.theta['delta']) + len(
        model.theta['beta']
    ) + len(model.theta['gamma']), 'Coefficients shape is incorrect'


@pytest.mark.skipif(not JAX_AVAILABLE, reason='JAX not available')
def test_summary(mock_data):
    # Test the summary method
    covariates = ['w1', 'w2']
    treatment = 'z'
    response = 'y'
    model = DRGU(mock_data, covariates, treatment, response)

    # Fit the model
    model.fit()

    # Generate the summary
    summary = model.summary()

    # Assertions
    assert isinstance(summary, pd.DataFrame), 'Summary is not a DataFrame'
    assert 'Coefficient' in summary.columns, "Summary missing 'Coefficient' column"
    assert 'Std_Error' in summary.columns, "Summary missing 'Std_Error' column"
    assert 'P_Value' in summary.columns, "Summary missing 'P_Value' column"
    assert summary.shape[0] == len(model.coefficients), 'Summary row count is incorrect'


@pytest.mark.skipif(not JAX_AVAILABLE, reason='JAX not available')
def test_regularization_with_lambda():
    """Test that JAX implementation correctly handles lamb > 0 and doesn't penalize delta."""
    import jax.numpy as jnp
    import numpy as np

    # Create larger dataset for better regularization effect
    np.random.seed(42)
    n = 100

    data = pd.DataFrame(
        {
            'y': np.random.randn(n) + 2.0,
            'z': np.random.randint(0, 2, n),
            'w1': np.random.randn(n) + 1.0,
            'w2': np.random.randn(n) - 1.0,
        }
    )

    covariates = ['w1', 'w2']
    treatment = 'z'
    response = 'y'

    # Test with small regularization (for numerical stability)
    model_small_reg = DRGU(data, covariates, treatment, response)
    model_small_reg.fit(lamb=0.01, max_iter=20, tol=1e-5, verbose=False)

    # Test with moderate regularization
    model_mod_reg = DRGU(data, covariates, treatment, response)
    model_mod_reg.fit(lamb=1.0, max_iter=20, tol=1e-5, verbose=False)

    # Test with large regularization
    model_large_reg = DRGU(data, covariates, treatment, response)
    model_large_reg.fit(lamb=10.0, max_iter=20, tol=1e-5, verbose=False)

    # Extract parameter norms (excluding delta)
    def param_norm_without_delta(theta):
        """Compute L2 norm of beta and gamma only (not delta)."""
        beta_gamma = jnp.concatenate([theta['beta'].flatten(), theta['gamma'].flatten()])
        return float(jnp.linalg.norm(beta_gamma))

    norm_small_reg = param_norm_without_delta(model_small_reg.theta)
    norm_mod_reg = param_norm_without_delta(model_mod_reg.theta)
    norm_large_reg = param_norm_without_delta(model_large_reg.theta)

    # Check that all norms are finite
    assert not jnp.isnan(norm_small_reg), 'Small regularization produced NaN'
    assert not jnp.isnan(norm_mod_reg), 'Moderate regularization produced NaN'
    assert not jnp.isnan(norm_large_reg), 'Large regularization produced NaN'

    # Assertions: Larger lambda should shrink beta/gamma parameters
    assert norm_mod_reg < norm_small_reg, (
        f'Moderate regularization should shrink beta/gamma: {norm_mod_reg} < {norm_small_reg}'
    )
    assert norm_large_reg < norm_mod_reg, (
        f'Larger lambda should shrink more: {norm_large_reg} < {norm_mod_reg}'
    )

    # Check that delta is NOT systematically shrunk by regularization
    # (it can change but shouldn't be systematically pushed toward zero)
    delta_small_reg = float(model_small_reg.theta['delta'][0])
    delta_mod_reg = float(model_mod_reg.theta['delta'][0])
    delta_large_reg = float(model_large_reg.theta['delta'][0])

    # Delta values should be similar (not shrinking toward zero like beta/gamma)
    # We allow some variation but it shouldn't shrink as much as beta/gamma
    delta_variation = max(
        abs(delta_small_reg - delta_mod_reg), abs(delta_mod_reg - delta_large_reg)
    )
    beta_gamma_shrinkage = norm_small_reg - norm_large_reg

    # Delta variation should be much smaller than beta/gamma shrinkage
    assert delta_variation < beta_gamma_shrinkage * 0.5, (
        f'Delta should not shrink as much as beta/gamma: '
        f'delta_variation={delta_variation}, beta_gamma_shrinkage={beta_gamma_shrinkage}'
    )

    print('JAX regularization test passed:')
    print(f'  norm_small_reg: {norm_small_reg:.4f}')
    print(f'  norm_mod_reg: {norm_mod_reg:.4f}')
    print(f'  norm_large_reg: {norm_large_reg:.4f}')
    print(f'  delta_small_reg: {delta_small_reg:.4f}')
    print(f'  delta_mod_reg: {delta_mod_reg:.4f}')
    print(f'  delta_large_reg: {delta_large_reg:.4f}')
