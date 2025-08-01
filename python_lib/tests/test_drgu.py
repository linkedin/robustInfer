import pytest
import jax.numpy as jnp
import pandas as pd
from robustinfer.drgu import DRGU

@pytest.fixture
def mock_data():
    # Create mock data as a pandas DataFrame
    return pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "z": [0, 1, 0],
        "w1": [0.5, 1.5, 2.5],
        "w2": [1.0, 2.0, 3.0]
    })

def test_initialization(mock_data):
    # Test the initialization of the DRGU class
    covariates = ["w1", "w2"]
    treatment = "z"
    response = "y"
    model = DRGU(mock_data, covariates, treatment, response)

    # Assertions
    assert model.w.shape == (3, 2), "Covariates matrix shape is incorrect"
    assert model.z.shape == (3,), "Treatment vector shape is incorrect"
    assert model.y.shape == (3,), "Response vector shape is incorrect"
    assert "delta" in model.theta, "Theta does not contain 'delta'"
    assert "beta" in model.theta, "Theta does not contain 'beta'"
    assert "gamma" in model.theta, "Theta does not contain 'gamma'"

def test_fit(mock_data):
    # Test the fit method
    covariates = ["w1", "w2"]
    treatment = "z"
    response = "y"
    model = DRGU(mock_data, covariates, treatment, response)

    # Call the fit method
    model.fit()

    # Assertions
    assert hasattr(model, "coefficients"), "Model coefficients were not set"
    assert hasattr(model, "variance_matrix"), "Variance matrix was not set"
    assert model.coefficients.shape[0] == len(model.theta["delta"]) + \
           len(model.theta["beta"]) + len(model.theta["gamma"]), \
           "Coefficients shape is incorrect"

def test_summary(mock_data):
    # Test the summary method
    covariates = ["w1", "w2"]
    treatment = "z"
    response = "y"
    model = DRGU(mock_data, covariates, treatment, response)

    # Fit the model
    model.fit()

    # Generate the summary
    summary = model.summary()

    # Assertions
    assert isinstance(summary, pd.DataFrame), "Summary is not a DataFrame"
    assert "Coefficient" in summary.columns, "Summary missing 'Coefficient' column"
    assert "Std_Error" in summary.columns, "Summary missing 'Std_Error' column"
    assert "P_Value" in summary.columns, "Summary missing 'P_Value' column"
    assert summary.shape[0] == len(model.coefficients), "Summary row count is incorrect"

