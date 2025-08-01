import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm

from .ee import EstimatingEquation
from .utils import data_pairwise, compute_B_U_Sig, compute_delta, update_theta, get_theta_init

class DRGU(EstimatingEquation):
    """
    Doubly Robust Generalized U model.
    This class extends the EstimatingEquation class to implement a doubly robust estimator for Doubly Robust U.
    """
    def __init__(self, data, covariates, treatment, response):
        """
        Initialize the DRGU model with data, covariates, treatment, and response.

        :param data: np.ndarray or pandas.DataFrame, the dataset
        :param covariates: list, names of covariate columns
        :param treatment: str, name of the treatment variable
        :param response: str, name of the response variable
        """
        super().__init__(data, covariates, treatment, response)
        self.w = self.data[self.covariates].values
        self.z = self.data[self.treatment].values
        self.y = self.data[self.response].values
        self.theta = {
            "delta": jnp.array([0.5]),
            "beta": jnp.array([0.0] * (len(self.covariates)+1)),
            "gamma": jnp.array([0.0] * (2*len(self.covariates)+1))
        }

    def fit(self):
        """
        Fit the DRGU model to the data.
        This method should implement the logic for fitting the model.
        """
        # Prepare data for pairwise computation
        data = data_pairwise(self.y, self.z, self.w)
        
        # Initialize parameters
        theta_init = get_theta_init(data, self.z)
        
        # Solve the estimating equation
        theta, J, Var = self._solve_ugee(data, theta_init)
        
        # Store results
        self.theta = theta
        self.coefficients = jnp.concatenate([v for v in theta.values()])
        self.variance_matrix = Var* (1.0/self.w.shape[0])

    def _solve_ugee(self, data, theta_init, max_iter=100, tol=1e-6, lamb=0.0, option="fisher", verbose=True):
        V_inv = jnp.eye(3)
        theta = {k: v.copy() for k, v in theta_init.items()}
        for i in range(max_iter):
            step, J = compute_delta(theta, V_inv, data, lamb, option)
            # jax.debug.print("Step {i}: {x}", i=i, x=step)
            if i % 10 == 0 and verbose:
                jax.debug.print("Step {i} gradient norm: {x}", i=i, x=jnp.linalg.norm(step))
            theta = update_theta(theta, step)
            if jnp.linalg.norm(step) < tol:
                if verbose:
                    print(f"converged after {i} iterations")
                break
        if i == max_iter-1 and verbose:
            print(f"did not converge, norm step = {jnp.linalg.norm(step)}")
        B, U, Sig = compute_B_U_Sig(theta, V_inv, data)
        B_inv = jnp.linalg.inv(B)
        Var = 4 * B_inv @ Sig @ B_inv.T
        return theta, J, Var
    
    def summary(self):
        """
        Generate a summary of the model fit, including coefficients, standard errors, z-scores, and p-values.
        """
        # Compute standard errors
        standard_errors = jnp.sqrt(jnp.diag(self.variance_matrix))

        # Compute z-scores
        null_hypothesis = jnp.zeros_like(self.coefficients).at[0].set(0.5)
        z_scores = (self.coefficients - null_hypothesis) / standard_errors

        # Compute p-values
        p_values = 2 * (1 - norm.cdf(jnp.abs(z_scores)))

        # Create a summary table
        # Generate row names
        row_names = ["delta"] + \
            [f"beta_{i}" for i in range(len(self.theta["beta"]))] + \
                [f"gamma_{i}" for i in range(len(self.theta["gamma"]))]
        summary = pd.DataFrame({
            "Names": row_names,
            "Coefficient": self.coefficients,
            "Null_Hypothesis": null_hypothesis,
            "Std_Error": standard_errors,
            "Z_Score": z_scores,
            "P_Value": p_values
        })

        return summary
