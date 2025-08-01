from abc import ABC, abstractmethod
import numpy as np

class EstimatingEquation(ABC):
    """
    Abstract base class for models based on estimating equations.
    Provides a structure for point estimation and variance estimation.
    """
    def __init__(self, data, covariates, treatment, response):
        """
        Initialize the model with data, covariates, treatment, and response.

        :param data: np.ndarray or pandas.DataFrame, the dataset
        :param covariates: list, names of covariate columns
        :param treatment: str, name of the treatment variable
        :param response: str, name of the response variable
        """
        self.data = data
        self.covariates = covariates
        self.treatment = treatment
        self.response = response
        self.coefficients = None
        self.variance_matrix = None

    @abstractmethod
    def fit(self):
        """
        Fit the model to the data. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def summary(self):
        """
        Return a summary of the model fit. Must be implemented by subclasses.
        """
        pass

    def get_point_estimates(self):
        """
        Return the point estimates (coefficients).
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        return self.coefficients

    def get_variance_estimates(self):
        """
        Return the variance-covariance matrix of the estimates.
        """
        if self.variance_matrix is None:
            raise ValueError("Model has not been fitted yet.")
        return self.variance_matrix
