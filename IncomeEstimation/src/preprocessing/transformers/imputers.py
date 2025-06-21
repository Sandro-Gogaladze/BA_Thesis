"""
Missing value imputation for income estimation models.

This module contains transformers for handling missing values in the data.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using K-Nearest Neighbors algorithm while preserving
    the pandas DataFrame structure.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for imputation.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Fits the KNNImputer to the input data.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), default=None
            Target values. Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        self.imputer.fit(X)
        return self

    def transform(self, X):
        """
        Imputes missing values in X using the fitted KNNImputer.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Data to impute. If a numpy array, column names from the training
            data (if available) will be used in the output DataFrame.

        Returns
        -------
        X_imputed : pandas.DataFrame of shape (n_samples, n_features)
            The imputed data as a pandas DataFrame, preserving the original
            index and column names.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=self.feature_names_in_, index=X.index)
