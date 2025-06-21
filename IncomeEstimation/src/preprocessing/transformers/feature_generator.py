"""
Feature generation for income estimation models.

This module contains transformers for generating new features from existing ones.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates new features, primarily financial ratios, from the existing features
    in the dataset.
    """
    def fit(self, X, y=None):
        """
        Fits the transformer to the input data. In this case, it simply stores
        the input feature names if the input is a pandas DataFrame.

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
        self.feature_names_in_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        """
        Generates new financial ratio features and adds them to the DataFrame.
        Handles potential division by zero by adding a small epsilon.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Data to transform. If a numpy array, column names from the training
            data (if available) will be used.

        Returns
        -------
        X_transformed : pandas.DataFrame of shape (n_samples, n_features + n_new_features)
            The DataFrame with the newly generated features appended.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_transformed = X.copy()
        eps = 1e-6  # Small epsilon to avoid division by zero

        # Create financial ratio features
        if {'Loan', 'Inc_Past'}.issubset(X_transformed.columns):
            X_transformed['Loan_to_Income'] = X_transformed['Loan'] / (X_transformed['Inc_Past'] + eps)
        if {'Balance', 'Liab_Tot'}.issubset(X_transformed.columns):
            X_transformed['Balance_to_Liab'] = X_transformed['Balance'] / (X_transformed['Liab_Tot'] + eps)
        if {'Inc_6M', 'Inc_Past'}.issubset(X_transformed.columns):
            X_transformed['Income_Growth'] = (X_transformed['Inc_6M'] / 6) / (X_transformed['Inc_Past'] / 12 + eps) - 1
        if {'Tot_in', 'Liab_Tot'}.issubset(X_transformed.columns):
            X_transformed['Income_to_Liability'] = X_transformed['Tot_in'] / (X_transformed['Liab_Tot'] + eps)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names after feature generation.

        Parameters
        ----------
        input_features : list of str, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : list of str
            List of output feature names, including the original features and
            the newly generated features.
        """
        output_features = list(self.feature_names_in_) if self.feature_names_in_ else list(input_features) if input_features else []
        if {'Loan', 'Inc_Past'}.issubset(output_features):
            output_features.append('Loan_to_Income')
        if {'Balance', 'Liab_Tot'}.issubset(output_features):
            output_features.append('Balance_to_Liab')
        if {'Inc_6M', 'Inc_Past'}.issubset(output_features):
            output_features.append('Income_Growth')
        if {'Tot_in', 'Liab_Tot'}.issubset(output_features):
            output_features.append('Income_to_Liability')
        return output_features
