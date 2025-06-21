"""
Outlier handling for income estimation models.

This module contains transformers for handling outliers in the data.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EnhancedOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handles outliers in numerical features using a variety of strategies
    based on the specified `strategy_map`. It can cap outliers based on
    the Interquartile Range (IQR), percentiles, or a modified Z-score method
    (using Median Absolute Deviation). Optionally, it can create binary flags
    indicating the presence of outliers before capping.

    Parameters
    ----------
    strategy_map : dict, default=None
        A dictionary specifying the outlier handling strategy for each feature.
        Keys are feature names (str), and values are the strategy to use
        ('iqr', 'percentile', or 'zscore'). Features not in this map will
        not have outlier handling applied.

    iqr_params : dict, default={'multiplier': 1.5}
        Parameters for the 'iqr' strategy.
        'multiplier' (float): The factor multiplied by the IQR to determine
                              the upper and lower bounds (upper only is used here).

    percentile_params : dict, default={'upper_quantile': 0.99}
        Parameters for the 'percentile' strategy.
        'upper_quantile' (float): The upper percentile to use for capping.

    zscore_params : dict, default={'threshold': 3.0}
        Parameters for the 'zscore' strategy (modified Z-score using MAD).
        'threshold' (float): The number of Median Absolute Deviations from
                             the median to consider a point an outlier (upper bound).

    create_flags : bool, default=True
        Whether to create binary flag features (e.g., 'feature_outlier')
        indicating if a value was an outlier before capping.
    """
    def __init__(self, strategy_map=None, iqr_params=None, percentile_params=None,
                 zscore_params=None, create_flags=True):
        self.strategy_map = strategy_map or {}
        self.iqr_params = iqr_params or {'multiplier': 1.5}
        self.percentile_params = percentile_params or {'upper_quantile': 0.99}
        self.zscore_params = zscore_params or {'threshold': 3.0}
        self.create_flags = create_flags
        self.thresholds_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Calculates the upper capping thresholds for each feature based on the
        specified strategy in `strategy_map` using the training data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
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

            for col, strategy in self.strategy_map.items():
                if col in X.columns:
                    if strategy == 'iqr':
                        # IQR method (robust to extreme outliers)
                        q1 = X[col].quantile(0.25)
                        q3 = X[col].quantile(0.75)
                        iqr = q3 - q1
                        multiplier = self.iqr_params.get('multiplier', 1.5)
                        upper_bound = q3 + (multiplier * iqr)
                        self.thresholds_[col] = upper_bound

                    elif strategy == 'percentile':
                        # Percentile-based capping
                        upper_quantile = self.percentile_params.get('upper_quantile', 0.99)
                        upper_bound = X[col].quantile(upper_quantile)
                        self.thresholds_[col] = upper_bound

                    elif strategy == 'zscore':
                        # Modified Z-score method (using median and MAD)
                        median = X[col].median()
                        # MAD = median absolute deviation
                        mad = (X[col] - median).abs().median() * 1.4826  # Scale factor for normal distribution
                        threshold = self.zscore_params.get('threshold', 3.0)
                        upper_bound = median + (threshold * mad)
                        self.thresholds_[col] = upper_bound

        return self

    def transform(self, X):
        """
        Caps the outlier values in the specified features of the input data
        using the thresholds calculated during the `fit` stage. Optionally,
        creates binary outlier flags before capping.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Data to transform. If a numpy array, column names from the training
            data (if available) will be used.

        Returns
        -------
        X_transformed : pandas.DataFrame of shape (n_samples, n_features + n_flag_features)
            The transformed DataFrame with outliers capped. If `create_flags`
            is True, new binary flag columns are added for each capped feature.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        X_transformed = X.copy()

        # Apply thresholds and create flags
        for col, threshold in self.thresholds_.items():
            if col in X_transformed.columns:
                # Create flag before capping if requested
                if self.create_flags:
                    X_transformed[f"{col}_outlier"] = (X_transformed[col] > threshold).astype(int)

                # Apply capping
                X_transformed[col] = X_transformed[col].clip(upper=threshold)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names after outlier handling.

        If `create_flags` is True, new binary features with the suffix '_outlier'
        are added for each feature where outlier capping was applied.

        Parameters
        ----------
        input_features : list of str, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : list of str
            List of output feature names, including the original features and
            the newly created '_outlier' flag features (if `create_flags` is True).
        """
        output_features = list(self.feature_names_in_) if self.feature_names_in_ else list(input_features) if input_features else []
        if self.create_flags:
            for col in self.thresholds_.keys():
                if col in output_features:
                    output_features.append(f"{col}_outlier")
        return output_features
