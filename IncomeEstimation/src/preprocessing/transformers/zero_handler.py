"""
Zero handling for income estimation models.

This module contains transformers for handling zero values in financial data.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EnhancedContextAwareZeroHandler(BaseEstimator, TransformerMixin):
    """
    Handles zero values in financial features based on their predefined
    categories and their relationship with the target variable.

    This transformer creates binary flags indicating the presence of zero
    values and replaces these zeros with meaningful, small non-zero values
    derived from the distribution of non-zero values within each feature category.

    Parameters
    ----------
    account_status_features : list of str
        Features where zero is a valid and meaningful state (e.g., 'account_open').
        Zeros in these features will be preserved.

    strong_target_impact_features : list of str
        Features where zero values are hypothesized to have a strong relationship
        with the target variable. Zeros will be flagged and replaced with 50%
        of the 10th percentile of the non-zero values per target quantile.

    moderate_target_impact_features : list of str
        Features where zero values are hypothesized to have a moderate relationship
        with the target variable. Zeros will be flagged and replaced with 20%
        of the 10th percentile of the non-zero values per target quantile.

    minimal_target_impact_features : list of str
        Features where zero values are hypothesized to have a minimal relationship
        with the target variable. Zeros will be flagged and replaced with a
        fixed small value (50.0).

    low_prevalence_features : list of str
        Features where zero values are very rare (<1% prevalence). Zeros will be
        replaced with a small constant value (1.0) without creating a flag.

    income_column : str, default=None
        Column name to use for stratification (if different from target).
        If None, the target variable will be used for stratification.

    n_quantiles : int, default=5
        Number of quantiles to use for target stratification when calculating
        replacement values for zeros.
    """
    def __init__(self,
                 account_status_features,
                 strong_target_impact_features,
                 moderate_target_impact_features,
                 minimal_target_impact_features,
                 low_prevalence_features,
                 income_column=None,
                 n_quantiles=5):
        self.account_status_features = account_status_features
        self.strong_target_impact_features = strong_target_impact_features
        self.moderate_target_impact_features = moderate_target_impact_features
        self.minimal_target_impact_features = minimal_target_impact_features
        self.low_prevalence_features = low_prevalence_features
        self.income_column = income_column
        self.n_quantiles = n_quantiles

        # Storage for fitted parameters
        self.minimums_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Computes the minimum replacement values for zero handling based on the
        distribution of non-zero values in the training data, stratified by
        target quantiles.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Training data to fit the transformer.

        y : array-like of shape (n_samples,), default=None
            Target values used for stratification when computing quantile-specific
            replacement values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Store feature names
        self.feature_names_in_ = X.columns.tolist()
        self.minimums_ = {}

        # Use target for quantile stratification if provided
        if y is not None:
            # Create quantile bins from the target variable
            quantile_bins = pd.qcut(y, self.n_quantiles, labels=False, duplicates='drop')
            quantile_bins = quantile_bins.loc[X.index] 

            # Calculate per-quantile minimums for each feature category
            for feature_list, factor in [
                (self.strong_target_impact_features, 0.5),  # 50% of 10th percentile
                (self.moderate_target_impact_features, 0.2)  # 20% of 10th percentile
            ]:
                for feature in feature_list:
                    if feature in X.columns:
                        self.minimums_[feature] = {}
                        
                        # Calculate separate minimum for each target quantile
                        for q in range(self.n_quantiles):
                            # Get subset of data for this quantile
                            subset = X.loc[quantile_bins[quantile_bins == q].index]
                            
                            # Get non-zero values for this feature in this quantile
                            non_zero = subset[subset[feature] > 0][feature]
                            
                            # Calculate replacement value (minimum of 1.0)
                            if len(non_zero) > 0:
                                min_value = non_zero.quantile(0.1) * factor
                                value = max(min_value, 1.0)
                            else:
                                value = 1.0
                                
                            self.minimums_[feature][q] = value
        else:
            # Without target, use global calculation (fallback)
            for feature_list, factor in [
                (self.strong_target_impact_features, 0.5),
                (self.moderate_target_impact_features, 0.2)
            ]:
                for feature in feature_list:
                    if feature in X.columns:
                        # Get non-zero values for this feature
                        non_zero = X[X[feature] > 0][feature]
                        
                        # Calculate replacement value (minimum of 1.0)
                        if len(non_zero) > 0:
                            min_value = non_zero.quantile(0.1) * factor
                            value = max(min_value, 1.0)
                        else:
                            value = 1.0
                            
                        # Store as single quantile for compatibility
                        self.minimums_[feature] = {0: value}

        return self

    def transform(self, X, y=None):
        """
        Transforms the data by handling zero values based on the predefined
        feature categories and target quantiles. Creates binary flags for zero values 
        in relevant categories and replaces these zeros with the calculated 
        quantile-specific minimums or predefined constants.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Data to transform.

        y : array-like of shape (n_samples,), default=None
            Target values used to determine which quantile-specific replacement
            value to use for each sample.

        Returns
        -------
        X_transformed : pandas.DataFrame of shape (n_samples, n_features + n_flag_features)
            Transformed data with zero values handled. New binary flag columns
            are added for features in the strong, moderate, and minimal impact
            categories.
        """
        X_transformed = X.copy()

        # Create quantile bins if target is provided, otherwise use a single bin
        if y is not None and self.n_quantiles > 1:
            quantile_bins = pd.qcut(y, self.n_quantiles, labels=False, duplicates='drop')
            quantile_bins = quantile_bins.loc[X.index]
            quantile_bins.index = X.index
        else:
            # Default all rows to quantile 0 if no target provided
            quantile_bins = pd.Series([0] * len(X), index=X.index)

        # Create flags for zeros for relevant categories
        for feature in (
            self.strong_target_impact_features +
            self.moderate_target_impact_features +
            self.minimal_target_impact_features
        ):
            if feature in X_transformed.columns:
                X_transformed[f"{feature}_zero"] = (X_transformed[feature] == 0).astype(int)

        # Replace zeros with quantile-specific minimums for strong and moderate impact features
        for feature in self.strong_target_impact_features + self.moderate_target_impact_features:
            if feature in X_transformed.columns and feature in self.minimums_:
                for q in range(self.n_quantiles):
                    # Create mask for zeros in this quantile
                    zero_mask = (X_transformed[feature] == 0) & (quantile_bins == q)
                    
                    # Apply replacement if any zeros found
                    if zero_mask.sum() > 0:
                        # Get replacement value for this feature and quantile (default to 1.0)
                        value = self.minimums_[feature].get(q, 1.0)
                        X_transformed.loc[zero_mask, feature] = value

        # Category 4: Minimal target impact features: Small global minimum of 50
        for feature in self.minimal_target_impact_features:
            if feature in X_transformed.columns:
                zero_mask = X_transformed[feature] == 0
                if zero_mask.sum() > 0:
                    X_transformed.loc[zero_mask, feature] = 50.0  # Fixed small value as specified

        # Category 5: Low prevalence features: Constant minimum of 1.0
        for feature in self.low_prevalence_features:
            if feature in X_transformed.columns:
                zero_mask = X_transformed[feature] == 0
                if zero_mask.sum() > 0:
                    X_transformed.loc[zero_mask, feature] = 1.0  # Small constant

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names after transformation.

        For features in 'strong_target_impact_features',
        'moderate_target_impact_features', and 'minimal_target_impact_features',
        a new binary feature with the suffix '_zero' is added.

        Parameters
        ----------
        input_features : list of str, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : list of str
            List of output feature names, including the original features and
            the newly created '_zero' flag features.
        """
        output_features = list(self.feature_names_in_)
        for feature in self.strong_target_impact_features:
            if feature in self.feature_names_in_:
                output_features.append(f"{feature}_zero")
        for feature in self.moderate_target_impact_features:
            if feature in self.feature_names_in_:
                output_features.append(f"{feature}_zero")
        for feature in self.minimal_target_impact_features:
            if feature in self.feature_names_in_:
                output_features.append(f"{feature}_zero")
        return output_features
