"""
Financial inconsistency filtering for income estimation models.

This module contains a transformer that filters out training data rows
exhibiting extreme financial inconsistencies based on predefined rules.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialInconsistencyFilter(BaseEstimator, TransformerMixin):
    """
    Filters out training data rows exhibiting extreme financial inconsistencies
    based on predefined rules. Test data is passed through unchanged.

    Parameters
    ----------
    is_training : bool, default=True
        A flag indicating whether the transformer is being applied to training
        data. If False, no filtering will be performed.
    """
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.feature_names_in_ = None
        self.n_filtered_rows_ = 0
        self.filtered_indices_ = []

    def fit(self, X, y=None):
        """
        Identifies financially inconsistent cases in the training data based on
        a set of heuristic rules involving financial ratios and the target variable.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Training data to be filtered.

        y : array-like of shape (n_samples,), default=None
            Target values corresponding to the training data. Used in the
            inconsistency rules.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.feature_names_in_ = X.columns.tolist()

        # Only apply filtering to training data
        if not self.is_training:
            return self

        # Store original data shape for reporting
        original_shape = X.shape

        # Calculate financial ratios for filtering
        data = X.copy()

        if y is not None:
            data['target'] = y

        if 'target' in data.columns:
            # Calculate all ratios
            if 'Inc_in' in data.columns:
                data['Inc_in_ratio'] = data['Inc_in'] / (data['target'] + 1)
            if 'Payments' in data.columns:
                data['Payments_ratio'] = data['Payments'] / (data['target'] + 1)
            if 'Inc_Past' in data.columns:
                data['Inc_Past_ratio'] = data['Inc_Past'] / (data['target'] + 1)
            if 'Transfers_out' in data.columns:
                data['Transfers_out_ratio'] = data['Transfers_out'] / (data['target'] + 1)
            if 'Turnover' in data.columns:
                data['Turnover_ratio'] = data['Turnover'] / (data['target'] + 1)
            if 'Liab_Tot' in data.columns:
                data['Liab_Tot_ratio'] = data['Liab_Tot'] / (data['target'] + 1)

            # Identify inconsistent cases
            inconsistent_mask = (
                # === Existing Rules ===

                # Rule 1: Extreme Inc_in vs. target
                (data.get('Inc_in_ratio', pd.Series([0])) > 15) |

                # Rule 2: Extreme Payments vs. target
                (data.get('Payments_ratio', pd.Series([0])) > 300) |

                # Rule 3: Extreme Transfers_out vs. target
                (data.get('Transfers_out_ratio', pd.Series([0])) > 40) |

                # Rule 4: Multiple moderate income inconsistencies
                ((data.get('Inc_in_ratio', pd.Series([0])) > 5) &
                 (data.get('Inc_Past_ratio', pd.Series([0])) > 5) &
                 (data['target'] < 500)) |

                # Rule 5: High liabilities with low income
                ((data.get('Liab_Tot_ratio', pd.Series([0])) > 100) & (data['target'] < 500)) |

                # Rule 6: Past income too high for current low income
                ((data.get('Inc_Past_ratio', pd.Series([0])) > 15) & (data['target'] < 1000)) |

                # Rule 7: High incoming inflow & turnover, but low target
                ((data.get('Inc_in_ratio', pd.Series([0])) > 10) &
                 (data.get('Turnover_ratio', pd.Series([0])) > 10) &
                 (data['target'] < 800)) |

                # === New Rules ===

                # Rule 8: Payments disproportionately high compared to target
                (((data['Payments'] / (data['target'] + 1)) > 40) & (data['target'] < 800)) |

                # Rule 9: Total liabilities + payments highly disproportionate to target
                (((data['Liab_Tot'] + data['Payments']) / (data['target'] + 1)) > 150) & (data['target'] < 500)
            )


            # Store the indices of filtered rows for reporting
            self.filtered_indices_ = list(data[inconsistent_mask].index)
            self.n_filtered_rows_ = len(self.filtered_indices_)

            # Report filtering results
            if self.n_filtered_rows_ > 0:
                filter_percent = (self.n_filtered_rows_ / len(data)) * 100
                print(f"Financial inconsistency filter: Removed {self.n_filtered_rows_} rows ({filter_percent:.2f}%) from training data")

                # Analyze characteristics of filtered cases
                filtered_data = data[inconsistent_mask]
                if len(filtered_data) > 0:
                    print(f"  Average target value in filtered cases: {filtered_data['target'].mean():.2f}")

                    # Report top triggers
                    trigger_counts = {
                        "Inc_in_ratio > 15": (filtered_data.get('Inc_in_ratio', pd.Series(0)) > 15).sum(),
                        "Payments_ratio > 300": (filtered_data.get('Payments_ratio', pd.Series(0)) > 300).sum(),
                        "Transfers_out_ratio > 40": (filtered_data.get('Transfers_out_ratio', pd.Series(0)) > 40).sum(),
                        "Inc_in & Inc_Past > 5 with target < 500": (
                            ((filtered_data.get('Inc_in_ratio', pd.Series(0)) > 5) &
                            (filtered_data.get('Inc_Past_ratio', pd.Series(0)) > 5) &
                            (filtered_data['target'] < 500)).sum()
                        ),
                        "Liab_Tot_ratio > 100 with target < 500": (
                            ((filtered_data.get('Liab_Tot_ratio', pd.Series(0)) > 100) &
                            (filtered_data['target'] < 500)).sum()
                        ),
                        "Inc_Past_ratio > 15 with target < 1000": (
                            ((filtered_data.get('Inc_Past_ratio', pd.Series(0)) > 15) &
                            (filtered_data['target'] < 1000)).sum()
                        ),
                        "Inc_in_ratio > 10 and Turnover_ratio > 10 with target < 800": (
                            ((filtered_data.get('Inc_in_ratio', pd.Series(0)) > 10) &
                            (filtered_data.get('Turnover_ratio', pd.Series(0)) > 10) &
                            (filtered_data['target'] < 800)).sum()
                        ),
                        "Payments > 40x target & target < 800": (
                            ((filtered_data['Payments'] / (filtered_data['target'] + 1)) > 40).sum()
                        ),
                        "Liabilities + Payments > 150x target & target < 500": (
                            (((filtered_data['Liab_Tot'] + filtered_data['Payments']) / (filtered_data['target'] + 1)) > 150).sum()
                        )
                    }

                    top_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
                    top_triggers = [f"{k}: {v} cases" for k, v in top_triggers if v > 0]

                    if top_triggers:
                        print("  Top triggers: " + "; ".join(top_triggers[:5]))

        return self

    def transform(self, X):
        """
        Removes the identified financially inconsistent cases from the training data.
        Test data is returned without modification.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        X_filtered : pandas.DataFrame of shape (n_samples - n_filtered, n_features)
            Filtered data with inconsistent cases removed (only if applied to
            training data). If applied to test data, the original DataFrame is returned.
        """
        if not self.is_training:
            # Don't filter test data
            return X

        # Filter out inconsistent cases from training data
        if self.filtered_indices_:
            return X.drop(index=self.filtered_indices_)
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names.

        Parameters
        ----------
        input_features : list of str, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : list of str
            The same as the input feature names, as this transformer does not
            change the set of features.
        """
        return self.feature_names_in_
