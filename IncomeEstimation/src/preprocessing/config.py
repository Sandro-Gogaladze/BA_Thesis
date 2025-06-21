"""
Configuration settings for the preprocessing pipeline.

This module contains the standard configuration for preprocessing components
to ensure consistency between development and production environments.
"""

# # === Zero Handling Configuration ===

# # Features where zero is a valid state (not requiring replacement)
# ACCOUNT_STATUS_FEATURES = [
#     "account_open",  # If present
# ]

# # Features where zeros have a strong relationship with the target
# STRONG_TARGET_IMPACT_FEATURES = [
#     "Turnover",
#     "Inc_Past",
#     "Inc_6M",
#     "Salary",
#     "Inc_in",
#     "Payments",
#     "Tot_in",
#     "Trn_max"
# ]

# # Features where zeros have a moderate relationship with the target
# MODERATE_TARGET_IMPACT_FEATURES = [
#     "Transfers_in",
#     "Transfers_out",
#     "Inc_Past_avg",
#     "Inc_Past_max"
# ]

# # Features where zeros have minimal relationship with the target
# MINIMAL_TARGET_IMPACT_FEATURES = [
#     "Acc_age",
#     "Balance",
#     "Loan",
#     "Loan_Cnt",
#     "Payment_L",
#     "Min_transfer_In"
# ]

# # Features with very low zero prevalence (<1%)
# LOW_PREVALENCE_FEATURES = [
#     "Liab_Tot"
# ]

# # Number of target quantiles to use for zero handling
# N_QUANTILES = 5


# === Outlier Handling Configuration ===

# Mapping of features to outlier handling strategy
OUTLIER_STRATEGY_MAP = {
    # Features with extreme skew and long upper tails → use IQR (aggressive)
    "Liab_Tot": "iqr",       # Skew: 2.52, Max: 355,931
    "Trn_max": "iqr",        # Skew: 2.86, Max: 584,598
    "Tot_in": "iqr",         # Skew: 3.64, Max: 743,784
    "Payments": "iqr",       # Skew: 3.21, Extremely high values

    # Features with high but manageable skew → use Z-score (balanced)
    "Inc_in": "zscore",         # Skew: 2.43
    "Transfers_out": "zscore",  # Skew: 3.34
    "Acct_Trms": "zscore",      # Skew: 3.52
    "Balance": "zscore",        # Skew: 3.33 (may include negatives)
    "Bal_Cur": "zscore",        # Skew: 3.92
    "Transactions": "zscore",   # Skew: 2.69

    # Moderately skewed or business-critical → use Percentile (conservative)
    "Inc_Past": "percentile",       # Skew: 1.86
    "Min_transfer_In": "percentile",# Skew: 1.98
    "Turnover": "percentile",       # Skew: 2.35
    "Salary": "percentile",         # Skew: 1.94
    "Inc_6M": "percentile",         # Skew: 2.30
    "Loan": "percentile",           # Skew: 3.24
    "Inc_Past_avg": "percentile",   # Skew: 2.03
    "Inc_Past_max": "percentile"    # Skew: 1.88
}

# Parameters for IQR-based outlier handling
IQR_PARAMS = {
    "multiplier": 1.5  # Standard 1.5 x IQR
}

# Parameters for percentile-based outlier handling
PERCENTILE_PARAMS = {
    "upper_quantile": 0.99  # 99th percentile
}

# Parameters for Z-score based outlier handling
ZSCORE_PARAMS = {
    "threshold": 3.0  # 3 MADs from median
}
