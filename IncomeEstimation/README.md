# IncomeEstimation: Interactive Repository Implementation

This directory contains the structured implementation of income estimation models for the Georgian banking sector. It provides an interactive, reproducible codebase that implements and compares four different modeling approaches: baseline XGBoost, post-hoc quantile calibration, and two pre-hoc regulation-aware models.

## Overview

The implementation evaluates different strategies for balancing predictive accuracy with regulatory compliance, specifically addressing the National Bank of Georgia's requirements that income predictions should not exceed actual income by more than 200 GEL or 20% (whichever is greater).

### Modeling Approaches

* **Baseline**
  * **XGBoost** (`baseline/xgboost/`): Strong predictive accuracy but higher regulatory violations

* **Post-hoc**
  * **Quantile Calibration** (`posthoc/quantile/`): Significant improvement in compliance with minimal accuracy loss

* **Pre-hoc**
  * **Huber + Threshold** (`prehoc/huber_threshold/`): Balanced approach with built-in regulatory awareness
  * **Segment-Aware** (`prehoc/segment_aware/`): Best overall performance with balanced accuracy and compliance across all income segments

## Quick Start

### Prerequisites

```bash
# From the root directory - install dependencies
pip install -e .
```

### Complete Pipeline Execution

Run the entire workflow for all models with a single command:

```bash
make all
```

This executes:
1. Data preprocessing
2. Training, inference, and evaluation of all 4 models
3. Comprehensive visualization generation
4. Cleaning of any previous outputs and intermediate files

### Individual Model Execution

```bash
# XGBoost baseline
make baseline/xgboost

# Post-hoc quantile calibration  
make posthoc/quantile

# Pre-hoc Huber + threshold penalty
make prehoc/huber_threshold

# Pre-hoc segment-aware model
make prehoc/segment_aware

# View all available commands
make help
```

## Detailed Project Structure

```
IncomeEstimation/
├── src/                                     # Core source code modules
│   ├── preprocessing/                       # Advanced data preprocessing pipeline  
│   │   ├── config.py                       # Preprocessing configuration
│   │   ├── loader.py                       # Data loading utilities
│   │   ├── preprocessor.py                 # Main preprocessing logic
│   │   └── transformers/                   # Custom feature transformations
│   ├── train/                              # Model training infrastructure
│   │   └── train.py                        # Universal training script
│   ├── inference/                          # Prediction and inference
│   │   └── predict.py                      # Universal prediction script
│   ├── evaluation/                         # Model evaluation and metrics
│   │   └── evaluate.py                     # Comprehensive evaluation framework
│   ├── toolkit/                            # Shared utilities and tools
│   │   ├── model_evaluator.py             # Model evaluation toolkit
│   │   ├── threshold_utils.py              # Regulatory threshold calculations
│   │   ├── visualization_utils.py          # Plotting and visualization
│   │   └── theme.py                        # Consistent visual styling
│   └── utils/                              # Common utilities
│       ├── logging.py                      # Logging configuration
│       └── paths.py                        # Path management
├── baseline/                               # Baseline models
│   └── xgboost/                           # Standard XGBoost implementation
│       ├── config/model_config.yaml       # Hyperparameters and settings
│       ├── model/xgboost_model.py         # Model implementation
│       └── results/                       # Training and evaluation outputs
├── posthoc/                               # Post-hoc calibration approaches
│   └── quantile/                          # Quantile regression calibration
│       ├── config/model_config.yaml       # Quantile-specific configuration
│       ├── model/quantile_model.py        # Quantile calibration implementation
│       └── results/                       # Calibration results and metrics
├── prehoc/                                # Pre-hoc regulation-aware models
│   ├── huber_threshold/                   # Huber loss + threshold penalty
│   │   ├── config/model_config.yaml       # Custom objective configuration
│   │   ├── model/huber_threshold_model.py # Custom objective implementation
│   │   └── results/                       # Training and evaluation results
│   └── segment_aware/                     # Segment-aware loss adjustment
│       ├── config/model_config.yaml       # Segment-aware configuration
│       ├── model/segment_aware_model.py   # Advanced regulatory model
│       └── results/                       # Performance and compliance metrics
├── cleanup.py                             # Model output cleanup utility
├── Makefile                              # Automated workflow commands
└── README.md                             # This file
```

**Note:** Dependencies are managed via `pyproject.toml` in the root directory. Use `pip install -e .` to install all required dependencies for the repository.

### Data Directory Structure

The preprocessing pipeline expects data in the following structure (relative to the repository root):

```
../data/
├── raw/
│   └── income_data.xlsx                   # Original dataset from BOG
└── processed/                             # Generated by preprocessing
    ├── train.csv                          # Training data (80%)
    └── test.csv                           # Test data (20%)
```

## Consistency and Reproducibility

This implementation ensures **complete consistency** with the research notebook (`../Notebooks/notebook.ipynb`):

- **Identical Preprocessing**: Uses the exact same KNN imputation, outlier capping, and feature engineering logic
- **Identical Hyperparameters**: All models use the same optimized parameters derived from the notebook analysis  
- **Identical Evaluation**: Same metrics, threshold calculations, and visualization approaches
- **Reproducible Results**: Fixed random seeds (42) ensure deterministic outputs across runs

The structured codebase produces the **exact same numerical results** as the notebook.ipynb, providing confidence in research validity and reproducibility.

## Data Preprocessing Pipeline

The preprocessing pipeline implements the research methodology with the following steps:

### 1. Missing Value Imputation
- **Method**: K-Nearest Neighbors (KNN) imputation with k=5
- **Strategy**: Preserves feature relationships better than simple mean/median imputation
- **Features**: Applied to all numerical features with missing values

### 2. Outlier Handling  
- **Feature-Specific Strategies**: Different methods based on feature distribution characteristics:
  * **IQR Method**: For extremely skewed features (e.g., Liab_Tot, Trn_max) - caps at Q3 + 1.5×IQR
  * **Z-Score Method**: For moderately skewed features (e.g., Inc_in, Balance) - caps at median + 3×MAD
  * **Percentile Method**: For business-critical features (e.g., Salary, Inc_Past) - caps at 99th percentile
- **Conservative Approach**: Values above thresholds are capped rather than removed
- **Rationale**: Preserves data volume while mitigating the impact of extreme outliers

### 3. Feature Engineering
- **Financial Ratios**: Creates meaningful derived features (debt-to-income, savings ratios)
- **Interaction Terms**: Captures relationships between related financial variables
- **Scaling**: Standard normalization for numerical stability

### Running Preprocessing

Preprocessing is automatically included in the full pipeline (`make all`) but can be run independently:

```bash
# Run preprocessing only
make preprocessing

# Or run directly
python -m IncomeEstimation.src.preprocessing.preprocessor
```

## Model Performance and Results

Results are automatically saved in each model's `results/` directory:

### Output Structure
```
{model}/results/
├── models/                    # Trained model files (.joblib)
├── metrics/                   # Performance metrics (.json)
├── predictions/               # Test set predictions (.csv)  
└── figures/                   # Evaluation visualizations (.png)
```

### Key Metrics Tracked
- **Accuracy Metrics**: RMSE, MAE, R²
- **Regulatory Compliance**: Threshold exceedance rates, overestimation percentages
- **Segment Analysis**: Performance across income groups (low, mid, high)
- **Conservative Estimation**: Within 10%, 20%, 30% accuracy rates

### Research Findings Summary

Based on comprehensive evaluation on the synthetic BOG dataset:

1. **Baseline XGBoost**: Strong predictive accuracy but higher regulatory violations
2. **Post-hoc Quantile**: Significant improvement in compliance with minimal accuracy loss
3. **Pre-hoc Huber + Threshold**: Balanced approach with built-in regulatory awareness
4. **Pre-hoc Segment-Aware**: Best overall performance combining accuracy and compliance

## Model Documentation

For detailed information about each modeling approach, including methodology, implementation details, and performance analysis, refer to the individual model READMEs:

### 📊 Model-Specific Documentation

| Model | Directory | README | Description |
|-------|-----------|--------|-------------|
| **XGBoost Baseline** | `baseline/xgboost/` | [README](baseline/xgboost/README.md) | Standard ML approach and performance baseline |
| **Quantile Calibration** | `posthoc/quantile/` | [README](posthoc/quantile/README.md) | Post-hoc correction methodology and results |
| **Huber + Threshold** | `prehoc/huber_threshold/` | [README](prehoc/huber_threshold/README.md) | Custom objective function and training approach |
| **Segment-Aware** | `prehoc/segment_aware/` | [README](prehoc/segment_aware/README.md) | Advanced segment-specific regulatory modeling |

Each model README contains:
- **Methodology**: Detailed approach and mathematical foundations
- **Implementation**: Configuration parameters and training details  
- **Results**: Performance metrics and regulatory compliance analysis
- **Usage**: Specific commands and examples for that model
- **Insights**: Strengths, limitations, and business implications
