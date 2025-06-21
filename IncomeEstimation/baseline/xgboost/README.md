# XGBoost Baseline Model

This directory contains the baseline XGBoost regression model for income estimation, serving as the foundation for comparing post-hoc calibration and pre-hoc regulation-aware approaches.

## Model Overview

The baseline XGBoost model represents the **standard machine learning approach** to income estimation without explicit regulatory constraints. It is optimized purely for predictive accuracy using traditional regression metrics.

### Model Selection

**XGBoost was chosen** as the baseline model after testing several tree-based algorithms including Random Forest, LightGBM, and CatBoost. All tree-based models showed similar performance levels, so XGBoost was selected.

### Configuration

The model uses **hyperparameters optimized in the research notebook.ipynb** through extensive grid search. No hyperparameter tuning is performed in this repository - the best parameters from notebook.ipynb analysis are directly used in the configuration files.

**Key parameters from `config/model_config.yaml`:**
- Learning rate: 0.05 (conservative to prevent overfitting)
- Max depth: 5 (balanced complexity)
- N estimators: 300 (optimal from grid search)
- Regularization: L1=1, L2=5 (prevents overfitting)

### Limitations of Basic Models

As expected, **such basic models don't perform well** for regulatory compliance. The standard XGBoost approach shows:

- High threshold exceedance rates (~28%)
- Significant overestimation bias (~45% of predictions)
- No built-in regulatory awareness
- Need for post-hoc adjustments for compliance

## Usage

The recommended command handles the complete pipeline including data cleaning, preprocessing, training, inference, and evaluation:

```bash
# Via Makefile (recommended)
make baseline/xgboost
```

This single command will:
1. Clean previously generated outputs and figures
2. Preprocess the raw data
3. Train the XGBoost model
4. Generate predictions on the test set
5. Evaluate model performance and create metrics
6. Generate visualization figures

## Results and Storage

**Model and results are stored in:**

```
baseline/xgboost/results/
├── models/               # Trained XGBoost model (.joblib)
├── metrics/              # Performance metrics (.json)
├── predictions/          # Test set predictions (.csv)
└── figures/              # Evaluation visualizations (.png)
```

**Key files:**
- `models/xgboost_model.joblib` - Trained model
- `metrics/evaluation_metrics.json` - All performance metrics
- `predictions/test_predictions.csv` - Predictions with actual values
- `figures/` - Performance plots and visualizations
