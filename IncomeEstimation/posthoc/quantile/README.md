# Post-hoc Quantile Calibration Model

This directory implements the **post-hoc quantile calibration approach** for regulatory-compliant income estimation. This method applies quantile regression adjustment to baseline XGBoost predictions as an additional correction layer.

## Model Overview

The post-hoc quantile calibration represents a **correction layer approach** that transforms baseline predictions to achieve better regulatory compliance while maintaining the underlying model's accuracy characteristics.

### Approach Rationale

**One way to achieve regulatory compliance is post-correction**, for which quantile regression is often used by regulators and financial institutions. This approach serves as an **additional layer** on top of existing models, making it practical for banks with established ML systems.

### Quantile Selection

**Different quantiles were tested** (0.1, 0.15, 0.2, 0.25, 0.3) and **0.2 was chosen** based on:
- Training performance on train data
- Optimal balance between accuracy preservation and compliance improvement

The 20th percentile provides conservative adjustments without overly aggressive corrections.

### Configuration

The model uses **parameters optimized through experimentation**:
- Target quantile: 0.20 (20th percentile)
- Solver: 'highs' (efficient linear programming)
- Regularization: minimal (alpha=0.01)
- Fit intercept: true (allows baseline shift)

### Advantages

- **No Retraining Required**: Can enhance existing models without disrupting production
- **Fast Implementation**: Minimal computational overhead for calibration  
- **Interpretable Adjustment**: Simple linear transformation maintains model transparency
- **Significant Compliance Improvement**: Substantial reduction in regulatory violations
- **Deployment Flexibility**: Easy to implement as additional prediction layer

### Disadvantages

- **Limited Problem Solving**: Did not fully solve low income overestimations
- **Linear Assumption**: Simple transformation may not capture complex patterns
- **Accuracy Trade-off**: Some reduction in overall predictive accuracy
- **Distribution Dependency**: Performance varies with different income distributions
- **Incomplete Coverage**: May require multiple calibration stages for optimal results

## Usage

The recommended command handles the complete pipeline including data cleaning, preprocessing, training, calibration, inference, and evaluation:

```bash
# Via Makefile (recommended)
make posthoc/quantile
```

This single command will:
1. Clean previously generated outputs and figures
2. Preprocess the raw data
3. Train the baseline model
4. Apply quantile calibration
5. Generate calibrated predictions on the test set
6. Evaluate model performance and create metrics
7. Generate visualization figures

## Results and Storage

**Model and results are stored in:**

```
posthoc/quantile/results/
├── models/               # Baseline + calibration models (.joblib)
├── metrics/              # Performance metrics (.json)
├── predictions/          # Calibrated predictions (.csv)
└── figures/              # Evaluation visualizations (.png)
```

**Key files:**
- `models/quantile_model.joblib` - Combined baseline + calibration pipeline
- `metrics/evaluation_metrics.json` - All performance metrics
- `predictions/test_predictions.csv` - Calibrated predictions with comparisons
- `figures/` - Calibration performance plots
