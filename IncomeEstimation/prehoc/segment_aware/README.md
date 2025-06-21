# Pre-hoc Segment-Aware Huber + Threshold Model

This directory implements the **most advanced regulation-aware approach** - the pre-hoc Segment-Aware Huber + Threshold penalty model. This approach extends the basic Huber + Threshold model by **treating segments differently** with different weights and penalties.

## Model Overview

The segment-aware model represents the **most sophisticated regulatory integration** in this research, building upon the Huber + Threshold foundation with income segment-specific adjustments for enhanced regulatory compliance across all income levels.

### Approach Development

**Same approach as Huber + Threshold but enhanced** with segment-specific considerations. **I'm treating segments differently** with different weights and penalty structures, which **allows for better control and ensuring the model performs similarly in all segments**.

### Segment-Specific Strategy

The model applies **different weights for different income segments**:
- **Low Income (≤1500 GEL)**: **High penalty in sensitive segments** to protect vulnerable borrowers
- **Mid Income (1500-2500 GEL)**: Balanced penalties maintaining accuracy-compliance equilibrium  
- **High Income (>2500 GEL)**: Reduced penalties allowing for higher variance in high-income predictions

This segmented approach ensures appropriate regulatory protection where it's most needed while maintaining accuracy where possible.

### Enhanced Objective Function

The model implements a sophisticated segment-aware loss function with two main components:

$$
\mathcal{L}_s = \mathcal{L}_{\mathrm{Huber,}s} + \mathcal{L}_{\mathrm{Penalty,}s}
$$

The first component is the segment-specific Huber loss:

$$
\mathcal{L}_{\mathrm{Huber,s}} = 
\begin{cases}
(\hat{y}_i - y_i)^2, & \text{if } |\hat{y}_i - y_i| \leq \delta_s \\
2\delta_s \cdot |\hat{y}_i - y_i| - \delta_s, & \text{otherwise}
\end{cases}
$$

The second component is the regulatory penalty term, which activates only when the prediction exceeds the allowable threshold:

$$
\mathcal{L}_{\mathrm{Penalty,s}} = \lambda_s \cdot (\hat{y}_i - y_i - T_i)^2 \cdot \mathbf{1}_{[\hat{y}_i > y_i+T_i]}
$$

**Components:**
1. **Segment-specific Huber Loss** ($\mathcal{L}_{\mathrm{Huber,s}}$): Provides a robust regression foundation with segment-specific delta parameters ($\delta_s$)
2. **Regulatory Penalty Term** ($\mathcal{L}_{\mathrm{Penalty,s}}$): Applies segment-specific penalties ($\lambda_s$) when predictions exceed the allowed threshold

### Segment-Specific Parameters

The model uses segment-specific parameters tuned for optimal regulatory compliance across income levels:

- **Low Income (≤1500 GEL)**: 
  - $\lambda_s = 12.0$ (high penalty weight for vulnerable borrowers)
  - $\delta_s = 100$ (smaller delta for tighter error bounds)
  
- **Mid Income (1500-2500 GEL)**:
  - $\lambda_s = 3.0$ (balanced penalty weight)
  - $\delta_s = 150$ (moderate delta value)
  
- **High Income (>2500 GEL)**:
  - $\lambda_s = 1.0$ (lower penalty weight)
  - $\delta_s = 300$ (larger delta allowing more flexibility)

These parameters control how strictly the model penalizes regulatory violations - a higher $\lambda_s$ leads to more conservative predictions.

### Advantages

- **Optimal Segment Performance**: Tailored protection for different income levels
- **Enhanced Fairness**: Improved performance equity across borrower segments
- **Better Control**: Fine-grained regulation of model behavior by income level
- **Comprehensive Compliance**: Optimal protection where most needed
- **Advanced ML Integration**: Sophisticated objective function design

### Disadvantages

- **Implementation Complexity**: Most complex approach requiring sophisticated objective design
- **Hyperparameter Sensitivity**: Multiple weights require careful tuning
- **Computational Overhead**: Increased training time due to segment calculations
- **Framework Limitations**: Advanced custom objectives may have limited support

## Usage

The recommended command handles the complete pipeline including data cleaning, preprocessing, training, inference, and evaluation:

```bash
# Via Makefile (recommended)
make prehoc/segment_aware
```

This single command will:
1. Clean previously generated outputs and figures
2. Preprocess the raw data
3. Train the segment-aware model
4. Generate predictions on the test set
5. Evaluate model performance and create metrics
6. Generate visualization figures

## Results and Storage

**Model and results are stored in:**

```
prehoc/segment_aware/results/
├── models/                    # Most advanced regulation-aware models (.joblib)
├── metrics/                   # Comprehensive segment-specific metrics (.json)
├── predictions/               # Optimally calibrated predictions (.csv)
└── figures/                   # Advanced performance visualizations (.png)
```

**Key files:**
- `models/segment_aware_model.joblib` - Most sophisticated trained model
- `metrics/evaluation_metrics.json` - Comprehensive segment-specific metrics
- `predictions/test_predictions.csv` - Segment-aware predictions with analysis
- `figures/` - Advanced segment performance visualizations
