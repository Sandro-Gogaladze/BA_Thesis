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

![Total Loss](https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D_s%20%3D%20%5Cmathcal%7BL%7D_%7B%5Cmathrm%7BHuber%2C%7Ds%7D%20&plus;%20%5Cmathcal%7BL%7D_%7B%5Cmathrm%7BPenalty%2C%7Ds%7D)

The first component is the segment-specific Huber loss:

![Huber Loss](https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D_%7B%5Cmathrm%7BHuber%2Cs%7D%7D%20%3D%20%5Cbegin%7Bcases%7D%20%28%5Chat%7By%7D_i%20-%20y_i%29%5E2%2C%20%26%20%5Ctext%7Bif%20%7D%20%7C%5Chat%7By%7D_i%20-%20y_i%7C%20%5Cleq%20%5Cdelta_s%20%5C%5C%202%5Cdelta_s%20%5Ccdot%20%7C%5Chat%7By%7D_i%20-%20y_i%7C%20-%20%5Cdelta_s%2C%20%26%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D)

The second component is the regulatory penalty term, which activates only when the prediction exceeds the allowable threshold:

![Penalty Loss](https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D_%7B%5Cmathrm%7BPenalty%2Cs%7D%7D%20%3D%20%5Clambda_s%20%5Ccdot%20%28%5Chat%7By%7D_i%20-%20y_i%20-%20T_i%29%5E2%20%5Ccdot%20%5Cmathbf%7B1%7D_%7B%5B%5Chat%7By%7D_i%20%3E%20y_i&plus;T_i%5D%7D)

**Components:**
1. **Segment-specific Huber Loss** (![Huber Loss Symbol](https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D_%7B%5Cmathrm%7BHuber%2Cs%7D%7D)): Provides a robust regression foundation with segment-specific delta parameters (![Delta Symbol](https://latex.codecogs.com/png.latex?%5Cdelta_s))
2. **Regulatory Penalty Term** (![Penalty Loss Symbol](https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D_%7B%5Cmathrm%7BPenalty%2Cs%7D%7D)): Applies segment-specific penalties (![Lambda Symbol](https://latex.codecogs.com/png.latex?%5Clambda_s)) when predictions exceed the allowed threshold

### Segment-Specific Parameters

The model uses segment-specific parameters tuned for optimal regulatory compliance across income levels:

- **Low Income (≤1500 GEL)**: 
  - ![Lambda Symbol](https://latex.codecogs.com/png.latex?%5Clambda_s%20%3D%2012.0) (high penalty weight for vulnerable borrowers)
  - ![Delta Symbol](https://latex.codecogs.com/png.latex?%5Cdelta_s%20%3D%20100) (smaller delta for tighter error bounds)
  
- **Mid Income (1500-2500 GEL)**:
  - ![Lambda Symbol](https://latex.codecogs.com/png.latex?%5Clambda_s%20%3D%203.0) (balanced penalty weight)
  - ![Delta Symbol](https://latex.codecogs.com/png.latex?%5Cdelta_s%20%3D%20150) (moderate delta value)
  
- **High Income (>2500 GEL)**:
  - ![Lambda Symbol](https://latex.codecogs.com/png.latex?%5Clambda_s%20%3D%201.0) (lower penalty weight)
  - ![Delta Symbol](https://latex.codecogs.com/png.latex?%5Cdelta_s%20%3D%20300) (larger delta allowing more flexibility)

These parameters control how strictly the model penalizes regulatory violations - a higher ![Lambda Symbol](https://latex.codecogs.com/png.latex?%5Clambda_s) leads to more conservative predictions.

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
