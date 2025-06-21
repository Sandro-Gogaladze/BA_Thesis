# Pre-hoc Huber + Threshold Penalty Model

This directory implements the **pre-hoc Huber + Threshold penalty approach** for regulation-aware income estimation. This method incorporates regulatory constraints directly into the training objective function by changing the model loss.

## Model Overview

The pre-hoc Huber + Threshold model represents a **training-time regulatory integration** approach that modifies the loss function to learn compliance during optimization. **Another way to achieve regulatory compliance is pre-hoc adjusting** - changing how the model learns rather than correcting afterwards.

### Approach Development

**I experimented with many different loss combinations** and the combination of **Huber loss and threshold penalty gave nice results**. This approach balances robust regression with explicit regulatory penalties.

### Huber Loss Foundation

**Huber loss** is a robust loss function that combines the benefits of:
- **Squared loss** (L2): Smooth gradients for small errors
- **Absolute loss** (L1): Robustness to outliers for large errors

**Huber Loss Formula:**

![Huber Loss](https://latex.codecogs.com/png.latex?L_%7B%5Cdelta%7D%28y_%7Btrue%7D%2C%20y_%7Bpred%7D%29%20%3D%20%5Cbegin%7Bcases%7D%20%5Cfrac%7B1%7D%7B2%7D%28y_%7Btrue%7D%20-%20y_%7Bpred%7D%29%5E2%20%26%20%5Ctext%7Bif%20%7D%20%7Cy_%7Btrue%7D%20-%20y_%7Bpred%7D%7C%20%5Cleq%20%5Cdelta%20%5C%5C%20%5Cdelta%7Cy_%7Btrue%7D%20-%20y_%7Bpred%7D%7C%20-%20%5Cfrac%7B1%7D%7B2%7D%5Cdelta%5E2%20%26%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D)

Where δ (delta) is the transition point between quadratic and linear behavior.

### Threshold Penalty Addition

**I added threshold penalty to the loss** to directly enforce regulatory constraints:

![Total Loss](https://latex.codecogs.com/png.latex?%5Ctext%7BTotal%5C_Loss%7D%20%3D%20%5Ctext%7BHuber%5C_Loss%7D%20&plus;%20%5Clambda%20%5Ctimes%20%5Ctext%7BThreshold%5C_Penalty%7D)

Where:
- **λ (lambda)**: Penalty weight controlling accuracy-compliance trade-off
- **Threshold\_Penalty**: Violation penalty based on NBG requirements

### Weight Selection

**I chose a weight of 3.0** for the threshold penalty based on:
- Experimentation across different weight values (1.0, 2.0, 3.0, 5.0, 10.0)
- Optimal balance between accuracy preservation and compliance improvement
- Stable training convergence and robust performance

### Advantages

- **Integrated Learning**: Simultaneously optimizes accuracy and compliance
- **Robust Architecture**: Huber loss provides stability against outliers
- **Direct Regulatory Mapping**: Explicit encoding of NBG requirements
- **Training-Time Awareness**: No post-hoc calibration required
- **Interpretable Parameters**: Clear control over accuracy-compliance trade-off

### Disadvantages

- **Custom Implementation Complexity**: Requires specialized objective function development
- **Hyperparameter Sensitivity**: Threshold weight requires careful tuning
- **Training Time**: Slightly increased computational overhead
- **Limited Framework Support**: Custom objectives may not be available in all ML libraries

## Usage

The recommended command handles the complete pipeline including data cleaning, preprocessing, training, inference, and evaluation:

```bash
# Via Makefile (recommended)
make prehoc/huber_threshold
```

This single command will:
1. Clean previously generated outputs and figures
2. Preprocess the raw data
3. Train the Huber + Threshold model with custom objective
4. Generate predictions on the test set
5. Evaluate model performance and create metrics
6. Generate visualization figures

## Results and Storage

**Model and results are stored in:**

```
prehoc/huber_threshold/results/
├── models/                   # Trained regulation-aware models (.joblib)
├── metrics/                  # Compliance and accuracy metrics (.json)
├── predictions/              # Conservative predictions (.csv)
└── figures/                  # Performance visualizations (.png)
```

**Key files:**
- `models/huber_threshold_model.joblib` - Trained model with custom objective
- `metrics/evaluation_metrics.json` - All performance metrics
- `predictions/test_predictions.csv` - Regulatory-aware predictions
- `figures/` - Custom objective performance plots
