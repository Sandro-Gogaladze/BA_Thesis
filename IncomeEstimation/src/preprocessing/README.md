# Preprocessing Module

This module provides a streamlined preprocessing pipeline for income estimation models that exactly matches the notebook.ipynb implementation.

## Structure

- `transformers/`: Contains modular transformer classes for preprocessing
  - `imputers.py`: Handles missing value imputation using KNN
  - `outlier_handler.py`: Caps outliers using strategies tailored to each feature
  - `feature_generator.py`: Creates financial ratio features to enhance predictive power
- `preprocessor.py`: Main preprocessing module
- `config.py`: Configuration constants for preprocessing
- `loader.py`: Data loading utilities

## Usage

The preprocessing module provides a simple function-based API:

```python
from IncomeEstimation.src.preprocessing.preprocessor import process_data

# Run preprocessing and save results
X_train, X_test, y_train, y_test, original_features = process_data(
    data_path='data/raw/income_data.xlsx',
    output_dir='data/processed',
    test_size=0.2,
    random_state=42
)
```

## Pipeline Steps

The preprocessing pipeline consists of three main steps:

1. **Missing Value Imputation**: Uses KNN imputation to handle missing values
2. **Outlier Handling**: Caps outliers using strategies tailored to each feature type
3. **Feature Generation**: Creates financial ratio features to enhance predictive power

This simplified pipeline matches exactly what is used in the original notebook.ipynb analysis.
