"""
Simple preprocessing module for income estimation models.

This module provides a streamlined preprocessing pipeline that exactly matches
the notebook implementation, using only:
1. KNNImputerTransformer for missing value imputation
2. EnhancedOutlierHandler for outlier handling
3. FeatureGenerator for feature engineering

No filtering or zero handling is performed, matching the notebook configuration.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Now import our modules
from IncomeEstimation.src.preprocessing.transformers.imputers import KNNImputerTransformer
from IncomeEstimation.src.preprocessing.transformers.outlier_handler import EnhancedOutlierHandler
from IncomeEstimation.src.preprocessing.transformers.feature_generator import FeatureGenerator
from IncomeEstimation.src.preprocessing.loader import load_income_data
from IncomeEstimation.src.preprocessing.config import (
    OUTLIER_STRATEGY_MAP,
    IQR_PARAMS,
    PERCENTILE_PARAMS,
    ZSCORE_PARAMS
)
from IncomeEstimation.src.utils.logging import get_logger

# Initialize logger for preprocessing
logger = get_logger('preprocessing')


def build_pipeline(random_state=42):
    """
    Build the preprocessing pipeline that exactly matches the notebook.
    
    The pipeline consists of:
    1. KNNImputerTransformer - Impute missing values using KNN
    2. EnhancedOutlierHandler - Handle outliers using specified strategies  
    3. FeatureGenerator - Generate new features
    
    Parameters
    ----------
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    Pipeline
        The preprocessing pipeline.
    """
    steps = [
        # Step 1: Impute missing values using KNN
        ("imputer", KNNImputerTransformer(n_neighbors=5)),
        
        # Step 2: Handle outliers
        ("outlier_handler", EnhancedOutlierHandler(
            strategy_map=OUTLIER_STRATEGY_MAP,
            iqr_params=IQR_PARAMS,
            zscore_params=ZSCORE_PARAMS,
            percentile_params=PERCENTILE_PARAMS,
            create_flags=False  # Match notebook: outlier flags not created
        )),
        
        # Step 3: Generate new features
        ("features", FeatureGenerator())
    ]
    
    return Pipeline(steps=steps)


def process_data(data_path, output_dir=None, test_size=0.2, random_state=42):
    """
    Process the data exactly as in the notebook.
    
    Parameters
    ----------
    data_path : str
        Path to the raw data.
    output_dir : str, default=None
        Directory to save the processed data.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, original_features
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = load_income_data(data_path)
    
    # Get original features list for reference
    original_features = df.drop(columns=['target']).columns.tolist()
    
    # Split into X and y
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Split data: {len(X_train)} train samples, {len(X_test)} test samples")
    
    # Create preprocessing pipeline
    pipeline = build_pipeline(random_state=random_state)
    
    # Process training data
    logger.info("Processing training data...")
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    
    # Process test data (use the same fitted pipeline to avoid data leakage)
    logger.info("Processing test data...")
    X_test_processed = pipeline.transform(X_test)
    
    # Log summary
    logger.info("=== PREPROCESSING SUMMARY ===")
    logger.info(f"Training data: {len(X_train_processed)} rows")
    logger.info(f"Test data: {len(X_test_processed)} rows")
    
    # Calculate derived features
    generated_features = [col for col in X_train_processed.columns if col not in original_features]
    
    logger.info("=== FEATURE SUMMARY ===")
    logger.info(f"Original features: {len(original_features)}")
    logger.info(f"Generated features: {len(generated_features)}")
    logger.info(f"Total features: {X_train_processed.shape[1]}")
    
    # Save processed data if output_dir is provided
    if output_dir:
        save_processed_data(
            X_train_processed, 
            X_test_processed, 
            y_train, 
            y_test,
            target_col='target',
            output_dir=output_dir
        )
    
    return X_train_processed, X_test_processed, y_train, y_test, original_features


def save_processed_data(X_train, X_test, y_train, y_test, target_col='target', output_dir=None):
    """
    Save processed training and test data to CSV files.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Processed training features.
    X_test : pandas.DataFrame
        Processed test features.
    y_train : pandas.Series
        Training target values.
    y_test : pandas.Series
        Test target values.
    target_col : str, default='target'
        Name of the target column.
    output_dir : str, default=None
        Directory to save the processed data. If None, 
        saves to '../data/processed/' relative to the current directory.

    Returns
    -------
    tuple
        Paths to the saved train and test CSV files.
    """
    # Merge features with target
    train = X_train.copy()
    train[target_col] = y_train.loc[train.index]

    test = X_test.copy()
    test[target_col] = y_test.loc[test.index]

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.abspath("../data/processed")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    logger.info(f"✅ Saved to {train_path} and {test_path}")
    
    return train_path, test_path


if __name__ == '__main__':
    logger.info("Running preprocessing pipeline")
    # Get the absolute path to the project root (BA_Thesis directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    
    # Path to the raw data
    data_path = os.path.join(project_root, 'data', 'raw', 'income_data.xlsx')
    # Output directory for processed data
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process data and save results
    X_train, X_test, y_train, y_test, original_features = process_data(
        data_path=data_path,
        output_dir=output_dir,
        test_size=0.2,
        random_state=42
    )
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Original features: {len(original_features)}")
    logger.info(f"Total features after preprocessing: {X_train.shape[1]}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Testing samples: {len(X_test)}")