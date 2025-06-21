import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from IncomeEstimation.src.utils.paths import resolve_data_path


def load_income_data(filepath):
    """
    Load the income dataset from the specified Excel file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the Excel file containing the income data.

    Returns
    -------
    pd.DataFrame
        Loaded income dataset from the 'income_data' sheet.
    """
    # Resolve the data path
    filepath = resolve_data_path(filepath)
    return pd.read_excel(filepath, sheet_name="income_data")


def load_data(data_path, test_size=0.2, random_state=42):
    """
    Load and split data into training and testing sets.
    
    Parameters
    ----------
    data_path : str
        Path to the raw data file.
    test_size : float, optional (default=0.2)
        Proportion of data to use for testing.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
        
    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Testing target values.
    """
    # Load raw data
    data = load_income_data(data_path)
    
    # Extract features and target
    X = data.drop(columns="target")
    y = data["target"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data loaded and split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test


def load_prediction_data(data_path):
    """
    Load data for prediction purposes.
    
    Parameters
    ----------
    data_path : str
        Path to the data file for prediction.
        
    Returns
    -------
    pd.DataFrame
        Features for prediction.
    """
    # Resolve the data path
    data_path = resolve_data_path(data_path)
    
    # Check file extension to determine loading method
    if str(data_path).endswith('.xlsx'):
        data = load_income_data(data_path)
        if 'target' in data.columns:
            return data.drop(columns=['target'])
        return data
    elif str(data_path).endswith('.csv'):
        data = pd.read_csv(data_path)
        if 'target' in data.columns:
            return data.drop(columns=['target'])
        return data
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
