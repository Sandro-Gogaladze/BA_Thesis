import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import QuantileRegressor
import yaml

class QuantileRegressionModel:
    """
    Post-hoc calibration model using quantile regression to improve regulatory compliance 
    by reducing overestimation beyond defined supervisory thresholds.

    This model calibrates baseline XGBoost predictions by learning a linear transformation 
    using quantile regression to shift outputs conservatively.
    """
    
    def __init__(self, base_model=None, quantile=0.2, random_state=42):
        """
        Initialize the quantile regression model.
        
        Parameters
        ----------
        base_model : object
            The baseline model object with predict method.
        quantile : float, default=0.2
            Quantile level for regression (0.2 selected from notebook calibration).
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.base_model = base_model
        self.quantile = quantile
        self.random_state = random_state
        self.quantile_regressor = None
        self.beta_0 = None  # Intercept
        self.beta_1 = None  # Slope
        self.is_fitted = False
    
    def fit(self, X, y, sample_size=8000):
        """
        Fit the quantile regression model to calibrate baseline predictions.
        
        Parameters
        ----------
        X : array-like
            Features for training. Will be used to generate base_model predictions.
        y : array-like
            True target values.
        sample_size : int, default=8000
            Number of samples to use for fitting quantile regression (for efficiency).
            
        Returns
        -------
        self : object
            Fitted model instance.
        """
        if self.base_model is None:
            raise ValueError("Base model must be provided before fitting calibration")
        
        # Generate predictions from the base model
        base_predictions = self.base_model.predict(X)
        
        # Reshape for quantile regression (required by sklearn)
        X_qr = base_predictions.reshape(-1, 1)
        
        # Sample subset for faster training (matching notebook exactly)
        if sample_size and sample_size < len(X_qr):
            # Use np.random.default_rng exactly as in the notebook
            rng = np.random.default_rng(self.random_state)
            subset_idx = rng.choice(len(X_qr), size=sample_size, replace=False)
            X_sub = X_qr[subset_idx]
            y_sub = y[subset_idx]
        else:
            X_sub = X_qr
            y_sub = y
        
        # Fit quantile regression model
        self.quantile_regressor = QuantileRegressor(
            quantile=self.quantile, 
            alpha=1e-2,  # L1 regularization term as in notebook
            solver='highs'  # Default solver
        )
        self.quantile_regressor.fit(X_sub, y_sub)
        
        # Extract coefficients for easier access
        self.beta_0 = self.quantile_regressor.intercept_
        self.beta_1 = self.quantile_regressor.coef_[0]
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Generate calibrated predictions using the base model and quantile regression.
        
        Parameters
        ----------
        X : array-like
            Features for prediction.
            
        Returns
        -------
        array-like
            Calibrated predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() before predict().")
        
        # Get base model predictions
        base_predictions = self.base_model.predict(X)
        
        # Apply calibration: y_adjusted = beta_0 + beta_1 * y_pred
        calibrated_predictions = self.beta_0 + self.beta_1 * base_predictions
        
        return calibrated_predictions
    
    def save(self, filepath):
        """
        Save the calibrated model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")
        
        model_data = {
            'quantile': self.quantile,
            'beta_0': self.beta_0,
            'beta_1': self.beta_1,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        # Save only the calibration parameters, not the base model
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath, base_model):
        """
        Load a saved calibrated model.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
        base_model : object
            The base model object to use with the loaded calibration.
            
        Returns
        -------
        QuantileRegressionModel
            Loaded model instance.
        """
        model_data = joblib.load(filepath)
        
        # Create a new instance with the saved parameters
        instance = cls(
            base_model=base_model,
            quantile=model_data['quantile'],
            random_state=model_data['random_state']
        )
        
        # Restore calibration parameters
        instance.beta_0 = model_data['beta_0']
        instance.beta_1 = model_data['beta_1']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
    
    @staticmethod
    def compute_overestimation_ratio(y_true, y_pred, threshold=0.2):
        """
        Compute the proportion of predictions that overestimate the true values
        by more than a specified percentage threshold.
        
        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
        threshold : float, default=0.2
            Overestimation threshold as a decimal (e.g., 0.2 for 20%).
            
        Returns
        -------
        float
            Fraction of predictions exceeding the overestimation threshold.
        """
        over_errors = ((y_pred - y_true) / (y_true + 1e-6)) > threshold
        return over_errors.sum() / len(y_true)
    
    @staticmethod
    def compute_dynamic_threshold_exceedance(y_true, y_pred, absolute=200, percentage=0.2):
        """
        Calculate the proportion of predictions whose error exceeds a dynamic threshold.
        The threshold is defined as the maximum of an absolute and percentage-based bound.
        
        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
        absolute : float, default=200
            Absolute error threshold.
        percentage : float, default=0.2
            Relative error threshold as a fraction (e.g., 0.2 for 20%).
            
        Returns
        -------
        float
            Ratio of predictions that exceed the dynamic threshold.
        """
        dynamic_threshold = np.maximum(absolute, y_true * percentage)
        error = y_pred - y_true
        exceeds = error > dynamic_threshold
        return exceeds.sum() / len(y_true)


def load_config(config_path):
    """
    Load model configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
        
    Returns
    -------
    dict
        Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
