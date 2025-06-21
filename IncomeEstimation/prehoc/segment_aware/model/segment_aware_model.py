"""
Segment-Aware Huber Threshold Model implementation using a custom objective function that
combines segment-specific Huber loss with dynamic threshold penalties tailored for different income groups.
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import xgboost as xgb
from xgboost import DMatrix, train as xgb_train
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from IncomeEstimation.src.toolkit.theme import Theme
from IncomeEstimation.src.toolkit.model_evaluator import ModelEvaluator
from IncomeEstimation.src.toolkit.threshold_utils import exceeds_dynamic_threshold


class SegmentAwareHuberThresholdModel:
    """
    Segment-Aware Huber Threshold Model for income estimation that combines segment-specific
    Huber loss with penalties for exceeding dynamic thresholds, with different parameters
    for different income segments.
    
    This model applies differentiated treatment based on income level:
    - Low income (≤1500): More conservative with higher penalties
    - Mid income (1500-2500): Moderate penalties
    - High income (>2500): More lenient with lower penalties
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Segment-Aware Huber Threshold Model.
        
        Parameters
        ----------
        config_path : str or Path, optional
            Path to the YAML configuration file
        """
        self.model = None
        self.config = None
        self.evaluator = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load model configuration from YAML file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to the configuration file
            
        Returns
        -------
        self
            Model instance with loaded configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize the evaluator with theme from config if available
        theme = self.config.get('visualization', {}).get('theme', 'teal')
        self.evaluator = ModelEvaluator(theme=theme)
        
        return self
    
    @staticmethod
    def exceeds_dynamic_threshold(y_true, y_pred, absolute_threshold=200, percentage_threshold=20):
        """
        Check if predictions exceed a dynamic threshold based on true values.
        
        Parameters
        ----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values
        absolute_threshold : float, default=200
            Absolute threshold in currency units
        percentage_threshold : float, default=20
            Percentage threshold
            
        Returns
        -------
        tuple
            Boolean mask of exceeded predictions, dynamic thresholds, and exceed amounts
        """
        # Calculate dynamic threshold for each true value
        dynamic_thresholds = np.maximum(
            absolute_threshold,
            y_true * (percentage_threshold / 100)
        )
        
        # Calculate exceed amount (how much the prediction exceeds the threshold)
        errors = y_pred - y_true
        exceed_amounts = errors - dynamic_thresholds
        
        # Create mask for predictions that exceed threshold
        exceeds = errors > dynamic_thresholds
        
        return exceeds, dynamic_thresholds, exceed_amounts
    
    def segment_aware_huber_threshold_loss(self, predt, dtrain):
        """
        Segment-aware custom objective combining Huber loss with a squared penalty
        for predictions exceeding a dynamic threshold, with different parameters
        for different income segments.
        
        Parameters
        ----------
        predt : array-like
            Model predictions
        dtrain : xgboost.DMatrix
            Training data containing the true target values
            
        Returns
        -------
        tuple
            Gradient and hessian values
        """
        y_true = dtrain.get_label()
        errors = predt - y_true
        abs_errors = np.abs(errors)
        
        # Get segment thresholds from config
        model_config = self.config.get('model', {})
        segments_config = model_config.get('segments', {})
        low_threshold = segments_config.get('low_threshold', 1500)
        mid_threshold = segments_config.get('mid_threshold', 2500)
        
        # === Segment assignment ===
        low_segment = y_true <= low_threshold
        mid_segment = (y_true > low_threshold) & (y_true <= mid_threshold)
        high_segment = y_true > mid_threshold
        
        # Get segment-specific parameters from config
        delta_config = model_config.get('delta', {})
        weight_config = model_config.get('threshold_weight', {})
        
        # === Segment-specific Huber delta ===
        delta = np.zeros_like(y_true)
        delta[low_segment] = delta_config.get('low_segment', 30.0)
        delta[mid_segment] = delta_config.get('mid_segment', 80.0)
        delta[high_segment] = delta_config.get('high_segment', 200.0)
        
        # === Segment-specific threshold weights ===
        penalty_weight = np.zeros_like(y_true)
        penalty_weight[low_segment] = weight_config.get('low_segment', 12.0)
        penalty_weight[mid_segment] = weight_config.get('mid_segment', 3.0)
        penalty_weight[high_segment] = weight_config.get('high_segment', 1.0)
        
        # === Base Huber loss ===
        is_small_error = abs_errors <= delta
        grad = np.where(is_small_error, 2 * errors, 2 * delta * np.sign(errors))
        hess = np.where(is_small_error, 2.0, 0.1)
        
        # Get threshold parameters
        threshold_config = model_config.get('dynamic_threshold', {})
        absolute_threshold = threshold_config.get('absolute', 200)
        percentage_threshold = threshold_config.get('percentage', 20)
        
        # === Threshold penalty (squared exceedance) ===
        exceeds, dynamic_thresholds, _ = exceeds_dynamic_threshold(
            y_true, predt, absolute_threshold, percentage_threshold
        )
        
        if np.any(exceeds):
            exceed_amount = predt[exceeds] - y_true[exceeds] - dynamic_thresholds[exceeds]
            exceed_grad = np.zeros_like(errors)
            exceed_hess = np.zeros_like(errors)
            
            exceed_grad[exceeds] = 2 * exceed_amount * penalty_weight[exceeds]
            exceed_hess[exceeds] = 2.0 * penalty_weight[exceeds]
            
            grad += exceed_grad
            hess += exceed_hess
        
        return grad, hess
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Segment-Aware Huber Threshold Model.
        
        Parameters
        ----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training target
        X_val : pandas.DataFrame, optional
            Validation features for evaluation during training
        y_val : pandas.Series, optional
            Validation target for evaluation during training
            
        Returns
        -------
        self
            Trained model instance
        """
        # Create DMatrix objects
        dtrain = DMatrix(X_train, label=y_train)
        
        # Set up evaluation watchlist
        watchlist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = DMatrix(X_val, label=y_val)
            watchlist.append((dval, 'validation'))
        
        # Get training parameters
        training_config = self.config.get('training', {})
        num_boost_round = training_config.get('num_boost_round', 300)
        early_stopping_rounds = training_config.get('early_stopping_rounds', 50)
        verbose_eval = training_config.get('verbose_eval', 50)
        
        # Extract XGBoost parameters from config
        xgb_params = {
            'max_depth': self.config.get('max_depth', 5),
            'eta': self.config.get('eta', 0.05),
            'subsample': self.config.get('subsample', 1.0),
            'colsample_bytree': self.config.get('colsample_bytree', 0.6),
            'lambda': self.config.get('lambda', 5),
            'alpha': self.config.get('alpha', 1),
            'gamma': self.config.get('gamma', 0),
            'tree_method': self.config.get('tree_method', 'hist'),
            'random_state': self.config.get('random_state', 42)
        }
        
        # Train the model
        evals_result = {}
        self.model = xgb_train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=self.segment_aware_huber_threshold_loss,
            evals=watchlist,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input features
            
        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        dmatrix = DMatrix(X)
        return self.model.predict(dmatrix)
    
    def save(self, filepath):
        """
        Save the trained model to disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the model
            
        Returns
        -------
        self
            Model instance
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model_data = {
            'model': self.model,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        
        return self
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved model
            
        Returns
        -------
        SegmentAwareHuberThresholdModel
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        model = cls()
        model.model = model_data['model']
        model.config = model_data['config']
        
        # Initialize the evaluator with theme from config if available
        theme = model.config.get('visualization', {}).get('theme', 'teal')
        model.evaluator = ModelEvaluator(theme=theme)
        
        return model
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input features
        y : pandas.Series
            True target values
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Calculate basic metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        # Calculate threshold exceedance
        model_config = self.config.get('model', {})
        threshold_config = model_config.get('dynamic_threshold', {})
        absolute_threshold = threshold_config.get('absolute', 200)
        percentage_threshold = threshold_config.get('percentage', 20)
        
        exceeds, _, _ = self.exceeds_dynamic_threshold(
            y, y_pred, absolute_threshold, percentage_threshold
        )
        
        metrics['exceeds_threshold_pct'] = np.mean(exceeds) * 100
        metrics['overestimation_pct'] = np.mean((y_pred - y) > 0) * 100
        metrics['within_20pct'] = np.mean(np.abs((y_pred - y) / (y + 1e-6)) <= 0.2) * 100
        
        return metrics


def load_config(config_path):
    """
    Load model configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
