"""
XGBoost Baseline Model Implementation
"""
import os
import joblib
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from IncomeEstimation.src.toolkit.theme import Theme
from IncomeEstimation.src.toolkit.model_evaluator import ModelEvaluator



class XGBoostModel:
    """
    XGBoost Baseline Model for Income Estimation.
    
    This class handles training, evaluation, and inference for the XGBoost model.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the XGBoost model with configuration.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the model configuration YAML file.
        """
        self.model = None
        self.config = None
        self.evaluator = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load model configuration from a YAML file.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set up evaluator with threshold parameters from config
        dynamic_threshold = self.config.get('evaluation', {}).get('dynamic_threshold', {})
        absolute = dynamic_threshold.get('absolute', 200)
        percentage = dynamic_threshold.get('percentage', 20)
        
        self.evaluator = ModelEvaluator(
            theme='blue',
            dynamic_threshold_absolute=absolute,
            dynamic_threshold_percentage=percentage
        )
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the XGBoost model using sklearn API (XGBRegressor).
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        X_test : pd.DataFrame, optional
            Test features for validation during training.
        y_test : pd.Series, optional
            Test target for validation during training.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        if self.config is None:
            raise ValueError("Model configuration not loaded. Call load_config first.")
        
        model_params = self.config.get('model_params', {})
        training_params = self.config.get('training', {})
        
        # Remove params not accepted by XGBRegressor or duplicate params
        model_params = {k: v for k, v in model_params.items() if k not in ['random_state']}
        # Set random_state separately
        random_state = self.config['model_params'].get('random_state', 42)
        
        # Don't pass n_estimators twice (it's already in model_params)
        self.model = xgb.XGBRegressor(
            **model_params,
            random_state=random_state,
            verbosity=1
        )
        
        fit_args = {}
        if X_test is not None and y_test is not None:
            fit_args['eval_set'] = [(X_test, y_test)]
            
            fit_args['verbose'] = training_params.get('verbose_eval', 50)
        
        self.model.fit(X_train, y_train, **fit_args)
        
        return self
    
    def predict(self, X):
        """
        Generate predictions using the trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction.
            
        Returns:
        --------
        np.ndarray
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        return self.evaluator.calculate_basic_metrics(y_test, y_pred)
    
    def evaluate_segments(self, X_test, y_test):
        """
        Evaluate the model on test data by income segments.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with evaluation metrics by segment.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Create analysis dataframe
        analysis_df = self.evaluator.create_analysis_dataframe(y_test, y_pred)
        
        # Calculate segment metrics
        return self.evaluator.calculate_segment_metrics(analysis_df)
    
    def save_model(self, model_path, config_path=None):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        model_path : str
            Path to save the model.
        config_path : str, optional
            Path to save the configuration. If None, config won't be saved.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Standardize extension to .joblib
        if not model_path.endswith('.joblib'):
            # Get base path without extension
            base_path = model_path.rsplit('.', 1)[0] if '.' in os.path.basename(model_path) else model_path
            model_path = f"{base_path}.joblib"
        
        # Simply save with joblib
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save config if path is provided
        if config_path and self.config:
            with open(config_path, 'w') as file:
                yaml.dump(self.config, file)
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Standardize extension to .joblib if needed
        model_path = str(model_path)  # Convert Path to string if needed
        
        if not model_path.endswith('.joblib'):
            # Try with .joblib extension
            base_path = model_path.rsplit('.', 1)[0] if '.' in os.path.basename(model_path) else model_path
            joblib_path = f"{base_path}.joblib"
            if os.path.exists(joblib_path):
                model_path = joblib_path
        
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Simply load with joblib
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from: {model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
        
        return self
    
    @classmethod
    def load(cls, model_path):
        """
        Load a trained model from disk as a class method.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model.
            
        Returns:
        --------
        XGBoostModel
            Loaded model instance.
        """
        instance = cls()
        instance.load_model(model_path)
        return instance
