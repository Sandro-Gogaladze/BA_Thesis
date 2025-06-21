"""
Script to train the Huber Threshold model
"""
import os
import yaml
import json
import pandas as pd
from pathlib import Path
import logging

from IncomeEstimation.src.utils.logging import setup_logger
from IncomeEstimation.src.utils.paths import get_project_root
from IncomeEstimation.prehoc.huber_threshold.model.huber_threshold_model import HuberThresholdModel

def train_huber_threshold_model(
    X_train, y_train, X_test, y_test, 
    config_path=None, 
    save_model=True, 
    save_metrics=True
):
    """
    Train the Huber Threshold model and evaluate its performance.
    
    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    config_path : str or Path, optional
        Path to the model configuration file
    save_model : bool, default=True
        Whether to save the trained model
    save_metrics : bool, default=True
        Whether to save the evaluation metrics
        
    Returns
    -------
    tuple
        Trained model, predictions, and evaluation metrics
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    
    # Default config path if none provided
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "prehoc" / "huber_threshold" / "config" / "model_config.yaml"
    
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    logger.info("Initializing Huber Threshold model")
    model = HuberThresholdModel(config_path=config_path)
    
    # Set up model save directory
    project_root = get_project_root()
    model_dir = project_root / "prehoc" / "huber_threshold" / "results" / "models"
    metrics_dir = project_root / "prehoc" / "huber_threshold" / "results" / "metrics"
    predictions_dir = project_root / "prehoc" / "huber_threshold" / "results" / "predictions"
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Train model
    logger.info("Training Huber Threshold model")
    model.train(
        X_train, y_train, X_test, y_test, 
        save_model=save_model, 
        model_dir=model_dir
    )
    
    # Make predictions
    logger.info("Making predictions")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate model
    logger.info("Evaluating model performance")
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    # Log metrics but don't save them
    logger.info("Train RMSE: {:.4f}".format(train_metrics.get('rmse', 0)))
    logger.info("Test RMSE: {:.4f}".format(test_metrics.get('rmse', 0)))
    logger.info("Train R²: {:.4f}".format(train_metrics.get('r2', 0)))
    logger.info("Test R²: {:.4f}".format(test_metrics.get('r2', 0)))
    
    # Skip saving metrics since they should be saved during inference only
    if save_metrics:
        logger.info("Metrics will be saved only during inference, not during training")
    
    # Combine metrics for return value
    evaluation = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_train_true': y_train,
        'y_train_pred': y_train_pred,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred
    })
    predictions_path = os.path.join(predictions_dir, 'huber_threshold_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    return model, (y_train_pred, y_test_pred), evaluation

if __name__ == "__main__":
    # Set up logger
    logger = setup_logger()
    
    # Load data
    logger.info("Loading data")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Split features and target
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    # Train model
    model, predictions, metrics = train_huber_threshold_model(
        X_train, y_train, X_test, y_test
    )
    
    # Print evaluation results
    logger.info("Training complete. Evaluation results:")
    logger.info(f"Train RMSE: {metrics['train']['rmse']:.4f}")
    logger.info(f"Test RMSE: {metrics['test']['rmse']:.4f}")
    logger.info(f"Train MAE: {metrics['train']['mae']:.4f}")
    logger.info(f"Test MAE: {metrics['test']['mae']:.4f}")
    logger.info(f"Train R2: {metrics['train']['r2']:.4f}")
    logger.info(f"Test R2: {metrics['test']['r2']:.4f}")
    logger.info(f"Train Threshold Exceedance: {metrics['train']['exceeds_threshold_pct']:.2f}%")
    logger.info(f"Test Threshold Exceedance: {metrics['test']['exceeds_threshold_pct']:.2f}%")
