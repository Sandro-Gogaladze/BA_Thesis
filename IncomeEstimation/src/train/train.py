"""
Main training script for all models
"""
import os
import argparse
import logging
from pathlib import Path

import pandas as pd

from IncomeEstimation.src.utils.logging import setup_logger
from IncomeEstimation.src.utils.paths import get_project_root, get_processed_data_dir

from IncomeEstimation.src.utils.logging import get_logger

# Initialize logger for training
logger = get_logger('training')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train income estimation models")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=["xgboost", "segment_aware", "huber_threshold", "quantile"],
        help="Model type to train"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save model and results"
    )
    
    return parser.parse_args()

def train_xgboost(train_data, test_data, config_path, output_dir):
    """
    Train XGBoost baseline model
    
    Parameters:
    -----------
    train_data : tuple
        Tuple containing (X_train, y_train)
    test_data : tuple
        Tuple containing (X_test, y_test)
    config_path : str
        Path to model configuration
    output_dir : str
        Directory to save model and results
    """
    from IncomeEstimation.baseline.xgboost.model.xgboost_model import XGBoostModel
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    logger.info(f"Training XGBoost model with config: {config_path}")
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Initialize and train model
    model = XGBoostModel(config_path)
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    segment_metrics = model.evaluate_segments(X_test, y_test)
    
    logger.info("Model performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")
    
    # Save model and results
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost_model.joblib")
    
    model.save_model(model_path)
    
    logger.info(f"Model saved to: {model_path}")
    
    return model, metrics, segment_metrics

def train_segment_aware(train_data, test_data, config_path, output_dir):
    """
    Train Segment-Aware Huber threshold model with custom objective function combining 
    segment-specific Huber loss with penalties for exceeding dynamic thresholds.
    
    Parameters:
    -----------
    train_data : tuple
        Tuple containing (X_train, y_train)
    test_data : tuple
        Tuple containing (X_test, y_test)
    config_path : str
        Path to model configuration
    output_dir : str
        Directory to save model and results
    """
    from IncomeEstimation.prehoc.segment_aware.model.segment_aware_model import SegmentAwareHuberThresholdModel
    import joblib
    import os
    import yaml
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    logger.info(f"Training Segment-Aware Huber threshold model with config: {config_path}")
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Create model directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model with configuration
    model = SegmentAwareHuberThresholdModel(config_path=config_path)
    
    # Fit model
    logger.info("Fitting Segment-Aware Huber threshold model...")
    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test
    )
    
    # Save the model
    model_output_path = os.path.join(output_dir, "segment_aware_model.joblib")
    logger.info(f"Saving model to: {model_output_path}")
    model.save(model_output_path)
    
    # Create evaluator
    logger.info("Creating model evaluator...")
    evaluator = model.evaluator
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Create analysis dataframes
    train_df = evaluator.create_analysis_dataframe(y_train, y_train_pred)
    test_df = evaluator.create_analysis_dataframe(y_test, y_test_pred)
    
    # Calculate metrics
    train_metrics = evaluator.calculate_basic_metrics(y_train, y_train_pred, train_df)
    test_metrics = evaluator.calculate_basic_metrics(y_test, y_test_pred, test_df)
    
    # Calculate segment metrics
    train_segment_metrics = evaluator.calculate_segment_metrics(train_df)
    test_segment_metrics = evaluator.calculate_segment_metrics(test_df)
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    metrics_output_path = os.path.join(output_dir, "segment_aware_metrics.json")
    logger.info(f"Saving metrics to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save segment metrics
    train_segment_metrics_path = os.path.join(output_dir, "segment_aware_train_segment_metrics.csv")
    test_segment_metrics_path = os.path.join(output_dir, "segment_aware_test_segment_metrics.csv")
    
    train_segment_metrics.to_csv(train_segment_metrics_path, index=False)
    test_segment_metrics.to_csv(test_segment_metrics_path, index=False)
    
    # Save predictions
    predictions_dir = os.path.join(os.path.dirname(output_dir), "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'error': y_test_pred - y_test,
        'error_pct': (y_test_pred - y_test) / y_test * 100
    })
    
    predictions_path = os.path.join(predictions_dir, "segment_aware_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    # Log performance summary
    logger.info("Model performance metrics:")
    for metric, value in train_metrics.items():
        logger.info(f"  Train {metric}: {value}")
    for metric, value in test_metrics.items():
        logger.info(f"  Test {metric}: {value}")
    
    return model, metrics, test_segment_metrics

def train_huber_threshold(train_data, test_data, config_path, output_dir):
    """
    Train Huber threshold model with custom objective function combining Huber loss
    with a penalty for exceeding dynamic thresholds.
    
    Parameters:
    -----------
    train_data : tuple
        Tuple containing (X_train, y_train)
    test_data : tuple
        Tuple containing (X_test, y_test)
    config_path : str
        Path to model configuration
    output_dir : str
        Directory to save model and results
    """
    from IncomeEstimation.prehoc.huber_threshold.model.huber_threshold_model import HuberThresholdModel
    import joblib
    import os
    import yaml
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    logger.info(f"Training Huber threshold model with config: {config_path}")
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Create model directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model with configuration
    model = HuberThresholdModel(config_path=config_path)
    
    # Fit model
    logger.info("Fitting Huber threshold model...")
    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test
    )
    
    # Save the model
    model_output_path = os.path.join(output_dir, "huber_threshold_model.joblib")
    logger.info(f"Saving model to: {model_output_path}")
    model.save(model_output_path)
    
    # Create evaluator
    logger.info("Creating model evaluator...")
    evaluator = model.evaluator
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Create analysis dataframes
    train_df = evaluator.create_analysis_dataframe(y_train, y_train_pred)
    test_df = evaluator.create_analysis_dataframe(y_test, y_test_pred)
    
    # Calculate metrics
    train_metrics = evaluator.calculate_basic_metrics(y_train, y_train_pred, train_df)
    test_metrics = evaluator.calculate_basic_metrics(y_test, y_test_pred, test_df)
    
    # Calculate segment metrics
    train_segment_metrics = evaluator.calculate_segment_metrics(train_df)
    test_segment_metrics = evaluator.calculate_segment_metrics(test_df)
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    metrics_output_path = os.path.join(output_dir, "huber_threshold_metrics.json")
    logger.info(f"Saving metrics to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save segment metrics
    train_segment_metrics_path = os.path.join(output_dir, "huber_threshold_train_segment_metrics.csv")
    test_segment_metrics_path = os.path.join(output_dir, "huber_threshold_test_segment_metrics.csv")
    
    train_segment_metrics.to_csv(train_segment_metrics_path, index=False)
    test_segment_metrics.to_csv(test_segment_metrics_path, index=False)
    
    # Save predictions
    predictions_dir = os.path.join(os.path.dirname(output_dir), "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'error': y_test_pred - y_test,
        'error_pct': (y_test_pred - y_test) / y_test * 100
    })
    
    predictions_path = os.path.join(predictions_dir, "huber_threshold_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    # Log key metrics
    logger.info(f"Training RMSE: {train_metrics['rmse']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Training MAE: {train_metrics['mae']:.4f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
    logger.info(f"Training R²: {train_metrics['r2']:.4f}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Training exceedance rate: {train_metrics['exceeds_threshold_pct']:.2f}%")
    logger.info(f"Test exceedance rate: {test_metrics['exceeds_threshold_pct']:.2f}%")
    
    return model, metrics, test_segment_metrics

def train_quantile(train_data, test_data, config_path, output_dir):
    """
    Train quantile regression model for post-hoc calibration of XGBoost
    
    Parameters:
    -----------
    train_data : tuple
        Tuple containing (X_train, y_train)
    test_data : tuple
        Tuple containing (X_test, y_test)
    config_path : str
        Path to model configuration
    output_dir : str
        Directory to save model and results
    """
    from IncomeEstimation.posthoc.quantile.model.quantile_model import QuantileRegressionModel, load_config
    from IncomeEstimation.baseline.xgboost.model.xgboost_model import XGBoostModel
    import joblib
    import os
    import yaml
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    logger.info(f"Training quantile regression model with config: {config_path}")
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Load configuration
    config = load_config(config_path)
    quantile = config.get('quantile', 0.2)
    random_state = config.get('random_state', 42)
    absolute_threshold = config.get('absolute_threshold', 200)
    percentage_threshold = config.get('percentage_threshold', 20)
    
    # First, we need to load the baseline XGBoost model
    project_root = get_project_root()
    xgboost_model_path = project_root / "baseline" / "xgboost" / "results" / "models" / "xgboost_model.joblib"
    
    if not os.path.exists(xgboost_model_path):
        logger.error(f"Baseline XGBoost model not found at: {xgboost_model_path}")
        logger.error("Please train the XGBoost model first using: --model xgboost")
        raise FileNotFoundError(f"Baseline model not found: {xgboost_model_path}")
    
    logger.info(f"Loading baseline XGBoost model from: {xgboost_model_path}")
    xgboost_model = XGBoostModel.load(xgboost_model_path)
    
    # Initialize quantile regression model with the base model
    model = QuantileRegressionModel(
        base_model=xgboost_model,
        quantile=quantile,
        random_state=random_state
    )
    
    # Get base model predictions
    logger.info("Generating base model predictions for calibration...")
    y_train_pred = xgboost_model.predict(X_train)
    y_test_pred = xgboost_model.predict(X_test)
    
    # Set numpy random seed explicitly to match notebook exactly
    np.random.seed(random_state)
    
    # Fit quantile regression model
    logger.info(f"Fitting quantile regression with q={quantile}...")
    model.fit(X_train, y_train)
    
    logger.info(f"Quantile regression coefficients: β₀={model.beta_0:.4f}, β₁={model.beta_1:.4f}")
    
    # Generate calibrated predictions
    y_train_calibrated = model.predict(X_train)
    y_test_calibrated = model.predict(X_test)
    
    # Calculate evaluation metrics
    metrics = {}
    
    # Basic accuracy metrics (before calibration)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics['base_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
    metrics['base_mae'] = mean_absolute_error(y_test, y_test_pred)
    metrics['base_r2'] = r2_score(y_test, y_test_pred)
    
    # Basic accuracy metrics (after calibration)
    metrics['calibrated_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_calibrated))
    metrics['calibrated_mae'] = mean_absolute_error(y_test, y_test_calibrated)
    metrics['calibrated_r2'] = r2_score(y_test, y_test_calibrated)
    
    # Threshold violation metrics (before calibration)
    base_overestimation = model.compute_overestimation_ratio(y_test, y_test_pred, threshold=percentage_threshold/100)
    base_exceeds = model.compute_dynamic_threshold_exceedance(
        y_test, y_test_pred, absolute=absolute_threshold, percentage=percentage_threshold/100
    )
    
    metrics['base_overestimation_pct'] = base_overestimation * 100
    metrics['base_exceeds_threshold_pct'] = base_exceeds * 100
    
    # Threshold violation metrics (after calibration)
    calibrated_overestimation = model.compute_overestimation_ratio(y_test, y_test_calibrated, threshold=percentage_threshold/100)
    calibrated_exceeds = model.compute_dynamic_threshold_exceedance(
        y_test, y_test_calibrated, absolute=absolute_threshold, percentage=percentage_threshold/100
    )
    
    metrics['calibrated_overestimation_pct'] = calibrated_overestimation * 100
    metrics['calibrated_exceeds_threshold_pct'] = calibrated_exceeds * 100
    
    # Calculate segment-specific metrics
    segments = pd.cut(
        y_test,
        bins=[0, 1500, 2500, float('inf')],
        labels=["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
    )
    
    segment_metrics = []
    
    # Add All Data segment
    segment_metrics.append({
        'segment': 'All Data',
        'count': len(y_test),
        'base_exceeds_threshold_pct': base_exceeds * 100,
        'calibrated_exceeds_threshold_pct': calibrated_exceeds * 100,
        'improvement_pct': (base_exceeds - calibrated_exceeds) * 100
    })
    
    # Calculate metrics per segment
    for segment_name in segments.unique():
        mask = segments == segment_name
        y_true_segment = y_test[mask]
        y_pred_segment = y_test_pred[mask]
        y_calibrated_segment = y_test_calibrated[mask]
        
        # Calculate threshold exceedance for segment
        base_segment_exceeds = model.compute_dynamic_threshold_exceedance(
            y_true_segment, y_pred_segment, 
            absolute=absolute_threshold, percentage=percentage_threshold/100
        )
        
        calibrated_segment_exceeds = model.compute_dynamic_threshold_exceedance(
            y_true_segment, y_calibrated_segment,
            absolute=absolute_threshold, percentage=percentage_threshold/100
        )
        
        segment_metrics.append({
            'segment': segment_name,
            'count': len(y_true_segment),
            'base_exceeds_threshold_pct': base_segment_exceeds * 100,
            'calibrated_exceeds_threshold_pct': calibrated_segment_exceeds * 100,
            'improvement_pct': (base_segment_exceeds - calibrated_segment_exceeds) * 100
        })
    
    # Convert to DataFrame for easier reporting
    segment_metrics_df = pd.DataFrame(segment_metrics)
    
    # Print metrics summary
    logger.info("Model performance metrics:")
    logger.info(f"  Base RMSE: {metrics['base_rmse']:.4f}, Calibrated RMSE: {metrics['calibrated_rmse']:.4f}")
    logger.info(f"  Base R²: {metrics['base_r2']:.4f}, Calibrated R²: {metrics['calibrated_r2']:.4f}")
    logger.info(f"  Base Threshold Exceedance: {metrics['base_exceeds_threshold_pct']:.2f}%, " 
                f"Calibrated: {metrics['calibrated_exceeds_threshold_pct']:.2f}%")
    
    logger.info("Segment-specific threshold exceedance:")
    for _, row in segment_metrics_df.iterrows():
        logger.info(f"  {row['segment']}: Base {row['base_exceeds_threshold_pct']:.2f}% → "
                    f"Calibrated {row['calibrated_exceeds_threshold_pct']:.2f}% "
                    f"(Improvement: {row['improvement_pct']:.2f}%)")
    
    # No need to create visualization here, it will be done in evaluation
    
    # Save model
    model_path = os.path.join(output_dir, "quantile_model.joblib")
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    logger.info("Training complete. Visualization and metrics will be generated during evaluation.")
    
    # We don't need to save metrics files or visualizations here
    
    return model, metrics, segment_metrics_df

def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    setup_logger('training')
    logger.info(f"Starting training for model: {args.model}")
    
    # Set up paths
    project_root = get_project_root()
    
    if args.config:
        config_path = Path(args.config)
    else:
        if args.model == "xgboost":
            config_path = project_root / "baseline" / "xgboost" / "config" / "model_config.yaml"
        elif args.model == "segment_aware":
            config_path = project_root / "prehoc" / "segment_aware" / "config" / "model_config.yaml"
        elif args.model == "huber_threshold":
            config_path = project_root / "prehoc" / "huber_threshold" / "config" / "model_config.yaml"
        elif args.model == "quantile":
            config_path = project_root / "posthoc" / "quantile" / "config" / "model_config.yaml"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.model == "xgboost":
            output_dir = project_root / "baseline" / "xgboost" / "results" / "models"
        elif args.model == "segment_aware":
            output_dir = project_root / "prehoc" / "segment_aware" / "results" / "models"
        elif args.model == "huber_threshold":
            output_dir = project_root / "prehoc" / "huber_threshold" / "results" / "models"
        elif args.model == "quantile":
            output_dir = project_root / "posthoc" / "quantile" / "results" / "models"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed data directly
    processed_data_dir = get_processed_data_dir()
    
    # Load training data
    train_path = processed_data_dir / "train.csv"
    logger.info(f"Loading processed training data from: {train_path}")
    train_data = pd.read_csv(train_path)
    
    # Load test data
    test_path = processed_data_dir / "test.csv"
    logger.info(f"Loading processed test data from: {test_path}")
    test_data = pd.read_csv(test_path)
    
    # Extract features and target
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    logger.info(f"Loaded processed data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    
    # Train model based on type
    if args.model == "xgboost":
        model, metrics, segment_metrics = train_xgboost(
            (X_train, y_train), (X_test, y_test), config_path, output_dir
        )
    elif args.model == "segment_aware":
        model, metrics, segment_metrics = train_segment_aware(
            (X_train, y_train), (X_test, y_test), config_path, output_dir
        )
    elif args.model == "huber_threshold":
        model, metrics, segment_metrics = train_huber_threshold(
            (X_train, y_train), (X_test, y_test), config_path, output_dir
        )
    elif args.model == "quantile":
        model, metrics, segment_metrics = train_quantile(
            (X_train, y_train), (X_test, y_test), config_path, output_dir
        )
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
