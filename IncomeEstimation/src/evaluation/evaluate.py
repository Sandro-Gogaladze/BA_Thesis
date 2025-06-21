"""
Evaluation script for income estimation models
"""
import os
import argparse
import logging
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IncomeEstimation.src.utils.logging import setup_logger
from IncomeEstimation.src.utils.paths import get_project_root, get_processed_data_dir
from IncomeEstimation.src.toolkit.model_evaluator import ModelEvaluator

from IncomeEstimation.src.utils.logging import get_logger

# Initialize logger for evaluation
logger = get_logger('evaluation')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate income estimation models")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=["xgboost", "segment_aware", "huber_threshold", "quantile"],
        help="Model type to evaluate (the only required argument)"
    )
    
    # The following arguments are kept for backward compatibility
    # but are no longer shown in the help message
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help=argparse.SUPPRESS
    )
    
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=None,
        help=argparse.SUPPRESS
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help=argparse.SUPPRESS
    )
    
    # Add model comparison argument
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare all trained models (XGBoost, Quantile, Huber Threshold)",
        dest="compare_models"
    )
    
    # Hidden arguments that are always enabled for visualizations
    parser.add_argument(
        "--no-visualize", 
        dest="visualize",
        action="store_false",
        help=argparse.SUPPRESS
    )
    parser.set_defaults(visualize=True)
    
    return parser.parse_args()

def evaluate_xgboost(test_data, model_path, config_path, output_dir, visualize=True, feature_analysis=True, purple_theme=True):
    """
    Evaluate XGBoost model performance
    
    Parameters:
    -----------
    test_data : tuple
        Tuple containing (X_test, y_test)
    model_path : str
        Path to trained model
    config_path : str
        Path to model configuration
    output_dir : str
        Directory to save evaluation results
    visualize : bool
        Whether to generate visualizations (default: True)
    feature_analysis : bool
        Whether to run feature importance analysis including SHAP and PDP plots (default: True)
    purple_theme : bool
        Whether to use purple theme for visualizations (default: True)
    """
    from IncomeEstimation.baseline.xgboost.model.xgboost_model import XGBoostModel
    import subprocess
    from pathlib import Path
    
    X_test, y_test = test_data
    # Fix path to avoid nested baseline/baseline structure
    project_root = Path(get_project_root())
    results_dir = project_root / "baseline" / "xgboost" / "results"
    predictions_dir = results_dir / "predictions"
    figures_dir = results_dir / "figures"
    
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        error_msg = f"Model file not found at: {model_path}. Please run training first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check if predictions exist, if not run inference first
    predictions_path = predictions_dir / "xgboost_predictions.csv"
    
    if not os.path.exists(predictions_path):
        logger.info("No predictions found. Running inference first...")
        
        # Run inference script with just the model argument (simplified)
        cmd = [
            "python", 
            "-m", "IncomeEstimation.src.inference.predict",
            "--model", "xgboost"
        ]
        logger.info(f"Running inference command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Inference completed successfully, predictions saved to {predictions_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run inference: {str(e)}")
            raise
    else:
        logger.info(f"Found existing predictions at {predictions_path}")
    
    # Load the model
    logger.info(f"Loading XGBoost model from: {model_path}")
    model = XGBoostModel(config_path)
    try:
        model.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    # Make predictions (or use existing ones from file)
    logger.info("Making predictions and creating analysis dataframe...")
    y_pred = model.predict(X_test)
    
    # Create analysis DataFrame for test set
    test_analysis_df = model.evaluator.create_analysis_dataframe(y_test, y_pred)
    
    # Load the training data too
    processed_data_dir = get_processed_data_dir()
    train_path = processed_data_dir / "train.csv"
    train_data = pd.read_csv(train_path)
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    
    # Make predictions on training data
    y_train_pred = model.predict(X_train)
    
    # Create analysis DataFrame for train set
    train_analysis_df = model.evaluator.create_analysis_dataframe(y_train, y_train_pred)
    
    # Evaluate model on test set
    test_metrics = model.evaluate(X_test, y_test)
    test_segment_metrics = model.evaluate_segments(X_test, y_test)
    
    # Evaluate model on training set
    train_metrics = model.evaluator.calculate_basic_metrics(y_train, y_train_pred)
    train_segment_metrics = model.evaluator.calculate_segment_metrics(train_analysis_df)
    
    # Print metrics comparison
    model.evaluator.print_metrics_table(train_metrics, test_metrics)
    
    logger.info("Model performance metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value}")
    
    # Save evaluation results
    metrics_path = os.path.join(output_dir, "metrics.json")
    
    # Convert segment metrics dataframes to dictionaries for JSON storage
    train_segment_dict = train_segment_metrics.to_dict(orient='records')
    test_segment_dict = test_segment_metrics.to_dict(orient='records')
    
    # Create a comprehensive metrics object that includes segment metrics
    all_metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'train_segments': train_segment_dict,
        'test_segments': test_segment_dict
    }
    
    # Save to JSON
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    logger.info(f"All metrics (including segments) saved to: {metrics_path}")
    
    # Generate visualizations matching notebook's 4 output images
    if visualize:
        logger.info("Generating comprehensive visualizations...")
        
        # Ensure directories exist
        os.makedirs(figures_dir, exist_ok=True)
        logger.info(f"Saving figures to: {figures_dir}")
        
        # Use the model's evaluator directly - it already has all the methods we need
        evaluator = model.evaluator
        
        # 1. Generate comprehensive evaluation plots for both train and test data
        logger.info("1. Generating comprehensive evaluation plots...")
        
        # Generate train set analysis
        train_eval_path = os.path.join(figures_dir, "train_set_analysis.png")
        try:
            # Use the evaluator's plot_comprehensive_evaluation method with save_path
            evaluator.plot_comprehensive_evaluation(
                train_analysis_df, 
                dataset_name="Train Set Prediction Analysis", 
                enhanced=True,
                save_path=train_eval_path
            )
            logger.info(f"Train set comprehensive evaluation saved to: {train_eval_path}")
        except Exception as e:
            logger.error(f"Error generating train set comprehensive evaluation plot: {e}")
        
        # Generate test set analysis
        test_eval_path = os.path.join(figures_dir, "test_set_analysis.png")
        try:
            # Use the evaluator's plot_comprehensive_evaluation method with save_path
            evaluator.plot_comprehensive_evaluation(
                test_analysis_df, 
                dataset_name="Test Set Prediction Analysis", 
                enhanced=True,
                save_path=test_eval_path
            )
            logger.info(f"Test set comprehensive evaluation saved to: {test_eval_path}")
        except Exception as e:
            logger.error(f"Error generating test set comprehensive evaluation plot: {e}")
        
        # 2. Generate SHAP feature importance
        logger.info("2. Generating SHAP feature importance...")
        try:
            # Use the evaluator's plot_feature_importance method
            importance_df = evaluator.plot_feature_importance(
                model.model, 
                X_test, 
                n_top=15, 
                enhanced=True,
                figures_dir=figures_dir
            )
            
            # Save feature importance data
            importance_path = os.path.join(output_dir, "feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance data saved to: {importance_path}")
        except Exception as e:
            logger.error(f"Error generating SHAP feature importance: {e}")
            importance_df = None
        
        # 3. Generate feature dependence plots for top features
        logger.info("3. Generating feature dependence plots...")
        try:
            # Use the evaluator's plot_feature_dependence method
            evaluator.plot_feature_dependence(
                model.model,
                X_test,
                importance_df,
                top_n=5,
                enhanced=True,
                figures_dir=figures_dir
            )
            logger.info(f"Feature dependence plots generated")
        except Exception as e:
            logger.error(f"Error generating feature dependence plots: {e}")
        
        logger.info(f"All visualization plots saved to: {figures_dir}")
    
    return {'train': train_metrics, 'test': test_metrics}, test_segment_metrics

def evaluate_segment_aware(test_data, model_path, config_path, output_dir, visualize=True, feature_analysis=True, teal_theme=True):
    """
    Evaluate segment-aware model performance
    
    Parameters:
    -----------
    test_data : tuple
        Tuple containing (X_test, y_test)
    model_path : str
        Path to trained model
    config_path : str
        Path to model configuration
    output_dir : str
        Directory to save evaluation results
    visualize : bool, default=True
        Whether to generate visualizations
    feature_analysis : bool, default=True
        Whether to run feature importance analysis including SHAP and PDP plots
    teal_theme : bool, default=True
        Whether to use teal theme for visualizations
        
    Returns:
    --------
    tuple
        Tuple containing metrics and segment metrics
    """
    from IncomeEstimation.prehoc.segment_aware.model.segment_aware_model import SegmentAwareHuberThresholdModel
    from IncomeEstimation.src.toolkit.theme import Theme
    import json
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    X_test, y_test = test_data
    
    logger.info(f"Evaluating Segment-Aware Huber threshold model: {model_path}")
    logger.info(f"Using configuration: {config_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        error_msg = f"Model file not found at: {model_path}. Please run training first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load the model
    logger.info(f"Loading Segment-Aware model from: {model_path}")
    model = SegmentAwareHuberThresholdModel.load(model_path)
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Create analysis DataFrame for test set
    test_analysis_df = model.evaluator.create_analysis_dataframe(y_test, y_pred)
    
    # Evaluate model on test set
    test_metrics = model.evaluator.calculate_basic_metrics(y_test, y_pred, test_analysis_df)
    test_segment_metrics = model.evaluator.calculate_segment_metrics(test_analysis_df)
    
    logger.info("Model performance metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value}")
    
    # Save evaluation results
    metrics_path = os.path.join(output_dir, "segment_aware_metrics.json")
    
    # Convert segment metrics dataframes to dictionaries for JSON storage
    test_segment_dict = test_segment_metrics.to_dict(orient='records')
    
    # Create a comprehensive metrics object that includes segment metrics
    all_metrics = {
        'test': test_metrics,
        'test_segments': test_segment_dict
    }
    
    logger.info(f"Saving evaluation metrics to: {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Save predictions to predictions directory, not metrics directory
    predictions_dir = os.path.join(os.path.dirname(output_dir), "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_path = os.path.join(predictions_dir, "segment_aware_predictions.csv")
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'error': y_pred - y_test,
        'error_pct': (y_pred - y_test) / (y_test + 1e-6) * 100
    })
    
    logger.info(f"Saving predictions to: {predictions_path}")
    predictions_df.to_csv(predictions_path, index=False)
    
    # Generate visualizations if requested
    if visualize:
        logger.info("Generating visualizations...")
        figures_dir = os.path.join(os.path.dirname(output_dir), "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Use the teal theme for segment-aware model
        logger.info("Generating threshold exceedance visualization with teal theme...")
        from IncomeEstimation.src.toolkit.visualization_utils import plot_threshold_exceedance_by_model, run_feature_analysis, create_prehoc_model_comparison
        
        # Create segmented data for specialized visualization
        plot_threshold_exceedance_by_model(
            y_test,
            {'Segment-Aware Huber': y_pred},
            colors={'Segment-Aware Huber': '#1abc9c'},  # Teal color from theme
            output_path=os.path.join(figures_dir, "segment_aware_exceedance_by_range.png"),
            figsize=(10, 6)
        )
        
        # Generate model comparison visualization
        logger.info("Generating model comparison visualization...")
        try:
            create_prehoc_model_comparison(
                y_true=y_test,
                prehoc_pred=y_pred,
                prehoc_name="segment_aware_huber_threshold",
                color_theme='teal',
                output_path=os.path.join(figures_dir, "model_comparison.png"),
                figsize=(15, 10)
            )
            logger.info(f"Model comparison plot saved to: {os.path.join(figures_dir, 'model_comparison.png')}")
        except Exception as e:
            logger.error(f"Error generating model comparison: {e}")
        
        # Always run feature analysis with SHAP and PDP plots
        logger.info("Running comprehensive feature analysis (SHAP, feature importance, PDP)...")
        
        try:
            # Set the theme to teal
            from IncomeEstimation.src.toolkit.theme import Theme
            model.evaluator.theme = Theme('teal')
            
            # First, generate feature importance with SHAP values
            logger.info("Generating feature importance with SHAP...")
            importance_df = model.evaluator.plot_feature_importance(
                model.model, 
                X_test, 
                n_top=10, 
                enhanced=True,
                figures_dir=figures_dir
            )
            
            # Then generate feature dependence plots (SHAP + PDP together)
            logger.info("Generating feature dependence plots (SHAP + PDP)...")
            model.evaluator.plot_feature_dependence(
                model.model, 
                X_test, 
                importance_df=importance_df, 
                top_n=5, 
                enhanced=True, 
                figures_dir=figures_dir
            )
            
            logger.info(f"Feature analysis completed and saved to: {figures_dir}")
        except Exception as e:
            logger.error(f"Error running feature analysis: {e}")
    
    logger.info("Segment-Aware model evaluation completed successfully")
    
    return test_metrics, test_segment_metrics

def evaluate_huber_threshold(test_data, model_path, config_path, output_dir, visualize=True):
    """
    Evaluate Huber threshold model performance
    
    Parameters:
    -----------
    test_data : tuple
        Tuple containing (X_test, y_test)
    model_path : str
        Path to trained model
    config_path : str
        Path to model configuration
    output_dir : str
        Directory to save evaluation results
    visualize : bool, default=True
        Whether to generate visualizations
        
    Returns:
    --------
    tuple
        Tuple containing metrics and segment metrics
    """
    from IncomeEstimation.prehoc.huber_threshold.model.huber_threshold_model import HuberThresholdModel
    from IncomeEstimation.src.toolkit.theme import Theme
    import json
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    X_test, y_test = test_data
    
    logger.info(f"Evaluating Huber threshold model: {model_path}")
    logger.info(f"Using configuration: {config_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = HuberThresholdModel.load(model_path)
    
    # Make predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Create analysis dataframe
    logger.info("Creating analysis dataframe...")
    evaluator = model.evaluator
    analysis_df = evaluator.create_analysis_dataframe(y_test, y_pred)
    
    # Calculate metrics (for logging purposes only)
    logger.info("Calculating metrics...")
    metrics = evaluator.calculate_basic_metrics(y_test, y_pred, analysis_df)
    
    # Calculate segment metrics (for logging purposes only)
    logger.info("Calculating segment metrics...")
    segment_metrics = evaluator.calculate_segment_metrics(analysis_df)
    
    # Log that metrics are not saved during evaluation
    logger.info("Metrics will be saved only during inference, not during evaluation")
    
    # Generate visualizations if requested
    if visualize:
        logger.info("Generating visualizations...")
        figures_dir = os.path.join(os.path.dirname(output_dir), "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Always use the purple theme
        logger.info("Generating threshold exceedance visualization with purple theme...")
        from IncomeEstimation.src.toolkit.visualization_utils import plot_threshold_exceedance_by_model, run_feature_analysis, create_prehoc_model_comparison
        
        # Create segmented data for specialized visualization
        plot_threshold_exceedance_by_model(
            y_test,
            {'Huber Threshold': y_pred},
            colors={'Huber Threshold': '#9b59b6'},  # Purple color
            output_path=os.path.join(figures_dir, "huber_threshold_exceedance_by_range.png"),
            figsize=(10, 6)
        )
        
        # Generate model comparison visualization
        logger.info("Generating model comparison visualization...")
        try:
            create_prehoc_model_comparison(
                y_true=y_test,
                prehoc_pred=y_pred,
                prehoc_name="huber_plus_threshold_loss",
                color_theme='purple',
                output_path=os.path.join(figures_dir, "model_comparison.png"),
                figsize=(15, 10)
            )
            logger.info(f"Model comparison plot saved to: {os.path.join(figures_dir, 'model_comparison.png')}")
        except Exception as e:
            logger.error(f"Error generating model comparison: {e}")
        
        # Always run feature analysis with SHAP and PDP plots
        logger.info("Running comprehensive feature analysis (SHAP, feature importance, PDP)...")
        
        try:
            # Set the theme to purple
            from IncomeEstimation.src.toolkit.theme import Theme
            evaluator.theme = Theme('purple')
            
            # First, generate feature importance with SHAP values
            logger.info("Generating feature importance with SHAP...")
            importance_df = evaluator.plot_feature_importance(
                model.model, 
                X_test, 
                n_top=10, 
                enhanced=True,
                figures_dir=figures_dir
            )
            
            # Then generate feature dependence plots (SHAP + PDP together)
            logger.info("Generating feature dependence plots (SHAP + PDP)...")
            evaluator.plot_feature_dependence(
                model.model, 
                X_test, 
                importance_df=importance_df, 
                top_n=5, 
                enhanced=True, 
                figures_dir=figures_dir
            )
            
            logger.info(f"Feature analysis completed and saved to: {figures_dir}")
        except Exception as e:
            logger.error(f"Error running feature analysis: {e}")
    
    # Log key metrics
    logger.info("Evaluation complete. Key metrics:")
    logger.info(f"• RMSE: {metrics['rmse']:.4f}")
    logger.info(f"• MAE: {metrics['mae']:.4f}")
    logger.info(f"• R²: {metrics['r2']:.4f}")
    logger.info(f"• Within 10%: {metrics['within_10pct']:.2f}%")
    logger.info(f"• Within 20%: {metrics['within_20pct']:.2f}%")
    logger.info(f"• Threshold exceedance: {metrics['exceeds_threshold_pct']:.2f}%")
    
    return metrics, segment_metrics

def evaluate_quantile(test_data, model_path, config_path, output_dir, visualize=True):
    """
    Evaluate quantile regression model performance
    
    Parameters:
    -----------
    test_data : tuple
        Tuple containing (X_test, y_test)
    model_path : str
        Path to trained quantile model
    config_path : str
        Path to model configuration
    output_dir : str
        Directory to save evaluation results
    visualize : bool
        Whether to generate visualizations (default: True)
    """
    from IncomeEstimation.posthoc.quantile.model.quantile_model import QuantileRegressionModel, load_config
    from IncomeEstimation.baseline.xgboost.model.xgboost_model import XGBoostModel
    from IncomeEstimation.src.toolkit.model_evaluator import ModelEvaluator
    import matplotlib.pyplot as plt
    import joblib
    import numpy as np
    import json
    from pathlib import Path
    
    X_test, y_test = test_data
    
    # Set up project paths
    project_root = Path(get_project_root())
    results_dir = project_root / "posthoc" / "quantile" / "results"
    predictions_dir = results_dir / "predictions"
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None
    
    # Load configuration
    config = load_config(config_path)
    quantile = config.get('quantile', 0.2)
    absolute_threshold = config.get('absolute_threshold', 200)
    percentage_threshold = config.get('percentage_threshold', 20)
    
    # First, we need to load the baseline XGBoost model
    xgboost_model_path = project_root / "baseline" / "xgboost" / "results" / "models" / "xgboost_model.joblib"
    
    if not os.path.exists(xgboost_model_path):
        logger.error(f"Baseline XGBoost model not found at: {xgboost_model_path}")
        return None, None
    
    logger.info(f"Loading baseline XGBoost model from: {xgboost_model_path}")
    xgboost_model = XGBoostModel.load(xgboost_model_path)
    
    # Load quantile model
    logger.info(f"Loading quantile model from: {model_path}")
    model = QuantileRegressionModel.load(model_path, xgboost_model)
    
    # Generate base model predictions
    logger.info("Generating base model predictions...")
    base_predictions = xgboost_model.predict(X_test)
    
    # Generate calibrated predictions
    logger.info(f"Generating calibrated predictions with quantile={quantile}...")
    calibrated_predictions = model.predict(X_test)
    
    # Create evaluator for both models
    evaluator = ModelEvaluator(
        dynamic_threshold_absolute=absolute_threshold,
        dynamic_threshold_percentage=percentage_threshold
    )
    
    # Calculate analysis dataframes
    base_analysis_df = evaluator.create_analysis_dataframe(y_test, base_predictions)
    calibrated_analysis_df = evaluator.create_analysis_dataframe(y_test, calibrated_predictions)
    
    # Calculate metrics
    base_metrics = evaluator.calculate_basic_metrics(y_test, base_predictions, base_analysis_df)
    calibrated_metrics = evaluator.calculate_basic_metrics(y_test, calibrated_predictions, calibrated_analysis_df)
    
    # Calculate segment metrics
    base_segment_metrics = evaluator.calculate_segment_metrics(base_analysis_df)
    calibrated_segment_metrics = evaluator.calculate_segment_metrics(calibrated_analysis_df)
    
    # Save predictions to csv
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'base_prediction': base_predictions,
        'calibrated_prediction': calibrated_predictions,
        'base_error': base_predictions - y_test,
        'calibrated_error': calibrated_predictions - y_test,
        'base_error_pct': (base_predictions - y_test) / (y_test + 1e-6) * 100,
        'calibrated_error_pct': (calibrated_predictions - y_test) / (y_test + 1e-6) * 100
    })
    
    # Add dynamic threshold and exceedance calculations
    predictions_df['dynamic_threshold'] = np.maximum(
        absolute_threshold, y_test * (percentage_threshold / 100)
    )
    predictions_df['base_exceeds_threshold'] = (
        predictions_df['base_error'] > predictions_df['dynamic_threshold']
    ).astype(int)
    predictions_df['calibrated_exceeds_threshold'] = (
        predictions_df['calibrated_error'] > predictions_df['dynamic_threshold']
    ).astype(int)
    
    # Add income range segmentation
    predictions_df['income_range'] = pd.cut(
        y_test,
        bins=[0, 1500, 2500, float('inf')],
        labels=["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
    )
    
    # Check if predictions file already exists from inference step
    predictions_path = os.path.join(predictions_dir, "quantile_predictions.csv")
    if not os.path.exists(predictions_path):
        # Only save if the file doesn't exist
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
    else:
        logger.info(f"Using existing predictions file: {predictions_path}")
    
    # Convert segment metrics dataframe to dictionary for JSON storage
    segment_metrics_list = []
    for _, row in calibrated_segment_metrics.iterrows():
        segment_metrics_list.append(row.to_dict())
    
    # Combine metrics
    combined_metrics = {
        'base': base_metrics,
        'calibrated': calibrated_metrics,
        'segment_metrics': segment_metrics_list,
        'quantile': quantile,
        'beta_0': model.beta_0,
        'beta_1': model.beta_1,
        'formula': f"y_calibrated = {model.beta_0:.4f} + {model.beta_1:.4f} * y_base"
    }
    
    # Save metrics to json
    metrics_path = os.path.join(metrics_dir, "quantile_evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(combined_metrics, f, indent=4)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # No need to save separate CSV files for segment metrics
    # We'll include segment metrics in the JSON file
    
    # Log metrics
    logger.info("Base model metrics:")
    for metric, value in base_metrics.items():
        logger.info(f"  {metric}: {value}")
    
    logger.info("Calibrated model metrics:")
    for metric, value in calibrated_metrics.items():
        logger.info(f"  {metric}: {value}")
    
    if visualize:
        logger.info("Generating visualizations...")
        
        # 1. Generate comprehensive evaluation plot for calibrated model only
        logger.info("1. Generating comprehensive evaluation plot for calibrated model...")
        
        # Create a new evaluator with green theme specifically for quantile model
        green_theme_evaluator = ModelEvaluator(
            theme='green',
            dynamic_threshold_absolute=absolute_threshold,
            dynamic_threshold_percentage=percentage_threshold
        )
        
        # Generate calibrated model comprehensive evaluation - this is the main visualization we need
        calibrated_eval_path = os.path.join(figures_dir, "quantile_model_analysis.png")
        try:
            green_theme_evaluator.plot_comprehensive_evaluation(
                calibrated_analysis_df,
                dataset_name="Test Set (Adjusted) Prediction Analysis",
                enhanced=True,
                save_path=calibrated_eval_path
            )
            logger.info(f"Calibrated model comprehensive evaluation saved to: {calibrated_eval_path}")
        except Exception as e:
            logger.error(f"Error generating calibrated model comprehensive evaluation plot: {e}")
        
        # 2. Calibration function plot
        plt.figure(figsize=(10, 6))
        plt.scatter(base_predictions, calibrated_predictions, alpha=0.5, s=20)
        
        # Plot the calibration line
        x_range = np.linspace(min(base_predictions), max(base_predictions), 100)
        y_range = model.beta_0 + model.beta_1 * x_range
        plt.plot(x_range, y_range, 'r-', linewidth=2, 
                label=f'y = {model.beta_0:.2f} + {model.beta_1:.2f}x (q={model.quantile})')
        
        # Identity line for reference
        plt.plot(x_range, x_range, 'k--', alpha=0.7, label='y = x (no calibration)')
        
        plt.title('Quantile Regression Calibration')
        plt.xlabel('Original Predictions')
        plt.ylabel('Calibrated Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        calibration_plot_path = os.path.join(figures_dir, "calibration_function.png")
        plt.savefig(calibration_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Calibration function plot saved to: {calibration_plot_path}")
        
        # 3. Comparison of error distributions
        plt.figure(figsize=(12, 8))
        
        # Error distribution
        plt.subplot(2, 2, 1)
        plt.hist(base_predictions - y_test, bins=50, alpha=0.5, label='Base Model', color='blue')
        plt.hist(calibrated_predictions - y_test, bins=50, alpha=0.5, label='Calibrated Model', color='green')
        plt.axvline(0, color='k', linestyle='--')
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Percentage error distribution
        plt.subplot(2, 2, 2)
        plt.hist((base_predictions - y_test) / (y_test + 1e-6) * 100, 
                bins=50, range=(-50, 100), alpha=0.5, label='Base Model', color='blue')
        plt.hist((calibrated_predictions - y_test) / (y_test + 1e-6) * 100, 
                bins=50, range=(-50, 100), alpha=0.5, label='Calibrated Model', color='green')
        plt.axvline(0, color='k', linestyle='--')
        plt.axvline(20, color='r', linestyle='--', label='20% Threshold')
        plt.title('Percentage Error Distribution')
        plt.xlabel('Percentage Error')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Actual vs predicted scatter plots
        plt.subplot(2, 2, 3)
        plt.scatter(y_test, base_predictions, alpha=0.3, s=20, color='blue')
        plt.plot([0, max(y_test)], [0, max(y_test)], 'k--')
        plt.title('Base Model: Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        plt.subplot(2, 2, 4)
        plt.scatter(y_test, calibrated_predictions, alpha=0.3, s=20, color='green')
        plt.plot([0, max(y_test)], [0, max(y_test)], 'k--')
        plt.title('Calibrated Model: Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(figures_dir, "model_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Model comparison plot saved to: {comparison_plot_path}")
        
        # 4. Threshold exceedance by income range
        plt.figure(figsize=(10, 6))
        
        # Prepare data for income range plot
        ranges = base_segment_metrics[base_segment_metrics['segment'] != 'All Data']['segment']
        base_exceedance = base_segment_metrics[base_segment_metrics['segment'] != 'All Data']['exceeds_threshold_pct']
        calibrated_exceedance = calibrated_segment_metrics[calibrated_segment_metrics['segment'] != 'All Data']['exceeds_threshold_pct']
        
        x_pos = np.arange(len(ranges))
        width = 0.35
        
        plt.bar(x_pos - width/2, base_exceedance, width, label='Base Model', color='blue', alpha=0.7)
        plt.bar(x_pos + width/2, calibrated_exceedance, width, label='Calibrated Model', color='green', alpha=0.7)
        
        plt.title(f'Threshold Exceedance by Income Range (max({absolute_threshold}, {percentage_threshold}%))')
        plt.xlabel('Income Range')
        plt.ylabel('Exceedance Rate (%)')
        plt.xticks(x_pos, ranges)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        exceedance_plot_path = os.path.join(figures_dir, "exceedance_by_range.png")
        plt.savefig(exceedance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Exceedance by range plot saved to: {exceedance_plot_path}")
        
        # We are no longer creating the "Error vs Actual Value with Dynamic Threshold" plot
        # as per user's requirements
    
    # Return metrics for potential further analysis
    return combined_metrics, calibrated_segment_metrics



def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Setup logging
    setup_logger('evaluation')
    logger.info(f"Starting evaluation for model: {args.model}")
    
    # Set up paths
    project_root = get_project_root()
    
    if args.config_path:
        config_path = Path(args.config_path)
    else:
        if args.model == "xgboost":
            config_path = project_root / "baseline" / "xgboost" / "config" / "model_config.yaml"
        elif args.model == "segment_aware":
            config_path = project_root / "prehoc" / "segment_aware" / "config" / "model_config.yaml"
        elif args.model == "huber_threshold":
            config_path = project_root / "prehoc" / "huber_threshold" / "config" / "model_config.yaml"
        elif args.model == "quantile":
            config_path = project_root / "posthoc" / "quantile" / "config" / "model_config.yaml"
    
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        if args.model == "xgboost":
            model_path = project_root / "baseline" / "xgboost" / "results" / "models" / "xgboost_model.joblib"
        elif args.model == "segment_aware":
            model_path = project_root / "prehoc" / "segment_aware" / "results" / "models" / "segment_aware_model.joblib"
        elif args.model == "huber_threshold":
            model_path = project_root / "prehoc" / "huber_threshold" / "results" / "models" / "huber_threshold_model.joblib"
        elif args.model == "quantile":
            model_path = project_root / "posthoc" / "quantile" / "results" / "models" / "quantile_model.joblib"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.model == "xgboost":
            output_dir = project_root / "baseline" / "xgboost" / "results" / "metrics"
        elif args.model == "segment_aware":
            output_dir = project_root / "prehoc" / "segment_aware" / "results" / "metrics"
        elif args.model == "huber_threshold":
            output_dir = project_root / "prehoc" / "huber_threshold" / "results" / "metrics"
        elif args.model == "quantile":
            output_dir = project_root / "posthoc" / "quantile" / "results" / "metrics"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed test data directly
    processed_data_dir = get_processed_data_dir()
    test_path = processed_data_dir / "test.csv"
    
    logger.info(f"Loading processed test data from: {test_path}")
    test_data = pd.read_csv(test_path)
    
    # Extract features and target
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    logger.info(f"Loaded processed test data: {X_test.shape[0]} samples with {X_test.shape[1]} features")
    
    # Evaluate model based on type
    try:
        if args.model == "xgboost":
            metrics, segment_metrics = evaluate_xgboost(
                (X_test, y_test), model_path, config_path, output_dir, visualize=args.visualize
            )
        elif args.model == "segment_aware":
            metrics, segment_metrics = evaluate_segment_aware(
                (X_test, y_test), model_path, config_path, output_dir, visualize=args.visualize
            )
        elif args.model == "huber_threshold":
            # Always run visualizations for huber_threshold model
            metrics, segment_metrics = evaluate_huber_threshold(
                (X_test, y_test), model_path, config_path, output_dir, visualize=True
            )
        elif args.model == "quantile":
            metrics, segment_metrics = evaluate_quantile(
                (X_test, y_test), model_path, config_path, output_dir, visualize=args.visualize
            )
        
        # Check if metrics is a dict with train and test keys (new format)
        if isinstance(metrics, dict) and 'train' in metrics and 'test' in metrics:
            train_metrics = metrics['train']
            test_metrics = metrics['test']
            logger.info("Evaluation completed successfully for both train and test sets")
            
            # Print a summary of key metrics
            logger.info("\nKey Metrics Summary (Train / Test):")
            logger.info(f"RMSE: {train_metrics['rmse']:.4f} / {test_metrics['rmse']:.4f}")
            logger.info(f"R²: {train_metrics['r2']:.4f} / {test_metrics['r2']:.4f}")
            logger.info(f"Within 20%: {train_metrics['within_20pct']:.2f}% / {test_metrics['within_20pct']:.2f}%")
            
        else:
            # Handle legacy format for backward compatibility
            logger.info("Evaluation completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
        logger.error("Please run training first with: python -m IncomeEstimation.src.train.train --model xgboost")
        return
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return

if __name__ == "__main__":
    main()
