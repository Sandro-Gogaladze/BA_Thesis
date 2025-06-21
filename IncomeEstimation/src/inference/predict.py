"""
Inference script for income estimation models
"""
import os
import argparse
import logging
import json
import pandas as pd
from pathlib import Path

from IncomeEstimation.src.utils.logging import setup_logger
from IncomeEstimation.src.utils.paths import get_project_root

from IncomeEstimation.src.utils.logging import get_logger

# Initialize logger for inference
logger = get_logger('inference')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with income estimation models")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=["xgboost", "segment_aware", "huber_threshold", "quantile"],
        help="Model type to use for inference"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to trained model (default: auto-detect based on model type)"
    )
    
    parser.add_argument(
        "--input_path", 
        type=str, 
        default=None,
        help="Path to input data for inference (default: processed test data)"
    )
    
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=None,
        help="Path to model configuration (default: auto-detect based on model type)"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save inference results (default: model's results/predictions directory)"
    )
    
    parser.add_argument(
        "--include_actual", 
        action="store_true",
        default=True,
        help="Include actual values in output if available (default: True)"
    )
    
    return parser.parse_args()

def predict_with_xgboost(df, model_path, config_path, include_actual=False):
    """
    Generate predictions using XGBoost model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data for inference
    model_path : str
        Path to trained model
    config_path : str
        Path to model configuration
    include_actual : bool
        Whether to include actual values in output if available
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions
    """
    from IncomeEstimation.baseline.xgboost.model.xgboost_model import XGBoostModel
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please train the model first by running: python -m IncomeEstimation.src.train.train --model xgboost")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading XGBoost model from: {model_path}")
    
    # Initialize model with config and load trained model
    model = XGBoostModel(config_path)
    model.load_model(model_path)
    
    # Extract target column if available
    target_col = 'target'
    y_actual = None
    
    if target_col in df.columns and include_actual:
        y_actual = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
    
    # Generate predictions
    logger.info(f"Generating predictions for {len(X)} samples")
    predictions = model.predict(X)
    
    # Create results DataFrame
    results = pd.DataFrame({'predicted_income': predictions})
    
    # Add actual values if available
    if y_actual is not None:
        results['actual_income'] = y_actual
        results['error'] = results['predicted_income'] - results['actual_income']
        results['error_pct'] = (results['error'] / results['actual_income']) * 100
    
    # Add dynamic threshold calculations if model evaluator is available
    if model.evaluator:
        absolute = model.evaluator.dynamic_threshold_absolute
        percentage = model.evaluator.dynamic_threshold_percentage / 100
        
        if y_actual is not None:
            results['dynamic_threshold'] = y_actual.apply(lambda x: max(absolute, x * percentage))
            results['exceeds_threshold'] = (results['error'] > results['dynamic_threshold']).astype(int)
        
        logger.info(f"Applied dynamic threshold: max({absolute}, {percentage * 100}% of actual)")
    
    return results

def predict_with_segment_aware(df, model_path, config_path, include_actual=False):
    """
    Generate predictions using segment-aware model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data for inference
    model_path : str
        Path to trained segment-aware model
    config_path : str
        Path to model configuration
    include_actual : bool, default=False
        Whether to include actual target values in results (if available)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions
    """
    from IncomeEstimation.prehoc.segment_aware.model.segment_aware_model import SegmentAwareHuberThresholdModel
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    logger.info(f"Loading Segment-Aware model from: {model_path}")
    model = SegmentAwareHuberThresholdModel.load(model_path)
    
    # Make predictions
    logger.info("Generating predictions...")
    
    # Create a copy of the dataframe without the target column if present
    features_df = df.copy()
    if 'target' in features_df.columns:
        y_true = features_df['target'].copy()
        features_df = features_df.drop(columns=['target'])
    else:
        y_true = None
    
    y_pred = model.predict(features_df)
    
    # Create results dataframe
    results_df = pd.DataFrame()
    
    # Add predictions
    results_df['segment_aware_prediction'] = y_pred
    
    # Add actual values and evaluation metrics if available
    if include_actual and y_true is not None:
        results_df['actual'] = y_true
        results_df['error'] = y_pred - y_true
        results_df['error_pct'] = (y_pred - y_true) / (y_true + 1e-6) * 100
        
        # Add threshold exceedance flag
        model_config = model.config.get('model', {})
        threshold_config = model_config.get('dynamic_threshold', {})
        absolute_threshold = threshold_config.get('absolute', 200)
        percentage_threshold = threshold_config.get('percentage', 20)
        
        # Calculate dynamic thresholds
        dynamic_thresholds = np.maximum(
            absolute_threshold,
            y_true * (percentage_threshold / 100)
        )
        
        # Check if predictions exceed threshold
        exceeds_threshold = (y_pred - y_true) > dynamic_thresholds
        results_df['exceeds_threshold'] = exceeds_threshold
        results_df['dynamic_threshold'] = dynamic_thresholds
        
        # Add income range categorization
        results_df['income_range'] = pd.cut(
            y_true,
            bins=[0, 1500, 2500, float('inf')],
            labels=["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
        )
        
        # Calculate overall metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        exceedance_rate = exceeds_threshold.mean()
        
        logger.info(f"Model performance:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Threshold exceedance rate: {exceedance_rate:.2%}")
        
        # Calculate additional metrics for consolidated output
        mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6))) * 100
        within_10pct = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6)) <= 0.1) * 100
        within_20pct = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6)) <= 0.2) * 100
        within_30pct = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6)) <= 0.3) * 100
        overestimation_pct = np.mean(y_pred > y_true) * 100
        overestimation_20plus_pct = np.mean((y_pred - y_true) / (y_true + 1e-6) > 0.2) * 100
        avg_dynamic_threshold = dynamic_thresholds.mean()
        
        # Create overall metrics dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'within_10pct': within_10pct,
            'within_20pct': within_20pct,
            'within_30pct': within_30pct,
            'overestimation_pct': overestimation_pct,
            'overestimation_20plus_pct': overestimation_20plus_pct,
            'exceeds_threshold_pct': exceedance_rate * 100,
            'avg_dynamic_threshold': avg_dynamic_threshold
        }
        
        # Calculate segment-specific metrics
        segment_metrics = []
        for segment, group in results_df.groupby('income_range'):
            if len(group) > 0:
                y_true_seg = group['actual']
                y_pred_seg = group['segment_aware_prediction']
                
                seg_rmse = np.sqrt(mean_squared_error(y_true_seg, y_pred_seg))
                seg_mae = mean_absolute_error(y_true_seg, y_pred_seg)
                seg_r2 = r2_score(y_true_seg, y_pred_seg)
                seg_mape = np.mean(np.abs((y_pred_seg - y_true_seg) / (y_true_seg + 1e-6))) * 100
                seg_within_10pct = np.mean(np.abs((y_pred_seg - y_true_seg) / (y_true_seg + 1e-6)) <= 0.1) * 100
                seg_within_20pct = np.mean(np.abs((y_pred_seg - y_true_seg) / (y_true_seg + 1e-6)) <= 0.2) * 100
                seg_within_30pct = np.mean(np.abs((y_pred_seg - y_true_seg) / (y_true_seg + 1e-6)) <= 0.3) * 100
                seg_overestimation_pct = np.mean(y_pred_seg > y_true_seg) * 100
                seg_overestimation_20plus_pct = np.mean((y_pred_seg - y_true_seg) / (y_true_seg + 1e-6) > 0.2) * 100
                seg_exceeds_pct = group['exceeds_threshold'].mean() * 100
                seg_avg_threshold = group['dynamic_threshold'].mean()
                
                segment_metrics.append({
                    'segment': str(segment),
                    'count': len(group),
                    'r2': seg_r2,
                    'rmse': seg_rmse,
                    'mae': seg_mae,
                    'mape': seg_mape,
                    'within_10pct': seg_within_10pct,
                    'within_20pct': seg_within_20pct,
                    'within_30pct': seg_within_30pct,
                    'overestimation_pct': seg_overestimation_pct,
                    'overestimation_20plus_pct': seg_overestimation_20plus_pct,
                    'exceeds_threshold_pct': seg_exceeds_pct,
                    'avg_dynamic_threshold': seg_avg_threshold
                })
                
                segment_exceeds = group['exceeds_threshold'].mean()
                logger.info(f"  {segment} exceedance rate: {segment_exceeds:.2%}")
        
        # Add overall metrics as a segment for consistency
        all_data_metric = metrics.copy()
        all_data_metric.update({
            'segment': 'All Data',
            'count': len(results_df)
        })
        segment_metrics.append(all_data_metric)
        
        # Convert to DataFrame for easier handling
        segment_metrics_df = pd.DataFrame(segment_metrics)
        
        # Create the consolidated metrics dictionary with clear indication it's test data
        consolidated_metrics = {
            'data_type': 'test',
            'overall': metrics,
            'segments': segment_metrics_df.to_dict(orient='records')
        }
        
        # Save consolidated metrics to the metrics directory
        from IncomeEstimation.src.utils.paths import get_project_root
        import os
        import json
        
        project_root = get_project_root()
        metrics_dir = project_root / "prehoc" / "segment_aware" / "results" / "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = metrics_dir / "segment_aware_metrics.json"
        
        logger.info(f"Saving consolidated metrics to: {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(consolidated_metrics, f, indent=4)
        
        # Remove existing CSV metrics files if they exist
        old_files = [
            metrics_dir / "segment_aware_segment_metrics.csv",
            project_root / "prehoc" / "segment_aware" / "results" / "models" / "segment_aware_metrics.json",
            project_root / "prehoc" / "segment_aware" / "results" / "models" / "segment_aware_test_segment_metrics.csv",
            project_root / "prehoc" / "segment_aware" / "results" / "models" / "segment_aware_train_segment_metrics.csv"
        ]
        
        for old_file in old_files:
            if os.path.exists(old_file):
                logger.info(f"Removing redundant metrics file: {old_file}")
                os.remove(old_file)
        
        logger.info(f"Evaluation metrics:")
        logger.info(f"• RMSE: {metrics['rmse']:.4f}")
        logger.info(f"• MAE: {metrics['mae']:.4f}")
        logger.info(f"• R²: {metrics['r2']:.4f}")
        logger.info(f"• MAPE: {metrics['mape']:.2f}%")
        logger.info(f"• Threshold exceedance: {metrics['exceeds_threshold_pct']:.2f}%")
    
    return results_df

def predict_with_huber_threshold(df, model_path, config_path, include_actual=False):
    """
    Generate predictions using Huber threshold model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data for inference
    model_path : str
        Path to trained Huber threshold model
    config_path : str
        Path to model configuration
    include_actual : bool, default=False
        Whether to include actual target values in results (if available)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions
    """
    from IncomeEstimation.prehoc.huber_threshold.model.huber_threshold_model import HuberThresholdModel
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    logger.info(f"Loading Huber threshold model from: {model_path}")
    model = HuberThresholdModel.load(model_path)
    
    # Make predictions
    logger.info("Generating predictions...")
    
    # Create a copy of the dataframe without the target column if present
    features_df = df.copy()
    if 'target' in features_df.columns:
        y_true = features_df['target'].copy()
        features_df = features_df.drop(columns=['target'])
    else:
        y_true = None
    
    y_pred = model.predict(features_df)
    
    # Create results dataframe
    results_df = pd.DataFrame()
    
    # Add predictions
    results_df['huber_threshold_prediction'] = y_pred
    
    # Add actual values and evaluation metrics if available
    if include_actual and y_true is not None:
        results_df['actual'] = y_true
        results_df['error'] = y_pred - y_true
        results_df['error_pct'] = (y_pred - y_true) / (y_true + 1e-6) * 100
        
        # Add threshold exceedance flag
        absolute_threshold = model.config.get('model', {}).get('dynamic_threshold', {}).get('absolute', 200)
        percentage_threshold = model.config.get('model', {}).get('dynamic_threshold', {}).get('percentage', 20)
        
        # Calculate dynamic threshold for each true value
        dynamic_thresholds = np.maximum(
            absolute_threshold,
            y_true * (percentage_threshold / 100)
        )
        
        # Flag threshold exceedances
        results_df['exceeds_threshold'] = (results_df['error'] > dynamic_thresholds).astype(int)
        
        # Add income range segmentation
        results_df['income_range'] = pd.cut(
            results_df['actual'],
            bins=[0, 1500, 2500, float('inf')],
            labels=["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
        )
        
        # Calculate aggregate metrics directly to ensure consistent results
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6))) * 100,  # Using same formula as ModelEvaluator
            'within_10pct': (np.abs(results_df['error_pct']) <= 10).mean() * 100,
            'within_20pct': (np.abs(results_df['error_pct']) <= 20).mean() * 100,
            'within_30pct': (np.abs(results_df['error_pct']) <= 30).mean() * 100,
            'overestimation_pct': (results_df['error'] > 0).mean() * 100,
            'overestimation_20plus_pct': ((results_df['error'] > 0) & (results_df['error_pct'] > 20)).mean() * 100,
            'exceeds_threshold_pct': results_df['exceeds_threshold'].mean() * 100,
            'avg_dynamic_threshold': dynamic_thresholds.mean()
        }
        
        # Calculate segment metrics
        segment_metrics = []
        for segment, group in results_df.groupby('income_range'):
            segment_metric = {
                'segment': segment,
                'count': len(group),
                'r2': r2_score(group['actual'], group['huber_threshold_prediction']),
                'rmse': np.sqrt(mean_squared_error(group['actual'], group['huber_threshold_prediction'])),
                'mae': mean_absolute_error(group['actual'], group['huber_threshold_prediction']),
                'mape': np.mean(np.abs((group['huber_threshold_prediction'] - group['actual']) / (group['actual'] + 1e-6))) * 100,
                'within_10pct': (np.abs(group['error_pct']) <= 10).mean() * 100,
                'within_20pct': (np.abs(group['error_pct']) <= 20).mean() * 100,
                'within_30pct': (np.abs(group['error_pct']) <= 30).mean() * 100,
                'overestimation_pct': (group['error'] > 0).mean() * 100,
                'overestimation_20plus_pct': ((group['error'] > 0) & (group['error_pct'] > 20)).mean() * 100,
                'exceeds_threshold_pct': group['exceeds_threshold'].mean() * 100,
                'avg_dynamic_threshold': dynamic_thresholds[group.index].mean()
            }
            segment_metrics.append(segment_metric)
        
        # Add overall metrics as a segment
        all_data_metric = metrics.copy()
        all_data_metric['segment'] = 'All Data'
        all_data_metric['count'] = len(results_df)
        segment_metrics.append(all_data_metric)
        
        # Convert to DataFrame for easier handling
        segment_metrics_df = pd.DataFrame(segment_metrics)
        
        # Create the consolidated metrics dictionary with clear indication it's test data
        consolidated_metrics = {
            'data_type': 'test',
            'overall': metrics,
            'segments': segment_metrics_df.to_dict(orient='records')
        }
        
        # Save consolidated metrics to the metrics directory
        project_root = get_project_root()
        metrics_dir = project_root / "prehoc" / "huber_threshold" / "results" / "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = metrics_dir / "huber_threshold_metrics.json"
        
        logger.info(f"Saving consolidated metrics to: {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(consolidated_metrics, f, indent=4)
        
        # Remove existing CSV metrics files if they exist
        old_files = [
            metrics_dir / "huber_threshold_segment_metrics.csv",
            project_root / "prehoc" / "huber_threshold" / "results" / "models" / "huber_threshold_metrics.json",
            project_root / "prehoc" / "huber_threshold" / "results" / "models" / "huber_threshold_test_segment_metrics.csv",
            project_root / "prehoc" / "huber_threshold" / "results" / "models" / "huber_threshold_train_segment_metrics.csv"
        ]
        
        for old_file in old_files:
            if os.path.exists(old_file):
                logger.info(f"Removing redundant metrics file: {old_file}")
                os.remove(old_file)
        
        logger.info(f"Evaluation metrics:")
        logger.info(f"• RMSE: {metrics['rmse']:.4f}")
        logger.info(f"• MAE: {metrics['mae']:.4f}")
        logger.info(f"• R²: {metrics['r2']:.4f}")
        logger.info(f"• MAPE: {metrics['mape']:.2f}%")
        logger.info(f"• Threshold exceedance: {metrics['exceeds_threshold_pct']:.2f}%")
    
    return results_df

def predict_with_quantile(df, model_path, config_path, include_actual=False):
    """
    Generate predictions using quantile regression model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data for inference
    model_path : str
        Path to trained quantile model
    config_path : str
        Path to model configuration
    include_actual : bool, default=False
        Whether to include actual target values in results (if available)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions and evaluation metrics (if available)
    """
    from IncomeEstimation.posthoc.quantile.model.quantile_model import QuantileRegressionModel, load_config
    from IncomeEstimation.baseline.xgboost.model.xgboost_model import XGBoostModel
    import joblib
    from pathlib import Path
    
    logger.info(f"Loading quantile model from: {model_path}")
    
    # Load configuration
    config = load_config(config_path)
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
    
    # Load quantile model
    logger.info(f"Loading calibration parameters from: {model_path}")
    model = QuantileRegressionModel.load(model_path, xgboost_model)
    
    # Extract target column if available
    target_col = 'target'
    y_actual = None
    
    if target_col in df.columns and include_actual:
        y_actual = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
    
    # Generate predictions with base model
    logger.info(f"Generating baseline predictions for {len(X)} samples")
    base_predictions = xgboost_model.predict(X)
    
    # Generate calibrated predictions
    logger.info(f"Applying quantile calibration (q={model.quantile})")
    calibrated_predictions = model.predict(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'base_predicted_income': base_predictions,
        'calibrated_predicted_income': calibrated_predictions
    })
    
    # Add actual values if available
    if y_actual is not None:
        results['actual_income'] = y_actual
        
        # Calculate errors for base model
        results['base_error'] = results['base_predicted_income'] - results['actual_income']
        results['base_error_pct'] = (results['base_error'] / results['actual_income']) * 100
        
        # Calculate errors for calibrated model
        results['calibrated_error'] = results['calibrated_predicted_income'] - results['actual_income']
        results['calibrated_error_pct'] = (results['calibrated_error'] / results['actual_income']) * 100
        
        # Add dynamic threshold calculations
        results['dynamic_threshold'] = y_actual.apply(
            lambda x: max(absolute_threshold, x * (percentage_threshold / 100))
        )
        
        # Flag threshold exceedances for both models
        results['base_exceeds_threshold'] = (
            results['base_error'] > results['dynamic_threshold']
        ).astype(int)
        
        results['calibrated_exceeds_threshold'] = (
            results['calibrated_error'] > results['dynamic_threshold']
        ).astype(int)
        
        # Add income range segmentation
        results['income_range'] = pd.cut(
            results['actual_income'],
            bins=[0, 1500, 2500, float('inf')],
            labels=["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
        )
        
        # Print some summary statistics
        base_exceed_rate = results['base_exceeds_threshold'].mean() * 100
        calib_exceed_rate = results['calibrated_exceeds_threshold'].mean() * 100
        
        logger.info(f"Applied dynamic threshold: max({absolute_threshold}, {percentage_threshold}% of actual)")
        logger.info(f"Base model threshold exceedance: {base_exceed_rate:.2f}%")
        logger.info(f"Calibrated model threshold exceedance: {calib_exceed_rate:.2f}%")
        logger.info(f"Improvement: {base_exceed_rate - calib_exceed_rate:.2f}%")
    
    return results

def main():
    """Main inference function"""
    args = parse_args()
    
    # Setup logging
    setup_logger('inference')
    logger.info(f"Starting inference with model: {args.model}")
    
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
            # Fallback for backward compatibility
            if not os.path.exists(model_path):
                alt_paths = [
                    project_root / "baseline" / "xgboost" / "results" / "models" / "xgboost_model.bin",
                    project_root / "baseline" / "xgboost" / "results" / "models" / "xgboost_model.json"
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        break
        elif args.model == "segment_aware":
            model_path = project_root / "prehoc" / "segment_aware" / "results" / "models" / "segment_aware_model.joblib"
        elif args.model == "huber_threshold":
            model_path = project_root / "prehoc" / "huber_threshold" / "results" / "models" / "huber_threshold_model.joblib"
        elif args.model == "quantile":
            model_path = project_root / "posthoc" / "quantile" / "results" / "models" / "quantile_model.joblib"
            
    # Convert Path to string for compatibility with model loading
    model_path_str = str(model_path)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        logger.error("Please run training first with: python -m IncomeEstimation.src.train.train --model xgboost")
        return
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        if args.model == "xgboost":
            output_dir = project_root / "baseline" / "xgboost" / "results" / "predictions"
        elif args.model == "segment_aware":
            output_dir = project_root / "prehoc" / "segment_aware" / "results" / "predictions"
        elif args.model == "huber_threshold":
            output_dir = project_root / "prehoc" / "huber_threshold" / "results" / "predictions"
        elif args.model == "quantile":
            output_dir = project_root / "posthoc" / "quantile" / "results" / "predictions"
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / f"{args.model}_predictions.csv"
    
    # Use the test data by default if input path is not specified
    if args.input_path is None:
        from IncomeEstimation.src.utils.paths import get_processed_data_dir
        input_path = get_processed_data_dir() / "test.csv"
        logger.info(f"Using default test data: {input_path}")
    else:
        input_path = Path(args.input_path)
        logger.info(f"Loading input data from: {input_path}")
    
    try:
        # Determine file type and load accordingly
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    except Exception as e:
        logger.error(f"Error loading input data: {str(e)}")
        return
    
    try:
        # Detect if this is a pre-hoc model
        prehoc_models = ["huber_threshold", "segment_aware"]
        is_prehoc = args.model in prehoc_models
        
        # Generate predictions based on model type
        if args.model == "xgboost":
            results = predict_with_xgboost(
                df, str(model_path), str(config_path), include_actual=args.include_actual
            )
        elif args.model == "segment_aware":
            results = predict_with_segment_aware(
                df, str(model_path), str(config_path), include_actual=args.include_actual
            )
        elif args.model == "huber_threshold":
            results = predict_with_huber_threshold(
                df, str(model_path), str(config_path), include_actual=args.include_actual
            )
        elif args.model == "quantile":
            results = predict_with_quantile(
                df, str(model_path), str(config_path), include_actual=args.include_actual
            )
        
        # Save results
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")
        
        # Print summary statistics if available
        if 'error' in results.columns:
            logger.info("Prediction summary statistics:")
            logger.info(f"  Mean absolute error: {results['error'].abs().mean():.2f}")
            logger.info(f"  Mean error: {results['error'].mean():.2f}")
            logger.info(f"  Mean absolute percentage error: {results['error_pct'].abs().mean():.2f}%")
            
            if 'exceeds_threshold' in results.columns:
                exceed_rate = results['exceeds_threshold'].mean() * 100
                logger.info(f"  Threshold exceedance rate: {exceed_rate:.2f}%")
        
        logger.info("Inference completed successfully")
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return

if __name__ == "__main__":
    main()
