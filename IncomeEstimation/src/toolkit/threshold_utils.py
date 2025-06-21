"""
Threshold utilities for income estimation models.
"""
import numpy as np

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
