"""
Visualization utilities for comparing model performance and feature analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from IncomeEstimation.src.toolkit.theme import Theme


def plot_model_comparison(
    y_true,
    standard_pred,
    quantile_pred,
    prehoc_pred,
    prehoc_name="Huber Threshold",
    color_theme='purple',
    output_path=None,
    figsize=(15, 10)
):
    """
    Generate a comprehensive comparison visualization between standard model, 
    post-hoc quantile model, and pre-hoc model predictions.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    standard_pred : array-like
        Standard model predictions
    quantile_pred : array-like
        Post-hoc quantile model predictions
    prehoc_pred : array-like
        Pre-hoc model predictions
    prehoc_name : str, default="Huber Threshold"
        Name of the pre-hoc model for labeling
    color_theme : str, default='purple'
        Color theme for the pre-hoc model ('blue', 'purple', 'green', 'orange', 'teal')
    output_path : str, optional
        Path to save the visualization. If None, the plot is displayed instead.
    figsize : tuple, default=(15, 10)
        Figure size (width, height) in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    # Get color for pre-hoc model based on theme
    if color_theme in Theme.THEMES:
        prehoc_color = Theme.THEMES[color_theme]['primary']
    else:
        prehoc_color = '#9b59b6'  # Default purple
    
    # Define consistent colors for models
    colors = {
        'standard': '#1f77b4',  # Blue
        'quantile': '#2ecc71',   # Green
        'prehoc': prehoc_color
    }
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # --- 1. Error Distribution ---
    plt.subplot(2, 2, 1)
    plt.hist(standard_pred - y_true, bins=50, alpha=0.5, label='Standard Model', color=colors['standard'])
    plt.hist(quantile_pred - y_true, bins=50, alpha=0.5, label='Post-hoc Quantile', color=colors['quantile'])
    plt.hist(prehoc_pred - y_true, bins=50, alpha=0.5, label=f'Pre-hoc ({prehoc_name})', color=colors['prehoc'])
    plt.axvline(0, color='k', linestyle='--')
    plt.title("Error Distribution Comparison")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.legend()
    
    # --- 2. Percentage Error Distribution ---
    plt.subplot(2, 2, 2)
    pct_std = (standard_pred - y_true) / (y_true + 1e-6) * 100
    pct_q = (quantile_pred - y_true) / (y_true + 1e-6) * 100
    pct_ph = (prehoc_pred - y_true) / (y_true + 1e-6) * 100
    plt.hist(pct_std, bins=50, range=(-50, 100), alpha=0.5, label='Standard Model', color=colors['standard'])
    plt.hist(pct_q, bins=50, range=(-50, 100), alpha=0.5, label='Post-hoc Quantile', color=colors['quantile'])
    plt.hist(pct_ph, bins=50, range=(-50, 100), alpha=0.5, label=f'Pre-hoc ({prehoc_name})', color=colors['prehoc'])
    plt.axvline(0, color='k', linestyle='--')
    plt.axvline(20, color='r', linestyle='--', label='20% Threshold')
    plt.title("Percentage Error Distribution")
    plt.xlabel("Percentage Error")
    plt.ylabel("Frequency")
    plt.legend()
    
    # --- 3. Dynamic Threshold Exceedance by Income Range ---
    plt.subplot(2, 2, 3)
    
    # Create income ranges
    income_ranges = pd.cut(
        y_true,
        bins=[0, 1500, 2500, float('inf')],
        labels=["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
    )
    
    # Calculate dynamic threshold for each true value
    absolute_threshold = 200
    percentage_threshold = 20
    dynamic_threshold = np.maximum(
        absolute_threshold,
        y_true * (percentage_threshold / 100)
    )
    
    # Calculate exceedance for each model
    exceed_std = (standard_pred - y_true) > dynamic_threshold
    exceed_q = (quantile_pred - y_true) > dynamic_threshold
    exceed_ph = (prehoc_pred - y_true) > dynamic_threshold
    
    # Calculate exceedance rates by range
    ranges = ["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
    std_vals = []
    q_vals = []
    ph_vals = []
    
    for r in ranges:
        mask = income_ranges == r
        if np.sum(mask) > 0:  # Ensure there are samples in this range
            std_vals.append(100 * np.mean(exceed_std[mask]))
            q_vals.append(100 * np.mean(exceed_q[mask]))
            ph_vals.append(100 * np.mean(exceed_ph[mask]))
        else:
            std_vals.append(0)
            q_vals.append(0)
            ph_vals.append(0)
    
    # Plot bar chart
    x_pos = np.arange(len(ranges))
    width = 0.25
    plt.bar(x_pos - width, std_vals, width, label='Standard Model', color=colors['standard'])
    plt.bar(x_pos, q_vals, width, label='Post-hoc Quantile', color=colors['quantile'])
    plt.bar(x_pos + width, ph_vals, width, label=f'Pre-hoc ({prehoc_name})', color=colors['prehoc'])
    plt.xticks(x_pos, ranges)
    plt.title("Dynamic Threshold Exceedance by Income Range")
    plt.xlabel("Income Range")
    plt.ylabel("Exceedance (%)")
    plt.legend()
    
    # --- 4. Prediction Comparison ---
    plt.subplot(2, 2, 4)
    plt.scatter(y_true, standard_pred, alpha=0.3, color=colors['standard'], label='Standard Model')
    plt.scatter(y_true, quantile_pred, alpha=0.3, color=colors['quantile'], label='Post-hoc Quantile')
    plt.scatter(y_true, prehoc_pred, alpha=0.3, color=colors['prehoc'], label=f'Pre-hoc ({prehoc_name})')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.title("Prediction Comparison")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    
    plt.tight_layout()
    plt.suptitle(f"Comparison of Models with Pre-hoc {prehoc_name}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def plot_feature_importance(model, feature_names, top_n=10, color_theme='purple', output_path=None, figsize=(10, 6)):
    """
    Plot feature importance for a model with a specified color theme.
    
    Parameters
    ----------
    model : object
        Trained model with feature_importances_ attribute or get_score() method
    feature_names : list
        List of feature names
    top_n : int, default=10
        Number of top features to show
    color_theme : str, default='purple'
        Color theme ('blue', 'purple', 'green', 'orange', 'teal')
    output_path : str, optional
        Path to save the plot. If None, the plot is displayed instead.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with feature importance values
    """
    # Get color for bars based on theme
    if color_theme in Theme.THEMES:
        bar_color = Theme.THEMES[color_theme]['primary']
    else:
        bar_color = '#9b59b6'  # Default purple
    
    # Extract feature importance
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_score'):
            importance_dict = model.get_score(importance_type='gain')
            importances = [importance_dict.get(name, 0) for name in feature_names]
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score'):
            importance_dict = model.get_booster().get_score(importance_type='gain')
            importances = [importance_dict.get(name, 0) for name in feature_names]
        else:
            raise AttributeError("Model has no feature_importances_ attribute or get_score method")
    except Exception as e:
        raise ValueError(f"Could not extract feature importances: {str(e)}")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=figsize)
    top_features = importance_df.head(top_n).sort_values('importance')
    plt.barh(top_features['feature'], top_features['importance'], color=bar_color)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Features by Importance')
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return importance_df


def plot_threshold_exceedance_by_model(
    y_true,
    predictions_dict,
    absolute_threshold=200, 
    percentage_threshold=20, 
    colors=None,
    output_path=None,
    figsize=(10, 6)
):
    """
    Plot threshold exceedance comparison across multiple models by income range.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values
    absolute_threshold : float, default=200
        Absolute threshold value
    percentage_threshold : float, default=20
        Percentage threshold value
    colors : dict, optional
        Dictionary mapping model names to colors
    output_path : str, optional
        Path to save the visualization. If None, the plot is displayed instead.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
        
    Returns
    -------
    dict
        Dictionary with exceedance rates for each model by income range
    """
    # Create income ranges
    income_ranges = pd.cut(
        y_true,
        bins=[0, 1500, 2500, float('inf')],
        labels=["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
    )
    
    # Calculate dynamic threshold for each true value
    dynamic_threshold = np.maximum(
        absolute_threshold,
        y_true * (percentage_threshold / 100)
    )
    
    # Default colors if not provided
    if colors is None:
        colors = {
            'Standard Model': '#1f77b4',  # Blue
            'Post-hoc Quantile': '#2ecc71',  # Green
            'Pre-hoc (Huber Threshold)': '#9b59b6'  # Purple
        }
    
    # Calculate exceedance rates by range for each model
    ranges = ["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
    model_exceedance = {}
    
    for model_name, predictions in predictions_dict.items():
        # Calculate exceedance
        exceed = (predictions - y_true) > dynamic_threshold
        
        # Calculate rates by range
        rates = []
        for r in ranges:
            mask = income_ranges == r
            if np.sum(mask) > 0:  # Ensure there are samples in this range
                rates.append(100 * np.mean(exceed[mask]))
            else:
                rates.append(0)
        
        model_exceedance[model_name] = rates
    
    # Plot bar chart
    plt.figure(figsize=figsize)
    x_pos = np.arange(len(ranges))
    width = 0.8 / len(predictions_dict)
    
    # Position bars
    bar_positions = {}
    for i, model_name in enumerate(predictions_dict.keys()):
        pos = x_pos + (i - len(predictions_dict)/2 + 0.5) * width
        bar_positions[model_name] = pos
        plt.bar(
            pos,
            model_exceedance[model_name],
            width,
            label=model_name,
            color=colors.get(model_name, f'C{i}')  # Use provided color or default
        )
    
    plt.xticks(x_pos, ranges)
    plt.title(f"Dynamic Threshold Exceedance by Income Range (max({absolute_threshold}, {percentage_threshold}%))")
    plt.xlabel("Income Range")
    plt.ylabel("Exceedance (%)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return model_exceedance


def run_prehoc_evaluation(
    y_true,
    standard_pred,
    quantile_pred,
    prehoc_pred,
    model=None,
    X=None,
    prehoc_name="huber_threshold",
    color_theme='purple',
    output_dir=None,
    run_feature_analysis=False,
    figsize=(15, 10)
):
    """
    Run a comprehensive evaluation of pre-hoc model comparison with standard and post-hoc models,
    including feature analysis if requested.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    standard_pred : array-like
        Standard model predictions
    quantile_pred : array-like
        Post-hoc quantile model predictions
    prehoc_pred : array-like
        Pre-hoc model predictions
    model : object, optional
        Trained pre-hoc model for feature analysis
    X : DataFrame, optional
        Features dataset for feature analysis
    prehoc_name : str, default="huber_threshold"
        Name of the pre-hoc model for labeling
    color_theme : str, default='purple'
        Color theme for the pre-hoc model ('blue', 'purple', 'green', 'orange', 'teal')
    output_dir : str, optional
        Directory to save visualizations. If None, plots are displayed instead.
    run_feature_analysis : bool, default=False
        Whether to run feature importance analysis
    figsize : tuple, default=(15, 10)
        Figure size (width, height) in inches
        
    Returns
    -------
    dict
        Dictionary containing all generated plots and metrics
    """
    # Get color for pre-hoc model based on theme
    if color_theme in Theme.THEMES:
        prehoc_color = Theme.THEMES[color_theme]['primary']
    else:
        prehoc_color = '#9b59b6'  # Default purple
    
    # Define consistent colors for models
    colors = {
        'Standard Model': '#1f77b4',  # Blue
        'Post-hoc Quantile': '#2ecc71',   # Green
        f'Pre-hoc ({prehoc_name})': prehoc_color
    }
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create model directory
    model_dir = os.path.join(output_dir, "model_comparison") if output_dir else None
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # 1. Generate comprehensive model comparison plot
    model_comparison_plot = plot_model_comparison(
        y_true,
        standard_pred,
        quantile_pred,
        prehoc_pred,
        prehoc_name=prehoc_name,
        color_theme=color_theme,
        output_path=os.path.join(model_dir, "model_comparison.png") if model_dir else None,
        figsize=figsize
    )
    
    # 2. Generate threshold exceedance by income range plot
    threshold_exceedance_plot = plot_threshold_exceedance_by_model(
        y_true,
        {
            'Standard Model': standard_pred,
            'Post-hoc Quantile': quantile_pred,
            f'Pre-hoc ({prehoc_name})': prehoc_pred
        },
        colors=colors,
        output_path=os.path.join(model_dir, "threshold_exceedance.png") if model_dir else None,
        figsize=(10, 6)
    )
    
    results = {
        'model_comparison_plot': model_comparison_plot,
        'threshold_exceedance_plot': threshold_exceedance_plot
    }
    
    # 3. Feature analysis if requested and model + data provided
    if run_feature_analysis and model is not None and X is not None:
        feature_dir = os.path.join(output_dir, "feature_analysis") if output_dir else None
        if feature_dir:
            os.makedirs(feature_dir, exist_ok=True)
        
        feature_analysis_results = run_feature_analysis(
            model,
            X,
            color_theme=color_theme,
            output_dir=feature_dir,
            top_n_features=10,
            figsize=(10, 6)
        )
        
        results['feature_analysis'] = feature_analysis_results
    
    return results


def plot_shap_summary(model, X, max_display=10, color_theme='purple', output_path=None, figsize=(10, 8)):
    """
    Generate SHAP summary plot showing feature impacts.
    
    Parameters
    ----------
    model : object
        Trained model
    X : DataFrame
        Features to calculate SHAP values for
    max_display : int, default=10
        Maximum number of features to display
    color_theme : str, default='purple'
        Color theme ('blue', 'purple', 'green', 'orange', 'teal')
    output_path : str, optional
        Path to save the plot. If None, the plot is displayed instead.
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
        
    Returns
    -------
    tuple
        (shap_values, shap_importance_df)
    """
    # Sample data if too large
    if len(X) > 1000:
        X_sample = X.sample(1000, random_state=42)
    else:
        X_sample = X
    
    # Calculate SHAP values
    try:
        import shap
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
    except Exception as e:
        raise ValueError(f"Could not calculate SHAP values: {str(e)}")
    
    # Calculate mean absolute SHAP value for each feature
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    
    # Create DataFrame with SHAP importance
    shap_importance_df = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)
    
    # Set color theme
    theme_colors = Theme.THEMES.get(color_theme, Theme.THEMES['purple'])
    
    # Plot SHAP summary with customized colors
    plt.figure(figsize=figsize)
    
    # Create custom colormap for SHAP summary
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', 
        [theme_colors['secondary'], theme_colors['primary']], 
        N=256
    )
    
    # Generate SHAP summary plot
    shap.summary_plot(
        shap_values, 
        X_sample,
        plot_type="dot",
        max_display=max_display,
        plot_size=figsize,
        color=cmap,
        show=False
    )
    
    plt.title("SHAP Feature Impact Summary\n(red = high feature value, blue = low feature value)", 
              fontsize=14, pad=20)
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return shap_values, shap_importance_df


def plot_pdp(model, X, features, color_theme='purple', output_path=None, figsize=(12, 8)):
    """
    Generate Partial Dependence Plots (PDP) for specified features.
    
    Parameters
    ----------
    model : object
        Trained model (XGBoost Booster or sklearn-compatible model)
    X : DataFrame
        Features dataset
    features : list
        List of feature names to create PDPs for
    color_theme : str, default='purple'
        Color theme ('blue', 'purple', 'green', 'orange', 'teal')
    output_path : str, optional
        Path to save the plot. If None, the plot is displayed instead.
    figsize : tuple, default=(12, 8)
        Figure size (width, height) in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    # Set color based on theme
    if color_theme in Theme.THEMES:
        line_color = Theme.THEMES[color_theme]['primary']
        fill_color = line_color
        alpha = 0.2
    else:
        line_color = '#9b59b6'  # Default purple
        fill_color = line_color
        alpha = 0.2
    
    # Check if features exist in the dataset
    valid_features = [feat for feat in features if feat in X.columns]
    if not valid_features:
        raise ValueError(f"None of the provided features {features} found in the dataset")
    
    # Create a custom PDP for XGBoost Booster or similar models that don't implement sklearn API
    import numpy as np
    
    # Create figure with grid layout based on number of features
    n_features = len(valid_features)
    n_cols = min(3, n_features)  # Max 3 columns
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Make axes accessible as a 1D array regardless of grid layout
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = np.array(axes).flatten()
    elif n_cols == 1:
        axes = np.array(axes).flatten()
    else:
        axes = axes.flatten()
    
    # Custom implementation for PDP plots that works with XGBoost booster
    import pandas as pd
    
    # Define a predict function that works with XGBoost booster
    def predict_func(X_input):
        if hasattr(model, 'predict'):
            # Standard sklearn interface
            return model.predict(X_input)
        elif hasattr(model, '__call__'):
            # Callable model
            return model(X_input)
        elif str(type(model)).find('xgboost') >= 0:
            # XGBoost booster
            try:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X_input)
                return model.predict(dmatrix)
            except Exception:
                return None
        else:
            raise ValueError("Model type not supported for PDP generation")
            
    # Create PDP for each feature
    for i, feature in enumerate(valid_features):
        if i >= len(axes):
            break  # Safety check
            
        ax = axes[i]
        feature_values = X[feature].unique()
        feature_values.sort()
        
        # For better visualization, sample the unique values if there are too many
        if len(feature_values) > 20:
            # Get quantiles for better distribution
            feature_values = np.percentile(X[feature], np.linspace(0, 100, 20))
        
        pdp_values = []
        
        # Calculate average predictions across the feature grid
        for val in feature_values:
            X_temp = X.copy()
            X_temp[feature] = val
            pred = predict_func(X_temp)
            if pred is not None:
                pdp_values.append(np.mean(pred))
            else:
                pdp_values.append(np.nan)
        
        # Plot the PDP
        ax.plot(feature_values, pdp_values, color=line_color, linewidth=2)
        
        # Add a histogram of the feature distribution
        ax_twin = ax.twinx()
        histogram = ax_twin.hist(X[feature], bins=20, alpha=alpha, color=fill_color, density=True)
        ax_twin.set_yticks([])  # Hide y-axis for histogram
        
        # Set labels and title
        ax.set_xlabel(feature)
        ax.set_ylabel('Partial dependence')
        ax.set_title(f'PDP: {feature}')
        ax.grid(True, alpha=0.3)
        
        # Add feature importance if available
        if hasattr(model, 'get_score'):
            try:
                importances = model.get_score(importance_type='gain')
                if feature in importances:
                    imp_value = importances[feature]
                    ax.set_title(f"PDP: {feature} (Imp: {imp_value:.2f})")
            except:
                pass
    
    # Hide any unused axes
    for i in range(len(valid_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Feature Dependency Analysis - SHAP and Partial Dependence", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def run_feature_analysis(
    model, 
    X, 
    color_theme='purple', 
    output_dir=None, 
    top_n_features=10, 
    figsize=(10, 6)
):
    """
    Run a comprehensive feature analysis including importance, SHAP and PDP plots.
    
    Parameters
    ----------
    model : object
        Trained model
    X : DataFrame
        Features dataset
    color_theme : str, default='purple'
        Color theme ('blue', 'purple', 'green', 'orange', 'teal')
    output_dir : str, optional
        Directory to save visualizations. If None, plots are displayed instead.
    top_n_features : int, default=10
        Number of top features to analyze
    figsize : tuple, default=(10, 6)
        Base figure size (width, height) in inches
        
    Returns
    -------
    dict
        Dictionary containing feature importance DataFrame and plots
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # 1. Feature Importance
    importance_df = plot_feature_importance(
        model, 
        X.columns, 
        top_n=top_n_features, 
        color_theme=color_theme,
        output_path=os.path.join(output_dir, "feature_importance.png") if output_dir else None,
        figsize=figsize
    )
    
    # 2. SHAP Summary
    try:
        shap_values, shap_importance_df = plot_shap_summary(
            model, 
            X, 
            max_display=top_n_features, 
            color_theme=color_theme,
            output_path=os.path.join(output_dir, "shap_summary.png") if output_dir else None,
            figsize=(figsize[0], figsize[1] * 1.3)
        )
    except Exception as e:
        print(f"Warning: Could not generate SHAP summary plot: {str(e)}")
        shap_values, shap_importance_df = None, None
    
    # 3. PDP Plots for top features
    try:
        top_features = importance_df['feature'].head(min(5, top_n_features)).tolist()
        pdp_fig = plot_pdp(
            model, 
            X, 
            top_features, 
            color_theme=color_theme,
            output_path=os.path.join(output_dir, "pdp_plots.png") if output_dir else None,
            figsize=(figsize[0] * 1.2, figsize[1] * 1.5)
        )
    except Exception as e:
        print(f"Warning: Could not generate PDP plots: {str(e)}")
        pdp_fig = None
    
    return {
        'importance_df': importance_df,
        'shap_values': shap_values,
        'shap_importance_df': shap_importance_df,
        'pdp_fig': pdp_fig
    }


def create_prehoc_model_comparison(
    y_true,
    prehoc_pred,
    prehoc_name,
    color_theme='purple',
    output_path=None,
    figsize=(15, 10)
):
    """
    Create a comprehensive model comparison for a pre-hoc model by loading
    the standard and post-hoc models automatically.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    prehoc_pred : array-like
        Pre-hoc model predictions
    prehoc_name : str
        Name of the pre-hoc model (e.g., "huber_plus_threshold_loss")
    color_theme : str, default='purple'
        Color theme for the pre-hoc model
    output_path : str, optional
        Path to save the visualization
    figsize : tuple, default=(15, 10)
        Figure size (width, height) in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    from pathlib import Path
    from IncomeEstimation.src.utils.paths import get_project_root
    import os
    import logging
    
    logger = logging.getLogger('evaluation')
    
    # Get project root
    project_root = get_project_root()
    
    # Get data for all models
    try:
        # Load the baseline XGBoost model
        from IncomeEstimation.baseline.xgboost.model.xgboost_model import XGBoostModel
        xgboost_path = Path(project_root) / "baseline" / "xgboost" / "results" / "models" / "xgboost_model.joblib"
        
        if not os.path.exists(xgboost_path):
            logger.warning(f"XGBoost model not found at: {xgboost_path}")
            raise FileNotFoundError(f"XGBoost model not found at: {xgboost_path}")
            
        # Load XGBoost model
        logger.info(f"Loading XGBoost model from: {xgboost_path}")
        xgboost_model = XGBoostModel.load(xgboost_path)
        
        # Make standard predictions
        logger.info("Generating standard model predictions...")
        X_test = None
        # Extract features from the same test data as used for y_true
        if hasattr(y_true, 'index'):
            # Check if we have a pandas Series with an index
            if hasattr(y_true, '_data'):
                # Extract X from the same source DataFrame
                from IncomeEstimation.src.utils.paths import get_processed_data_dir
                import pandas as pd
                
                # Try to load the processed test data
                processed_data_dir = get_processed_data_dir()
                test_path = processed_data_dir / "test.csv"
                if os.path.exists(test_path):
                    test_data = pd.read_csv(test_path)
                    X_test = test_data.drop(columns=['target'])
        
        # If we couldn't get X_test from the index, try another approach
        if X_test is None:
            # Try to get test data from some place where it's likely to be available
            import pandas as pd
            from IncomeEstimation.src.utils.paths import get_processed_data_dir
            
            try:
                # Try to load the processed test data
                processed_data_dir = get_processed_data_dir()
                test_path = processed_data_dir / "test.csv"
                if os.path.exists(test_path):
                    test_data = pd.read_csv(test_path)
                    X_test = test_data.drop(columns=['target'])
                    y_true_test = test_data['target']
                    
                    # Verify this is likely the same data
                    if len(y_true) == len(y_true_test):
                        logger.info("Using test data from processed directory")
                    else:
                        X_test = None
                        logger.warning("Test data length doesn't match y_true length")
            except Exception as e:
                logger.warning(f"Could not load test data: {e}")
                X_test = None
        
        if X_test is None:
            logger.error("Could not find X_test data for prediction")
            raise ValueError("Could not find X_test data for prediction")
            
        # Generate standard predictions
        standard_pred = xgboost_model.predict(X_test)
        
        # Now try to load the quantile model
        from IncomeEstimation.posthoc.quantile.model.quantile_model import QuantileRegressionModel
        quantile_path = Path(project_root) / "posthoc" / "quantile" / "results" / "models" / "quantile_model.joblib"
        
        if not os.path.exists(quantile_path):
            logger.warning(f"Quantile model not found at: {quantile_path}")
            raise FileNotFoundError(f"Quantile model not found at: {quantile_path}")
            
        # Load Quantile model
        logger.info(f"Loading Quantile model from: {quantile_path}")
        quantile_model = QuantileRegressionModel.load(quantile_path, xgboost_model)
        
        # Make quantile predictions
        logger.info("Generating quantile model predictions...")
        quantile_pred = quantile_model.predict(X_test)
        
        # Generate comparison plot
        logger.info(f"Generating model comparison visualization for {prehoc_name}...")
        return plot_model_comparison(
            y_true=y_true,
            standard_pred=standard_pred,
            quantile_pred=quantile_pred,
            prehoc_pred=prehoc_pred,
            prehoc_name=prehoc_name,
            color_theme=color_theme,
            output_path=output_path,
            figsize=figsize
        )
        
    except Exception as e:
        logger.error(f"Error generating model comparison: {e}")
        return None
