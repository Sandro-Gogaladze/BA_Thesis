"""
Model Evaluator class for comprehensive model evaluation
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import shap

from IncomeEstimation.src.toolkit.theme import Theme

class ModelEvaluator:
    """
    A class for evaluating regression model performance with rich visualizations,
    metrics summaries, and SHAP/PDP-based explainability tools. 

    Supports theming via the Theme class and dynamic threshold analysis
    to assess conservative prediction behavior.
    """
    
    def __init__(self, theme='blue', dynamic_threshold_absolute=200, dynamic_threshold_percentage=20):
        """
        Initialize the ModelEvaluator with a visual theme and threshold parameters.

        Parameters:
        -----------
        theme : str or Theme
            The theme to use for visualizations.
        dynamic_threshold_absolute : float
            The minimum threshold for overestimation detection.
        dynamic_threshold_percentage : float
            The percentage-based threshold for overestimation detection.
        """
        # Set theme
        if isinstance(theme, str):
            self.theme = Theme(theme)
        elif isinstance(theme, Theme):
            self.theme = theme
        else:
            print("Warning: Invalid theme type. Using default 'blue' theme.")
            self.theme = Theme('blue')
        
        # Set dynamic threshold parameters
        self.dynamic_threshold_absolute = dynamic_threshold_absolute
        self.dynamic_threshold_percentage = dynamic_threshold_percentage
        
        # Apply theme palette to seaborn
        self.theme.set_palette()
    
    def create_analysis_dataframe(self, y_true, y_pred, dynamic_threshold=True):
        """
        Create a DataFrame that contains true/predicted values, errors,
        segmented income ranges, and error bucket labels.

        Optionally computes dynamic thresholds for conservative predictions.

        Returns:
        --------
        pd.DataFrame
            Enriched DataFrame for downstream analysis.
        """
        df = pd.DataFrame({
            "true": y_true,
            "pred": y_pred,
            "error": y_pred - y_true,
            "error_pct": (y_pred - y_true) / (y_true + 1e-6) * 100,
            "abs_error_pct": abs((y_pred - y_true) / (y_true + 1e-6) * 100)
        })
        
        # Create income ranges for segmented analysis
        df["range"] = pd.cut(
            df["true"],
            bins=[0, 1500, 2500, float('inf')],
            labels=["Low (≤1500)", "Mid (1500-2500)", "High (>2500)"]
        )
        
        # Error buckets
        bins = [-np.inf, -30, -20, -10, 0, 10, 20, 30, np.inf]
        labels = [
            "Under 30%+", "Under 20–30%", "Under 10–20%", "Under 0–10%",
            "Over 0–10%", "Over 10–20%", "Over 20–30%", "Over 30%+"
        ]
        df["bucket"] = pd.cut(df["error_pct"], bins=bins, labels=labels)
        
        # Add dynamic threshold calculations if requested
        if dynamic_threshold:
            # Calculate the dynamic threshold for each prediction
            df["dynamic_threshold"] = np.maximum(
                self.dynamic_threshold_absolute,
                df["true"] * (self.dynamic_threshold_percentage / 100)
            )
            
            # Flag overestimations and threshold exceedances
            df["is_overestimation"] = df["error"] > 0
            df["exceeds_threshold"] = df["error"] > df["dynamic_threshold"]
        
        return df

    def calculate_basic_metrics(self, y_true, y_pred, df=None):
        """
        Compute standard regression evaluation metrics and threshold-aware stats.

        Returns:
        --------
        dict
            Dictionary of metrics such as RMSE, MAE, R², MAPE, and threshold-based metrics.
        """
        # If no DataFrame is provided, create one
        if df is None:
            df = self.create_analysis_dataframe(y_true, y_pred, dynamic_threshold=True)
        
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100,
            "r2": r2_score(y_true, y_pred),
            "within_10pct": (df["abs_error_pct"] <= 10).mean() * 100,
            "within_20pct": (df["abs_error_pct"] <= 20).mean() * 100,
            "within_30pct": (df["abs_error_pct"] <= 30).mean() * 100
        }
        
        # Add threshold metrics if available
        if "is_overestimation" in df.columns and "exceeds_threshold" in df.columns:
            threshold_metrics = {
                "overestimation_pct": df["is_overestimation"].mean() * 100,
                "overestimation_20plus_pct": (df["error_pct"] > 20).mean() * 100,
                "exceeds_threshold_pct": df["exceeds_threshold"].mean() * 100,
                "avg_dynamic_threshold": df["dynamic_threshold"].mean() if "dynamic_threshold" in df.columns else 0
            }
            metrics.update(threshold_metrics)
        
        return metrics

    def calculate_segment_metrics(self, df, include_threshold_metrics=True):
        """
        Calculate error metrics per income range segment and the whole dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with evaluation metrics from create_analysis_dataframe.
        include_threshold_metrics : bool
            Whether to include threshold-related metrics in segment evaluation.

        Returns:
        --------
        pd.DataFrame
            Metrics by segment in tabular form.
        """
        segments = []
        
        for segment in df["range"].unique():
            segment_data = df[df["range"] == segment]
            
            if len(segment_data) > 0:
                # Basic metrics
                segment_metrics = {
                    "segment": segment,
                    "count": len(segment_data),
                    "r2": r2_score(segment_data["true"], segment_data["pred"]),
                    "rmse": np.sqrt(mean_squared_error(segment_data["true"], segment_data["pred"])),
                    "within_10pct": (segment_data["abs_error_pct"] <= 10).mean() * 100,
                    "within_20pct": (segment_data["abs_error_pct"] <= 20).mean() * 100,
                    "within_30pct": (segment_data["abs_error_pct"] <= 30).mean() * 100
                }
                
                # Add threshold metrics if requested and available
                if include_threshold_metrics and "is_overestimation" in df.columns:
                    threshold_metrics = {
                        "overestimation_pct": segment_data["is_overestimation"].mean() * 100,
                        "overestimation_20plus_pct": (segment_data["error_pct"] > 20).mean() * 100
                    }
                    segment_metrics.update(threshold_metrics)
                    
                    if "exceeds_threshold" in df.columns:
                        dynamic_metrics = {
                            "exceeds_threshold_pct": segment_data["exceeds_threshold"].mean() * 100,
                            "avg_dynamic_threshold": segment_data["dynamic_threshold"].mean() if "dynamic_threshold" in df.columns else 0
                        }
                        segment_metrics.update(dynamic_metrics)
                
                segments.append(segment_metrics)
        
        # Add all data segment
        all_metrics = {
            "segment": "All Data",
            "count": len(df),
            "r2": r2_score(df["true"], df["pred"]),
            "rmse": np.sqrt(mean_squared_error(df["true"], df["pred"])),
            "within_10pct": (df["abs_error_pct"] <= 10).mean() * 100,
            "within_20pct": (df["abs_error_pct"] <= 20).mean() * 100,
            "within_30pct": (df["abs_error_pct"] <= 30).mean() * 100
        }
        
        # Add threshold metrics for all data if requested and available
        if include_threshold_metrics and "is_overestimation" in df.columns:
            all_threshold_metrics = {
                "overestimation_pct": df["is_overestimation"].mean() * 100,
                "overestimation_20plus_pct": (df["error_pct"] > 20).mean() * 100
            }
            all_metrics.update(all_threshold_metrics)
            
            if "exceeds_threshold" in df.columns:
                all_dynamic_metrics = {
                    "exceeds_threshold_pct": df["exceeds_threshold"].mean() * 100,
                    "avg_dynamic_threshold": df["dynamic_threshold"].mean() if "dynamic_threshold" in df.columns else 0
                }
                all_metrics.update(all_dynamic_metrics)
        
        segments.append(all_metrics)
        
        return pd.DataFrame(segments)

    def print_metrics_table(self, train_metrics, test_metrics):
        """
        Pretty-print model performance metrics for train and test sets side by side.

        Parameters:
        -----------
        train_metrics : dict
        test_metrics : dict
            Output from calculate_basic_metrics for respective datasets.
        """
        colors = self.theme.get_colors()
        title_color = colors['accent']
        
        print(f"\n\033[1;38;2;{int(title_color[1:3], 16)};{int(title_color[3:5], 16)};{int(title_color[5:7], 16)}m📊 Model Performance and Threshold Metrics\033[0m")
        print("-" * 60)
        print(f"{'Metric':<30} {'Train':<15} {'Test':<15}")
        print("-" * 60)
        print(f"{'Standard Metrics:'}")
        print(f"{'RMSE':<30} {train_metrics['rmse']:<15.4f} {test_metrics['rmse']:<15.4f}")
        print(f"{'MAE':<30} {train_metrics['mae']:<15.4f} {test_metrics['mae']:<15.4f}")
        print(f"{'MAPE (%)':<30} {train_metrics['mape']:<15.4f} {test_metrics['mape']:<15.4f}")
        print(f"{'R²':<30} {train_metrics['r2']:<15.4f} {test_metrics['r2']:<15.4f}")
        print("\n{'Prediction Accuracy:'}")
        print(f"{'Within 10% (%)':<30} {train_metrics['within_10pct']:<14.2f} {test_metrics['within_10pct']:<14.2f}")
        print(f"{'Within 20% (%)':<30} {train_metrics['within_20pct']:<14.2f} {test_metrics['within_20pct']:<14.2f}")
        print(f"{'Within 30% (%)':<30} {train_metrics['within_30pct']:<14.2f} {test_metrics['within_30pct']:<14.2f}")
        
        # Print threshold metrics if available
        if "exceeds_threshold_pct" in train_metrics and "exceeds_threshold_pct" in test_metrics:
            print("\n{'Threshold Metrics:'}")
            print(f"{'Overestimations (%)':<30} {train_metrics['overestimation_pct']:<14.2f} {test_metrics['overestimation_pct']:<14.2f}")
            print(f"{'Overestimations >20% (%)':<30} {train_metrics['overestimation_20plus_pct']:<14.2f} {test_metrics['overestimation_20plus_pct']:<14.2f}")
            thresh_str = f"Exceeds max({self.dynamic_threshold_absolute},{self.dynamic_threshold_percentage}%) (%)"
            print(f"{thresh_str:<30} {train_metrics['exceeds_threshold_pct']:<14.2f} {test_metrics['exceeds_threshold_pct']:<14.2f}")

    def format_and_display_segment_metrics(self, segment_df, columns=None, rename_dict=None):
        """
        Clean and format segment metrics DataFrame for display.

        Parameters:
        -----------
        segment_df : pd.DataFrame
        columns : list
            Columns to include in formatted output.
        rename_dict : dict
            Optional column renaming.

        Returns:
        --------
        pd.DataFrame
            Formatted metrics table.
        """
        if columns is None:
            columns = ["segment", "count", "rmse", "r2", "within_10pct", 
                      "within_20pct", "exceeds_threshold_pct"]
        
        if rename_dict is None:
            rename_dict = {
                "rmse": "RMSE",
                "r2": "R²",
                "within_10pct": "Within 10% (%)",
                "within_20pct": "Within 20% (%)",
                "exceeds_threshold_pct": f"Exceeds max({self.dynamic_threshold_absolute},{self.dynamic_threshold_percentage}%) (%)"
            }
        
        # Create display DataFrame
        display_df = segment_df[columns].copy()
        display_df = display_df.rename(columns=rename_dict)
        
        # Round numeric columns
        for col in display_df.columns:
            if col != "segment" and col != "count":
                display_df[col] = display_df[col].round(2)
        
        return display_df
    
    def plot_calibration(self, df, ax=None, enhanced=True):
        """
        Plot a scatter of actual vs. predicted values colored by absolute error percentage.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame from create_analysis_dataframe.
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes to plot on. If None, creates a new one.
        enhanced : bool
            Whether to apply themed styling and color bar.

        Returns:
        --------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        colors = self.theme.get_colors()
        
        # Get the specific gradient colors for the current theme
        gradient_colors = colors['gradient']
        
        # Create a custom colormap based on the theme's gradient
        theme_cmap = LinearSegmentedColormap.from_list("theme_cmap", gradient_colors, N=256)
        
        if enhanced:
            # Use the theme-specific colormap
            scatter = ax.scatter(
                df["true"], 
                df["pred"], 
                alpha=0.8,
                c=df["abs_error_pct"],
                cmap=theme_cmap,  # Use theme-specific colormap
                s=30,
                edgecolor='none'
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Absolute % Error")
            
            max_val = max(df["true"].max(), df["pred"].max()) * 1.05
            min_val = min(df["true"].min(), df["pred"].min()) * 0.95
        else:
            scatter = ax.scatter(
                df["true"], 
                df["pred"], 
                alpha=0.7, 
                c=df["abs_error_pct"],
                cmap=theme_cmap,
                s=30
            )
        
        max_val = max(df["true"].max(), df["pred"].max()) * 1.05
        min_val = min(df["true"].min(), df["pred"].min()) * 0.95
        
        # Perfect prediction line
        ax.plot([min_val, max_val], [min_val, max_val], '--', 
                color='#333333', alpha=0.6, linewidth=1.5)
        
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title("Prediction Calibration", fontsize=14)
        
        # Add grid and styling
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return ax

    def plot_error_distribution(self, df, ax=None, enhanced=True):
        """
        Visualize the distribution of raw prediction errors.

        Enhanced mode includes mean/std deviation annotations.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        colors = self.theme.get_colors()
        
        if enhanced:
            # Enhanced distribution plot without kde_kws parameter
            sns.histplot(
                df["error"], 
                bins=30, 
                kde=True, 
                ax=ax, 
                color=colors['primary'],
                alpha=0.7
            )
            
            # Add a vertical line at 0 (no error)
            ax.axvline(0, color=colors['error'], linestyle='--', alpha=0.8, linewidth=2)
            
            # Add annotations for mean and standard deviation
            mean_error = df["error"].mean()
            std_error = df["error"].std()
            
            ax.axvline(mean_error, color=colors['secondary'], linestyle='-', 
                    alpha=0.8, label=f"Mean Error: {mean_error:.1f}")
            
            # Add annotation boxes
            # Calculate y positions dynamically
            y_height = ax.get_ylim()[1]
            
            ax.annotate(f'Mean: {mean_error:.1f}', 
                    xy=(mean_error, y_height*0.9),
                    xytext=(mean_error + std_error, y_height*0.9),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=colors['accent']),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors['accent'], alpha=0.7))
            
            ax.annotate(f'Std Dev: {std_error:.1f}', 
                    xy=(mean_error + std_error, y_height*0.75),
                    xytext=(mean_error + std_error*1.5, y_height*0.75),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=colors['accent']),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors['accent'], alpha=0.7))
            
            ax.legend()
        else:
            sns.histplot(df["error"], bins=30, kde=True, ax=ax, color=colors['primary'])
            ax.axvline(0, color=colors['error'], linestyle='--', alpha=0.8)
        
        ax.set_xlabel("Prediction Error", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Error Distribution", fontsize=14)
        
        return ax

    def plot_error_buckets_by_range(self, df, ax=None, enhanced=True):
        """
        Visualize distribution of errors across income ranges using labeled error buckets.

        Adds annotations for under- and over-estimation regions in enhanced mode.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))
        
        colors = self.theme.get_colors()
        
        # Prepare data for error buckets
        income_ranges = sorted(df["range"].unique())
        bucket_order = [
            "Under 30%+", "Under 20–30%", "Under 10–20%", "Under 0–10%",
            "Over 0–10%", "Over 10–20%", "Over 20–30%", "Over 30%+"
        ]
        
        # Calculate percentage in each bucket for each range
        bucket_data = []
        for income_range in income_ranges:
            range_data = df[df["range"] == income_range]
            for bucket in bucket_order:
                bucket_data.append({
                    "range": income_range,
                    "bucket": bucket,
                    "percentage": (range_data["bucket"] == bucket).mean() * 100
                })
        
        # Add "All Data"
        for bucket in bucket_order:
            bucket_data.append({
                "range": "All Data",
                "bucket": bucket,
                "percentage": (df["bucket"] == bucket).mean() * 100
            })
        
        bucket_df = pd.DataFrame(bucket_data)
        
        if enhanced:
            # Enhanced bar plot
            palette = colors['categorical']
            
            # First, add the shaded regions before plotting the bars
            # Add under/over estimation regions with theme-appropriate colors
            middle_idx = len(bucket_order) // 2
            
            # Use primary color with alpha for under-estimation regions
            under_color = colors['primary']
            # Use error color with alpha for over-estimation regions
            over_color = colors['error']
            
            for i, bucket in enumerate(bucket_order):
                if i < middle_idx:  # Under estimation buckets
                    ax.axvspan(i-0.4, i+0.4, color=under_color, alpha=0.05, zorder=0)
                else:  # Over estimation buckets
                    ax.axvspan(i-0.4, i+0.4, color=over_color, alpha=0.05, zorder=0)
            
            # Now plot the bar chart on top
            sns.barplot(
                data=bucket_df,
                x="bucket",
                y="percentage",
                hue="range",
                ax=ax,
                palette=palette,
                alpha=0.8,
                edgecolor='white',
                linewidth=1,
                zorder=1  # Ensure bars are plotted above the shaded regions
            )
            
            # Add value labels with rounded percentages and smaller font
            for i, bar_container in enumerate(ax.containers):
                ax.bar_label(bar_container, 
                            fmt='%.0f%%',  # Round to integers
                            fontsize=7,    # Smaller font size
                            padding=3)     # Less padding
            
            # Get the current y-axis limits
            y_max = ax.get_ylim()[1]
            
            # Add labels with high zorder to bring them to the front, using theme colors
            # Use theme primary color for UNDER-ESTIMATION
            ax.annotate("UNDER-ESTIMATION", 
                    xy=(1.5, y_max * 0.6),
                    xytext=(1.5, y_max * 0.6),
                    ha='center', 
                    color=colors['primary'],  # Use theme primary color
                    fontsize=14,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors['primary'], alpha=0.9),
                    zorder=10)
            
            # Use theme error color for OVER-ESTIMATION
            ax.annotate("OVER-ESTIMATION", 
                    xy=(5.5, y_max * 0.6),
                    xytext=(5.5, y_max * 0.6),
                    ha='center', 
                    color=colors['error'],  # Use theme error color
                    fontsize=14,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors['error'], alpha=0.9),
                    zorder=10)
        else:
            sns.barplot(
                data=bucket_df,
                x="bucket",
                y="percentage",
                hue="range",
                ax=ax
            )
        
        ax.set_title("Error Distribution by Income Range", fontsize=14)
        ax.set_xlabel("Error Bucket", fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Move legend to the top-left corner and make it more compact
        ax.legend(title="Income Range", fontsize=9, title_fontsize=10, loc='upper left')
        
        return ax

    def plot_error_vs_actual(self, df, ax=None, enhanced=True):
        """
        Scatter plot of error percentage vs. true value.

        Helpful to observe systematic over- or under-predictions.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        colors = self.theme.get_colors()
        
        # Get the specific gradient colors for the current theme
        gradient_colors = colors['gradient']
        
        # Create a custom colormap based on the theme's gradient
        theme_cmap = LinearSegmentedColormap.from_list("theme_cmap", gradient_colors, N=256)
        
        if enhanced:
            # Use the theme-specific colormap
            scatter = ax.scatter(
                df["true"], 
                df["error_pct"], 
                alpha=0.8,
                c=np.abs(df["error_pct"]),
                cmap=theme_cmap,  # Use theme-specific colormap
                s=30,
                edgecolor='none'
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Absolute % Error")
            
            # Zero line - solid
            ax.axhline(0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
            
        else:
            ax.scatter(df["true"], df["error_pct"], alpha=0.7, c=np.abs(df["error_pct"]), cmap=theme_cmap, s=30)
            ax.axhline(0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel("Actual Value", fontsize=12)
        ax.set_ylabel("Prediction Error (%)", fontsize=12)
        ax.set_title("Prediction Error (%) vs. Actual Value", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return ax

    def plot_error_vs_dynamic_threshold(self, df, ax=None, enhanced=True):
        """
        Plot raw prediction errors against actual values, 
        with dynamic threshold overlays and classifications.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        colors = self.theme.get_colors()
        
        # Ensure dynamic threshold columns exist
        if "dynamic_threshold" not in df.columns or "exceeds_threshold" not in df.columns:
            raise ValueError("DataFrame must include dynamic threshold calculations")
        
        # Use theme colors instead of hardcoded orange
        # For points that exceed threshold, use the 'error' color from the theme
        # For points within threshold, use the 'primary' color from the theme
        point_colors = np.where(df["exceeds_threshold"], colors['error'], colors['primary'])
        
        if enhanced:
            # Enhanced scatter plot
            ax.scatter(
                df["true"], 
                df["error"], 
                alpha=0.7, 
                c=point_colors,
                s=np.minimum(df["abs_error_pct"], 50) + 20,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Add reference lines and regions
            unique_true_values = np.sort(df["true"].unique())
            dynamic_thresholds = np.maximum(
                self.dynamic_threshold_absolute, 
                unique_true_values * (self.dynamic_threshold_percentage / 100)
            )
            
            # Add threshold lines using theme colors
            # Upper threshold (use error color from theme instead of orange)
            ax.plot(
                unique_true_values, 
                dynamic_thresholds, 
                color=colors['error'],  # Use error color from theme
                linestyle='--', 
                alpha=0.8, 
                linewidth=2,
                label=f"Dynamic Threshold (max({self.dynamic_threshold_absolute}, {self.dynamic_threshold_percentage}%))"
            )
            
            # Lower threshold (for reference)
            ax.plot(
                unique_true_values, 
                -dynamic_thresholds, 
                color=colors['error'],  # Use error color from theme
                linestyle=':', 
                alpha=0.5, 
                linewidth=1,
                label="Lower Bound (for reference)"
            )
            
            # Fill between thresholds with light theme color
            ax.fill_between(
                unique_true_values,
                -dynamic_thresholds,
                dynamic_thresholds,
                color=colors['secondary'],
                alpha=0.1
            )
        else:
            # Basic scatter plot
            ax.scatter(
                df["true"], 
                df["error"], 
                alpha=0.5, 
                c=point_colors,
                s=30
            )
            
            # Add threshold line
            unique_true_values = np.sort(df["true"].unique())
            dynamic_thresholds = np.maximum(
                self.dynamic_threshold_absolute, 
                unique_true_values * (self.dynamic_threshold_percentage / 100)
            )
            
            ax.plot(
                unique_true_values, 
                dynamic_thresholds, 
                color=colors['error'],  # Use error color from theme
                linestyle='--', 
                alpha=0.8,
                label=f"Dynamic Threshold (max({self.dynamic_threshold_absolute}, {self.dynamic_threshold_percentage}%))"
            )
        
        # Zero line
        ax.axhline(0, color="black", linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Add legend with custom colored points
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['primary'], 
                markersize=10, label='Within Threshold'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['error'], 
                markersize=10, label='Exceeds Threshold'),
            Line2D([0], [0], color=colors['error'], linestyle='--', linewidth=2,
                label=f"Dynamic Threshold (max({self.dynamic_threshold_absolute}, {self.dynamic_threshold_percentage}%))"),
            Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='No Error')
        ]
        
        if enhanced:
            legend_elements.append(
                Line2D([0], [0], color=colors['error'], linestyle=':', linewidth=1,
                    label="Lower Bound (for reference)")
            )
        ax.legend(handles=legend_elements, loc='best')
        
        # Labels and grid
        ax.set_xlabel("Actual Value", fontsize=12)
        ax.set_ylabel("Error (Predicted - Actual)", fontsize=12)
        ax.set_title("Error vs. Actual Value with Dynamic Threshold", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return ax
    
    def plot_threshold_exceedance_by_range(self, df, ax=None, enhanced=True):
        """
        Show % of predictions that exceed dynamic thresholds, segmented by income range.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        colors = self.theme.get_colors()
        
        # Calculate percentage exceeding threshold by range
        ranges = list(df["range"].unique()) + ["All Data"]
        percentages = []
        
        for range_name in ranges:
            if range_name == "All Data":
                percentage = df["exceeds_threshold"].mean() * 100
            else:
                range_data = df[df["range"] == range_name]
                percentage = range_data["exceeds_threshold"].mean() * 100
            
            percentages.append(percentage)
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            "Income Range": ranges,
            "Exceeds Threshold (%)": percentages
        })
        
        # Sort to ensure consistent order
        range_order = ["High (>2500)", "Mid (1500-2500)", "Low (≤1500)", "All Data"]
        plot_df["Income Range"] = pd.Categorical(
            plot_df["Income Range"],
            categories=range_order,
            ordered=True
        )
        plot_df = plot_df.sort_values("Income Range")
        
        if enhanced:
            # Enhanced bar plot with theme colors
            bar_colors = colors['categorical'][:len(plot_df)]
            
            bars = ax.bar(
                plot_df["Income Range"],
                plot_df["Exceeds Threshold (%)"],
                color=bar_colors,
                alpha=0.8,
                edgecolor='white',
                linewidth=1,
                width=0.7
            )
            
            # Add bar labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.5,
                    f'{height:.1f}%',
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight='bold',
                    color=colors['accent']
                )
            
            # Calculate the overall average
            avg_exceed = plot_df.loc[plot_df["Income Range"] == "All Data", "Exceeds Threshold (%)"].values[0]
            
            # Add overall average line with label
            ax.axhline(avg_exceed, color=colors['accent'], linestyle='--', 
                    linewidth=2, alpha=0.7, label=f"Overall Average: {avg_exceed:.1f}%")
            
            # Add acceptable threshold lines at 10% and 20%
            ax.axhline(10, color='green', linestyle='-', 
                    linewidth=1.5, alpha=0.6, label="Acceptable (<10%)")
            
            # Add legend in upper right
            ax.legend(loc='upper right')
        else:
            # Basic bar plot
            bars = ax.bar(
                plot_df["Income Range"],
                plot_df["Exceeds Threshold (%)"],
                color=[colors['primary'], colors['secondary'], colors['accent'], colors['error']]
            )
            
            # Add simple labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.5,
                    f'{height:.1f}%',
                    ha="center"
                )
        
        # Configure plot
        ax.set_xlabel("Income Range", fontsize=12)
        ax.set_ylabel(f"Exceeds Threshold (max({self.dynamic_threshold_absolute}, {self.dynamic_threshold_percentage}%)) (%)", fontsize=12)
        ax.set_title("Dynamic Threshold Exceedance by Income Range", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        return ax

    def plot_feature_importance(self, model, X_test, n_top=20, enhanced=True, figures_dir=None):
        """
        Generate SHAP summary plots of feature importances.

        Falls back to native model importances if SHAP fails.

        Parameters:
        -----------
        model : object
            The trained model object
        X_test : pd.DataFrame
            Test features for SHAP analysis
        n_top : int
            Number of top features to display
        enhanced : bool
            Whether to use themed annotations and enhancements
        figures_dir : str, optional
            Directory to save figures to. If None, uses default xgboost figures location.

        Returns:
        --------
        pd.DataFrame
            Feature importances sorted by mean absolute SHAP value.
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Apply theme styling to figure
        self.theme.apply_figure_style(plt.gcf())
        
        colors = self.theme.get_colors()
        
        try:
            # Get native feature importances for comparison
            if hasattr(model, 'feature_importances_'):
                # Standard sklearn-like feature importance
                native_importances = model.feature_importances_
                feature_names = X_test.columns
                importance_type = "Native feature importance"
                
            elif hasattr(model, 'get_booster'):
                # XGBoost-specific gain-based importance
                booster = model.get_booster()
                importance_dict = booster.get_score(importance_type='gain')
                feature_names = list(importance_dict.keys())
                native_importances = list(importance_dict.values())
                importance_type = "XGBoost gain importance"
                
            elif hasattr(model, 'get_score'):
                # Direct XGBoost booster case
                importance_dict = model.get_score(importance_type='gain')
                feature_names = list(importance_dict.keys())
                native_importances = list(importance_dict.values())
                importance_type = "XGBoost gain importance"
            else:
                # If we can't get native importances, use feature names from data
                feature_names = X_test.columns
                native_importances = np.ones(len(feature_names))
                importance_type = "Placeholder importance"
            
            # Create a basic importance DataFrame from native importances
            native_importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": native_importances
            }).sort_values(by="importance", ascending=False)
            
            # Calculate SHAP values for more accurate feature importance
            # Use a reasonable sample size for SHAP computation
            sample_size = min(1000, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test
            
            # Create SHAP explainer
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            
            # Calculate mean absolute SHAP value for each feature
            shap_importance = np.abs(shap_values.values).mean(axis=0)
            
            # Create SHAP-based importance DataFrame
            shap_importance_df = pd.DataFrame({
                "feature": X_test.columns,
                "importance": shap_importance
            }).sort_values(by="importance", ascending=False)
            
            # Plot classic SHAP summary plot
            shap.summary_plot(
                shap_values, 
                X_sample,
                max_display=min(n_top, len(X_test.columns)),
                plot_size=(12, 8),
                show=False
            )
            
            plt.title("SHAP Feature Impact Summary\n(red = high feature value, blue = low feature value)", 
                    fontsize=16, pad=20)
            plt.tight_layout()
            
            # Save the SHAP summary plot
            if figures_dir is None:
                figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                         "baseline", "xgboost", "results", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            shap_path = os.path.join(figures_dir, "shap_feature_importance.png")
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Return the SHAP-based importance DataFrame
            return shap_importance_df
            
        except Exception as e:
            print(f"Warning: Could not create SHAP feature importance plot - {str(e)}")
            plt.close()
            
            # Try to return native importance if available
            if 'native_importance_df' in locals():
                print("Falling back to native feature importance.")
                return native_importance_df
            else:
                # Create an empty importance DataFrame with feature names
                return pd.DataFrame({"feature": X_test.columns, "importance": np.ones(len(X_test.columns))})

    def plot_feature_dependence(self, model, X_test, importance_df=None, top_n=5, features=None, enhanced=True, figures_dir=None):
        """
        Plot SHAP dependence and partial dependence for multiple features in a single figure
        using subplots for each feature.
        
        Parameters:
        -----------
        model : model object
            Trained model to analyze
        X_test : DataFrame
            Test data for feature dependence analysis
        importance_df : DataFrame, optional
            DataFrame with feature importance rankings, if None will be calculated
        top_n : int, default=5
            Number of top features to analyze if features parameter is None
        features : list, optional
            List of specific feature names to analyze, overrides top_n
        enhanced : bool, default=True
            Whether to use enhanced styling
        figures_dir : str, optional
            Directory to save figures to. If None, uses default xgboost figures location.
        """
        print("\n🔍 Analyzing feature dependence...")
        colors = self.theme.get_colors()
        
        # If no importance_df is provided, calculate it
        if importance_df is None or len(importance_df) == 0:
            importance_df = self.plot_feature_importance(model, X_test, n_top=top_n, enhanced=enhanced)
        
        # Get features to analyze - either specified list or top features from importance
        if features is not None:
            top_features = [f for f in features if f in X_test.columns]
            if len(top_features) == 0:
                print("None of the specified features were found in the dataset.")
                top_features = importance_df.head(min(top_n, len(importance_df)))["feature"].tolist()
        else:
            top_features = importance_df.head(min(top_n, len(importance_df)))["feature"].tolist()
        
        print(f"Analyzing feature dependence for: {', '.join(top_features)}")
        
        try:
            # Calculate SHAP values
            print("Calculating SHAP values...")
            
            # Use an appropriate sample size
            sample_size = min(1000, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test
            
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            
            # Determine grid layout for subplots
            n_features = len(top_features)
            n_cols = 2  # Two columns (SHAP and PDP side by side for each feature)
            n_rows = n_features  # One row per feature
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 6 * n_rows))
            
            # Apply theme styling
            self.theme.apply_figure_style(fig)
            
            # Create a grid of subplots
            gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0.3, hspace=0.4)
            
            # For each feature, create SHAP and PDP plots
            for i, feature in enumerate(top_features):
                # Get feature ranking and importance value
                feature_rank = importance_df[importance_df['feature'] == feature].index[0] + 1
                importance_value = importance_df[importance_df['feature'] == feature]['importance'].values[0]
                
                # Create subplot for SHAP dependence
                ax_shap = fig.add_subplot(gs[i, 0])
                
                # Create subplot for PDP
                ax_pdp = fig.add_subplot(gs[i, 1])
                
                # 1. SHAP dependence plot
                try:
                    shap.dependence_plot(
                        feature, 
                        shap_values.values, 
                        X_sample,
                        ax=ax_shap,
                        show=False
                    )
                    ax_shap.set_title(f"SHAP Dependence: {feature}", fontsize=12)
                except Exception as e:
                    print(f"Could not create SHAP plot for {feature}: {str(e)}")
                    ax_shap.text(0.5, 0.5, f"SHAP dependence unavailable", 
                            ha='center', va='center', fontsize=10)
                
                # 2. Partial Dependence Plot
                try:
                    import xgboost as xgb
                    from sklearn.inspection import partial_dependence
                    
                    # Check if it's a sklearn-compatible model
                    if hasattr(model, 'fit') and hasattr(model, 'predict') and callable(model.predict):
                        # For sklearn API compatible models
                        disp = PartialDependenceDisplay.from_estimator(
                            model,
                            X_test,
                            [feature],
                            ax=ax_pdp,
                            grid_resolution=50,
                            line_kw={"color": colors['primary'], "linewidth": 2.5, "alpha": 0.8}
                        )
                    else:
                        # For XGBoost Booster model, calculate PDPs manually
                        print(f"Creating manual PDP for feature: {feature}")
                        
                        # Create grid of values
                        feature_values = np.linspace(
                            X_test[feature].min(),
                            X_test[feature].max(),
                            num=50
                        )
                        
                        # Use a subset of data for efficiency
                        sample_size = min(1000, len(X_test))
                        X_sample = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test.copy()
                        
                        # Calculate partial dependence efficiently
                        pd_values = []
                        
                        # For efficiency, create feature grid once and calculate in batches
                        original_values = X_sample[feature].copy()
                        
                        for value in feature_values:
                            # Set all rows to the same feature value
                            X_sample[feature] = value
                            
                            try:
                                # Predict using the model
                                dmatrix = xgb.DMatrix(X_sample)
                                preds = model.predict(dmatrix)
                                
                                # Store the average prediction
                                pd_values.append(np.mean(preds))
                            except Exception as e:
                                print(f"Warning during PDP calculation ({feature}={value}): {str(e)}")
                                # If prediction fails, use previous value or zero
                                if pd_values:
                                    pd_values.append(pd_values[-1])
                                else:
                                    pd_values.append(0)
                        
                        # Restore original feature values
                        X_sample[feature] = original_values
                        
                        # Convert to numpy array
                        pd_values = np.array(pd_values)
                        
                        # Plot the PDP
                        ax_pdp.plot(feature_values, pd_values, 
                                   color=colors['primary'], 
                                   linewidth=2.5, 
                                   alpha=0.8)
                        ax_pdp.set_xlabel(feature)
                        ax_pdp.set_ylabel('Partial Dependence')
                        
                        # Plotting is done above this point
                    
                    # Add feature distribution on twin axis
                    ax_hist = ax_pdp.twinx()
                    
                    # Histogram for feature distribution
                    sns.histplot(
                        X_test[feature], 
                        bins=15, 
                        ax=ax_hist, 
                        color=colors['secondary'], 
                        alpha=0.3,
                        kde=True
                    )
                    ax_hist.set_ylabel("Distribution", fontsize=9)
                    ax_hist.tick_params(axis='y', labelsize=8)
                    ax_hist.grid(False)
                    
                    # Add title with feature rank and importance
                    ax_pdp.set_title(f"PDP: {feature} (#{feature_rank}, Imp: {importance_value:.2f})", fontsize=12)
                    
                except Exception as e:
                    print(f"Could not create PDP for {feature}: {str(e)}")
                    ax_pdp.text(0.5, 0.5, f"Partial dependence unavailable: {str(e)}", 
                            ha='center', va='center', fontsize=10)
            
            # Add an overall title
            plt.suptitle("Feature Dependency Analysis - SHAP and Partial Dependence", 
                        fontsize=18, y=0.98)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.96)
            
            # Save the figure if a directory is provided
            if figures_dir:
                os.makedirs(figures_dir, exist_ok=True)
                dep_path = os.path.join(figures_dir, "feature_dependence.png")
                plt.savefig(dep_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.close()
            
            return importance_df
            
        except Exception as e:
            print(f"Warning: Could not create feature dependence plots - {str(e)}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Feature dependence analysis unavailable", 
                    ha='center', va='center', fontsize=14)
            plt.title("Feature Dependence Analysis (Error Occurred)", fontsize=16)
            if figures_dir:
                os.makedirs(figures_dir, exist_ok=True)
                error_path = os.path.join(figures_dir, "feature_dependence_error.png")
                plt.savefig(error_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.close()
            return importance_df

    def plot_comprehensive_evaluation(self, df, dataset_name="Dataset", enhanced=True, save_path=None, figures_dir=None):
        """
        Generate a multi-panel diagnostic visualization suite
        covering all core error and threshold evaluation plots.

        Parameters:
        -----------
        df : pd.DataFrame
            Output from create_analysis_dataframe.
        dataset_name : str
            Label to use in the figure title.
        enhanced : bool
            Whether to use themed annotations and enhancements.
        save_path : str, optional
            Path to save the figure. If None, the figure is saved to figures_dir/comprehensive_evaluation.png.
        figures_dir : str, optional
            Directory to save figures to. If None, figure is not saved unless save_path is provided.
        """
        # Create figure with proper spacing
        fig = plt.figure(figsize=(20, 18))
        
        # Apply theme to figure
        self.theme.apply_figure_style(fig)
        
        # Create GridSpec with better spacing
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], 
                        hspace=0.4, wspace=0.4)
        
        # First row: Calibration and Error Distribution
        ax1 = plt.subplot(gs[0, 0])
        self.plot_calibration(df, ax1, enhanced=enhanced)
        
        # For the error distribution, use a simpler approach to avoid KDE issues
        ax2 = plt.subplot(gs[0, 1])
        colors = self.theme.get_colors()
        
        # Plot basic histogram without KDE
        sns.histplot(
            df["error"], 
            bins=30, 
            kde=False,  # Turn off KDE to avoid the error
            ax=ax2, 
            color=colors['primary'],
            alpha=0.7
        )
        
        # Add annotations and reference lines
        mean_error = df["error"].mean()
        std_error = df["error"].std()
        
        # Add vertical lines
        ax2.axvline(0, color=colors['error'], linestyle='--', alpha=0.8, linewidth=2)
        ax2.axvline(mean_error, color=colors['secondary'], linestyle='-', 
                alpha=0.8, label=f"Mean Error: {mean_error:.1f}")
        
        # Add annotations
        y_height = ax2.get_ylim()[1]
        ax2.annotate(f'Mean: {mean_error:.1f}', 
                xy=(mean_error, y_height*0.9),
                xytext=(mean_error + std_error, y_height*0.9),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=colors['accent']),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors['accent'], alpha=0.7))
        
        ax2.annotate(f'Std Dev: {std_error:.1f}', 
                xy=(mean_error + std_error, y_height*0.75),
                xytext=(mean_error + std_error*1.5, y_height*0.75),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=colors['accent']),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors['accent'], alpha=0.7))
        
        ax2.legend()
        ax2.set_xlabel("Prediction Error", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Error Distribution", fontsize=14)
        
        # Second row: Error % vs True Value and Error Distribution by Income Range
        ax3 = plt.subplot(gs[1, 0])
        self.plot_error_vs_actual(df, ax3, enhanced=enhanced)
        
        ax4 = plt.subplot(gs[1, 1])
        self.plot_error_buckets_by_range(df, ax4, enhanced=enhanced)
        
        # Third row: Error vs Actual with Dynamic Threshold and Threshold Exceedance by Range
        ax5 = plt.subplot(gs[2, 0])
        self.plot_error_vs_dynamic_threshold(df, ax5, enhanced=enhanced)
        
        ax6 = plt.subplot(gs[2, 1])
        self.plot_threshold_exceedance_by_range(df, ax6, enhanced=enhanced)
        
        # Add main title
        colors = self.theme.get_colors()
        title_color = colors['accent']
        plt.suptitle(f"{dataset_name} Prediction Analysis", fontsize=18, y=0.98, 
                    color=title_color, fontweight='bold')
        
        # Use subplots_adjust
        plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.4)
        
        # Save the figure to the specified path or to figures_dir
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif figures_dir:
            os.makedirs(figures_dir, exist_ok=True)
            save_path = os.path.join(figures_dir, "comprehensive_evaluation.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return fig
