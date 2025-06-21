"""
Theme class for visualization styling
"""
import matplotlib.pyplot as plt
import seaborn as sns

class Theme:
    """
    A class for managing visualization themes and color schemes.

    Supports multiple color themes for consistent visualization styling
    across plots and analyses.
    """
    
    # Define color themes
    THEMES = {
        'blue': {
            'primary': '#3498db',
            'secondary': '#85c1e9',
            'accent': '#2980b9',
            'error': '#e74c3c',
            'success': '#2ecc71',
            'gradient': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
            'categorical': ['#3498db', '#6baed6', '#9ecae1', '#2980b9', '#1f618d', '#d4e6f1', '#85c1e9']
        },
        'purple': {
            'primary': '#9b59b6',
            'secondary': '#d7bde2',
            'accent': '#8e44ad',
            'error': '#e74c3c',
            'success': '#2ecc71',
            'gradient': ['#f2f0f7', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'],
            'categorical': ['#9b59b6', '#9e9ac8', '#bcbddc', '#8e44ad', '#7d3c98', '#ebdef0', '#d7bde2']
        },
        'green': {
            'primary': '#2ecc71',
            'secondary': '#a9dfbf',
            'accent': '#27ae60',
            'error': '#e74c3c',
            'success': '#2ecc71',
            'gradient': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'],
            'categorical': ['#2ecc71', '#74c476', '#a1d99b', '#27ae60', '#229954', '#d5f5e3', '#a9dfbf']
        },
        'orange': {
            'primary': '#e67e22',
            'secondary': '#f5cba7',
            'accent': '#d35400',
            'error': '#e74c3c',
            'success': '#2ecc71',
            'gradient': ['#feedde', '#fdbe85', '#fd8d3c', '#e6550d', '#a63603'],
            'categorical': ['#e67e22', '#fd8d3c', '#fdae6b', '#e6550d', '#a63603', '#fdd0a2', '#fee6ce']
        },
        'teal': {
            'primary': '#1abc9c',
            'secondary': '#a3e4d7',
            'accent': '#16a085',
            'error': '#e74c3c',
            'success': '#2ecc71',
            'gradient': ['#e5f5f9', '#ccece6', '#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#005824'],
            'categorical': ['#1abc9c', '#66c2a4', '#a1d3c9', '#17a589', '#138d75', '#d1f2eb', '#a3e4d7']
        }
    }

    def __init__(self, color_theme='blue', style='whitegrid'):
        """
        Initialize and apply the specified theme and seaborn style.

        Parameters:
        -----------
        color_theme : str
            Name of the theme to use (default is 'blue').
        style : str
            Seaborn style to apply (e.g., 'whitegrid', 'darkgrid').
        """
        if color_theme not in self.THEMES:
            print(f"Warning: Theme '{color_theme}' not found. Using 'blue' instead.")
            color_theme = 'blue'

        self.colors = self.THEMES[color_theme]
        self.theme_name = color_theme

        # Apply seaborn styling
        sns.set_style(style)

        # Set global matplotlib visual settings
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'
        plt.rcParams['text.color'] = '#333333'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

    def set_palette(self):
        """
        Set seaborn color palette to match the current theme.
        """
        sns.set_palette(self.colors['categorical'])

    def apply_figure_style(self, fig):
        """
        Apply background, grid, and spine styling to all axes in a figure.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure to style.
        """
        fig.patch.set_facecolor('#f8f9fa')  # Light background for the figure

        for ax in fig.axes:
            ax.set_facecolor('#ffffff')  # White background for plots
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#333333')
            ax.spines['bottom'].set_color('#333333')
            ax.grid(color='#dddddd', linestyle='-', linewidth=0.5, alpha=0.5)

    def get_colors(self):
        """
        Retrieve the current theme's color dictionary.

        Returns:
        --------
        dict : Color definitions for current theme.
        """
        return self.colors

    def __str__(self):
        """
        Return a string representation of the theme.
        """
        return f"Theme: {self.theme_name}"
