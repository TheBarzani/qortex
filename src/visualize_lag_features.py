"""
Visualization code for lag features in swaptions data.

This module creates visualizations to analyze lag features and their relationship
with original features, using technical and lag features created for SwaptionDataset.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import statsmodels for ACF/PACF
try:
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ACF/PACF functions will not work.")

# Suppress warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Import data loading functions
import sys
from pathlib import Path

# Add src directory to path if needed
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data_loading import (
    load_swaptions_data,
    get_feature_columns,
    create_technical_features,
    create_lagged_features,
    normalize_features
)

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


def visualize_lag_feature_comparison(df, original_col, lag_col, date_col='Date', 
                                     save_path=None, figsize=(14, 6)):
    """
    Visualize comparison between original feature and its lag feature.
    
    Args:
        df: DataFrame containing both original and lag features
        original_col: Name of the original feature column
        lag_col: Name of the lag feature column
        date_col: Name of the date column
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Overlay original and lag feature
    axes[0].plot(df[date_col], df[original_col], label=f'Original: {original_col}', 
                 alpha=0.7, linewidth=1.5)
    axes[0].plot(df[date_col], df[lag_col], label=f'Lag: {lag_col}', 
                 alpha=0.7, linewidth=1.5, linestyle='--')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Original vs Lag Feature: {original_col}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Difference between original and lag
    if len(df) == len(df.dropna(subset=[original_col, lag_col])):
        diff = df[original_col] - df[lag_col]
        axes[1].plot(df[date_col], diff, color='red', alpha=0.7, linewidth=1.5)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        axes[1].set_ylabel('Difference')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Difference (Original - Lag)')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_lag_correlation(df, original_col, lag_cols, save_path=None, figsize=(10, 6)):
    """
    Visualize correlation between original feature and multiple lag features.
    
    Args:
        df: DataFrame containing features
        original_col: Name of the original feature column
        lag_cols: List of lag feature column names
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    # Calculate correlations
    correlations = []
    lag_values = []
    
    for lag_col in lag_cols:
        if lag_col in df.columns:
            # Drop NaN for correlation calculation
            valid_data = df[[original_col, lag_col]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[original_col].corr(valid_data[lag_col])
                correlations.append(corr)
                # Extract lag value from column name (e.g., "col_lag1" -> 1)
                try:
                    lag_val = int(lag_col.split('lag')[-1])
                    lag_values.append(lag_val)
                except:
                    lag_values.append(len(lag_values) + 1)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(correlations)), correlations, alpha=0.7, color='steelblue')
    ax.set_xlabel('Lag Period')
    ax.set_ylabel('Correlation with Original')
    ax.set_title(f'Correlation: {original_col} vs Lag Features')
    ax.set_xticks(range(len(lag_values)))
    ax.set_xticklabels([f'Lag {lv}' for lv in lag_values])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved correlation plot to {save_path}")
    
    return fig


def visualize_lag_scatter(df, original_col, lag_col, date_col='Date', 
                          save_path=None, figsize=(10, 6)):
    """
    Create scatter plot showing relationship between original and lag feature.
    
    Args:
        df: DataFrame containing features
        original_col: Name of the original feature column
        lag_col: Name of the lag feature column
        date_col: Name of the date column (for coloring)
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    # Prepare data
    plot_data = df[[original_col, lag_col, date_col]].dropna()
    
    if len(plot_data) == 0:
        print(f"Warning: No valid data for {original_col} and {lag_col}")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with color based on date
    scatter = ax.scatter(plot_data[lag_col], plot_data[original_col], 
                        c=range(len(plot_data)), cmap='viridis', 
                        alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (y=x) for reference
    min_val = min(plot_data[lag_col].min(), plot_data[original_col].min())
    max_val = max(plot_data[lag_col].max(), plot_data[original_col].max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', alpha=0.5, linewidth=2, label='y=x')
    
    # Calculate and display correlation
    corr = plot_data[original_col].corr(plot_data[lag_col])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(f'Lag Feature: {lag_col}')
    ax.set_ylabel(f'Original Feature: {original_col}')
    ax.set_title(f'Scatter Plot: {original_col} vs {lag_col}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Time Index')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved scatter plot to {save_path}")
    
    return fig


def visualize_multiple_lag_features(df, original_col, lag_cols, date_col='Date',
                                    save_path=None, figsize=(14, 10)):
    """
    Visualize multiple lag features together with the original feature.
    
    Args:
        df: DataFrame containing features
        original_col: Name of the original feature column
        lag_cols: List of lag feature column names
        date_col: Name of the date column
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    n_lags = len(lag_cols)
    fig, axes = plt.subplots(n_lags + 1, 1, figsize=figsize, sharex=True)
    
    # Plot original feature
    axes[0].plot(df[date_col], df[original_col], label=original_col, 
                linewidth=2, color='black')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Original Feature: {original_col}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot each lag feature
    colors = plt.cm.tab10(np.linspace(0, 1, n_lags))
    for i, (lag_col, color) in enumerate(zip(lag_cols, colors)):
        if lag_col in df.columns:
            axes[i+1].plot(df[date_col], df[lag_col], label=lag_col, 
                          linewidth=1.5, color=color, alpha=0.8)
            axes[i+1].set_ylabel('Value')
            axes[i+1].set_title(f'Lag Feature: {lag_col}')
            axes[i+1].legend()
            axes[i+1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved multi-lag visualization to {save_path}")
    
    return fig


def visualize_lag_autocorrelation(df, original_col, max_lag=10, 
                                   save_path=None, figsize=(10, 6)):
    """
    Visualize autocorrelation of a feature with its lagged versions.
    
    Args:
        df: DataFrame containing features
        original_col: Name of the original feature column
        max_lag: Maximum lag to compute autocorrelation for
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    # Calculate autocorrelation
    series = df[original_col].dropna()
    if len(series) < max_lag:
        max_lag = len(series) - 1
    
    autocorrs = []
    lags = list(range(1, max_lag + 1))
    
    for lag in lags:
        if lag < len(series):
            shifted = series.shift(lag)
            valid = pd.concat([series, shifted], axis=1).dropna()
            if len(valid) > 0:
                corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                autocorrs.append(corr)
            else:
                autocorrs.append(np.nan)
        else:
            autocorrs.append(np.nan)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(lags, autocorrs, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Autocorrelation Function: {original_col}')
    ax.grid(True, alpha=0.3)
    
    # Add confidence intervals (approximate)
    ax.axhline(y=1.96/np.sqrt(len(series)), color='red', linestyle='--', 
              alpha=0.5, label='95% Confidence')
    ax.axhline(y=-1.96/np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved autocorrelation plot to {save_path}")
    
    return fig


def plot_acf_pacf(series, lags=None, alpha=0.05, figsize=(14, 6), 
                  save_path=None, title_prefix=""):
    """
    Plot ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)
    for a raw time series.
    
    Args:
        series: pandas Series or array-like, the time series data
        lags: Number of lags to include (default: min(40, len(series)//2 - 1))
        alpha: Significance level for confidence intervals (default: 0.05)
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        title_prefix: Optional prefix for the plot title
        
    Returns:
        matplotlib figure object
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for ACF/PACF plots. Install with: pip install statsmodels")
    
    # Convert to Series if needed
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # Remove NaN values
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        raise ValueError(f"Series too short for ACF/PACF analysis. Need at least 10 points, got {len(series_clean)}")
    
    # Determine number of lags
    if lags is None:
        lags = min(40, len(series_clean) // 2 - 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ACF
    plot_acf(series_clean, lags=lags, alpha=alpha, ax=axes[0], 
            title=f'{title_prefix}ACF - Autocorrelation Function')
    axes[0].grid(True, alpha=0.3)
    
    # Plot PACF
    plot_pacf(series_clean, lags=lags, alpha=alpha, ax=axes[1], 
             title=f'{title_prefix}PACF - Partial Autocorrelation Function')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved ACF/PACF plot to {save_path}")
    
    return fig


def plot_acf_pacf_from_dataframe(df, column_name, lags=None, alpha=0.05, 
                                 figsize=(14, 6), save_path=None):
    """
    Plot ACF and PACF for a specific column from a DataFrame.
    
    Args:
        df: DataFrame containing the time series
        column_name: Name of the column to plot
        lags: Number of lags to include
        alpha: Significance level for confidence intervals
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    series = df[column_name]
    title_prefix = f"{column_name}\n"
    
    return plot_acf_pacf(series, lags=lags, alpha=alpha, figsize=figsize,
                        save_path=save_path, title_prefix=title_prefix)


def plot_multiple_acf_pacf(df, columns, lags=None, alpha=0.05, 
                          figsize=(16, 12), save_path=None):
    """
    Plot ACF and PACF for multiple columns in a grid layout.
    
    Args:
        df: DataFrame containing the time series
        columns: List of column names to plot
        lags: Number of lags to include
        alpha: Significance level for confidence intervals
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for ACF/PACF plots. Install with: pip install statsmodels")
    
    n_cols = len(columns)
    if n_cols == 0:
        raise ValueError("No columns provided")
    
    # Create grid: 2 rows (ACF, PACF) x n_cols columns
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    
    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, col_name in enumerate(columns):
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found, skipping...")
            continue
        
        series = df[col_name].dropna()
        
        if len(series) < 10:
            print(f"Warning: Column '{col_name}' has insufficient data, skipping...")
            continue
        
        # Determine lags if not provided
        if lags is None:
            lags_use = min(40, len(series) // 2 - 1)
        else:
            lags_use = lags
        
        # Plot ACF (top row)
        try:
            plot_acf(series, lags=lags_use, alpha=alpha, ax=axes[0, idx],
                    title=f'ACF: {col_name[:30]}')
            axes[0, idx].grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error plotting ACF for {col_name}: {e}")
        
        # Plot PACF (bottom row)
        try:
            plot_pacf(series, lags=lags_use, alpha=alpha, ax=axes[1, idx],
                     title=f'PACF: {col_name[:30]}')
            axes[1, idx].grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error plotting PACF for {col_name}: {e}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved multiple ACF/PACF plots to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    data_path = "../data/Track2_QML/df_normalized_lags4.csv"
    output_dir = f"../plots/{data_path.split('/')[-1].split('.')[0].split('_')[-1]}/"
    os.makedirs(output_dir, exist_ok=True)
    df_normalized = pd.read_csv(data_path)

    lags = [1, 3, 5, 7]
    

    #MULTIPLE LAG FEATURES VISUALIZATION
    original_col = "Tenor : 4; Maturity : 1"
    lag_cols = [f"Tenor : 4; Maturity : 1_lag{i}" for i in lags]
    visualize_multiple_lag_features(df_normalized, original_col, lag_cols, date_col="Date", save_path=output_dir + "multiple_lag_features.png")

    #AUTOCORRELATION VISUALIZATION
    visualize_lag_autocorrelation(df_normalized, original_col, max_lag=10, save_path=output_dir + "autocorrelation.png")
    
    # ACF/PACF PLOTS FOR RAW TIME SERIES
    if STATSMODELS_AVAILABLE:
        # Plot ACF/PACF for the original column
        # plot_acf_pacf_from_dataframe(
        #     df_normalized, 
        #     original_col, 
        #     lags=40, 
        #     save_path=output_dir + "acf_pacf.png"
        # )
        
        # Example: Plot ACF/PACF for multiple columns
        sample_cols = df_normalized.columns.tolist()[10:14]
        plot_multiple_acf_pacf(
            df_normalized,
            sample_cols,  # Limit to 3 columns for readability
            lags=30,
            save_path=output_dir + "multiple_acf_pacf.png"
        )
    else:
        print("statsmodels not available. Skipping ACF/PACF plots.")

