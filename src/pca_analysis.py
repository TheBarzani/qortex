"""
PCA (Principal Component Analysis) analysis on prepared swaptions dataset.

This module applies PCA from sklearn to reduce dimensionality and analyze
the principal components of the feature-engineered dataset.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


def load_prepared_data(data_path):
    """
    Load the prepared dataset (normalized with features).
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with the prepared data
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert Date column to datetime if it exists
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        except:
            pass
    
    print(f"Loaded dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    return df


def prepare_features_for_pca(df, exclude_cols=None):
    """
    Prepare features for PCA by excluding non-numeric and date columns.
    
    Args:
        df: DataFrame with features
        exclude_cols: List of column names to exclude (default: ['Date'])
        
    Returns:
        Tuple of (feature_matrix, feature_names, excluded_data)
    """
    if exclude_cols is None:
        exclude_cols = ['Date']
    
    # Get all columns to exclude
    cols_to_exclude = [col for col in exclude_cols if col in df.columns]
    
    # Separate features and metadata
    feature_cols = [col for col in df.columns if col not in cols_to_exclude]
    excluded_data = df[cols_to_exclude].copy() if cols_to_exclude else None
    
    # Extract feature matrix
    X = df[feature_cols].copy()
    
    # Check for non-numeric columns and exclude them
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    
    if non_numeric_cols:
        print(f"Warning: Excluding non-numeric columns: {non_numeric_cols}")
        X = X[numeric_cols]
        feature_cols = numeric_cols
    
    # Handle missing values
    if X.isnull().any().any():
        print("Warning: Found missing values. Filling with column means.")
        X = X.fillna(X.mean())
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    return X.values, feature_cols, excluded_data


def apply_pca(X, n_components=None, standardize=True):
    """
    Apply PCA to the feature matrix.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_components: Number of components to keep (None = all, or int/float)
        standardize: Whether to standardize features before PCA
        
    Returns:
        Tuple of (pca_model, X_transformed, scaler)
    """
    # Standardize features
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)
    
    print(f"\nPCA Results:")
    print(f"  Original dimensions: {X.shape}")
    print(f"  Transformed dimensions: {X_transformed.shape}")
    print(f"  Number of components: {pca.n_components_}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_[:10].sum():.4f} (first 10 components)")
    print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return pca, X_transformed, scaler


def plot_explained_variance(pca, save_path=None, figsize=(12, 5)):
    """
    Plot explained variance and cumulative explained variance.
    
    Args:
        pca: Fitted PCA model
        save_path: Optional path to save the figure
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)
    
    # Plot 1: Individual explained variance
    axes[0].bar(range(1, min(21, n_components + 1)), 
                explained_var[:min(20, n_components)], 
                alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Explained Variance by Component')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Cumulative explained variance
    axes[1].plot(range(1, n_components + 1), cumsum_var, 
                marker='o', linewidth=2, markersize=4, color='darkgreen')
    axes[1].axhline(y=0.95, color='red', linestyle='--', 
                   label='95% variance', alpha=0.7)
    axes[1].axhline(y=0.90, color='orange', linestyle='--', 
                   label='90% variance', alpha=0.7)
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Find number of components for 95% variance
    n_95 = np.argmax(cumsum_var >= 0.95) + 1
    n_90 = np.argmax(cumsum_var >= 0.90) + 1
    
    axes[1].axvline(x=n_95, color='red', linestyle=':', alpha=0.5)
    axes[1].axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
    
    print(f"\nVariance Analysis:")
    print(f"  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved explained variance plot to {save_path}")
    
    return fig


def plot_component_loadings(pca, feature_names, n_components=5, 
                           save_path=None, figsize=(14, 10)):
    """
    Plot the loadings (weights) of the first few principal components.
    
    Args:
        pca: Fitted PCA model
        feature_names: List of feature names
        n_components: Number of components to visualize
        save_path: Optional path to save the figure
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure object
    """
    n_components = min(n_components, pca.n_components_)
    
    # Get component loadings
    components = pca.components_[:n_components]
    
    # Create subplots
    fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True)
    
    if n_components == 1:
        axes = [axes]
    
    for i in range(n_components):
        ax = axes[i]
        loadings = components[i]
        
        # Get top 20 features by absolute loading
        top_indices = np.argsort(np.abs(loadings))[-20:][::-1]
        top_loadings = loadings[top_indices]
        top_names = [feature_names[idx] if idx < len(feature_names) 
                    else f"Feature_{idx}" for idx in top_indices]
        
        # Create bar plot
        colors = ['red' if x < 0 else 'blue' for x in top_loadings]
        ax.barh(range(len(top_names)), top_loadings, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel('Loading')
        ax.set_title(f'PC{i+1} Loadings (Top 20 Features)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved component loadings plot to {save_path}")
    
    return fig


def plot_pca_components_2d(X_transformed, pca, excluded_data=None, 
                           component_indices=(0, 1), save_path=None, 
                           figsize=(10, 8)):
    """
    Plot first two principal components in 2D.
    
    Args:
        X_transformed: Transformed data from PCA
        pca: Fitted PCA model
        excluded_data: Optional DataFrame with metadata (e.g., Date)
        component_indices: Tuple of (component1, component2) indices
        save_path: Optional path to save the figure
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    pc1_idx, pc2_idx = component_indices
    pc1 = X_transformed[:, pc1_idx]
    pc2 = X_transformed[:, pc2_idx]
    
    # Color by time if Date column exists
    if excluded_data is not None and 'Date' in excluded_data.columns:
        dates = excluded_data['Date']
        scatter = ax.scatter(pc1, pc2, c=range(len(pc1)), 
                           cmap='viridis', alpha=0.6, s=30, 
                           edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, ax=ax, label='Time Index')
    else:
        ax.scatter(pc1, pc2, alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    
    ax.set_xlabel(f'PC{pc1_idx+1} ({pca.explained_variance_ratio_[pc1_idx]:.2%} variance)')
    ax.set_ylabel(f'PC{pc2_idx+1} ({pca.explained_variance_ratio_[pc2_idx]:.2%} variance)')
    ax.set_title(f'First Two Principal Components')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved 2D component plot to {save_path}")
    
    return fig


def get_pca_components_as_features(X_transformed, excluded_data=None, 
                                   n_components=None, component_names=None):
    """
    Convert PCA transformed data into a DataFrame with component features.
    
    Args:
        X_transformed: Transformed data from PCA (n_samples, n_components)
        excluded_data: Optional DataFrame with metadata (e.g., Date)
        n_components: Number of components to include (None = all)
        component_names: Optional list of component names (default: PC1, PC2, ...)
        
    Returns:
        DataFrame with PCA components as features
    """
    # Determine number of components to use
    if n_components is None:
        n_components = X_transformed.shape[1]
    else:
        n_components = min(n_components, X_transformed.shape[1])
    
    # Create component names
    if component_names is None:
        component_names = [f'PC{i+1}' for i in range(n_components)]
    
    # Create DataFrame with PCA components
    pca_df = pd.DataFrame(
        X_transformed[:, :n_components],
        columns=component_names[:n_components]
    )
    
    # Add back excluded columns if they exist
    if excluded_data is not None:
        pca_df = pd.concat([excluded_data.reset_index(drop=True), pca_df], axis=1)
    
    return pca_df


def save_pca_results(pca, X_transformed, feature_names, excluded_data, 
                    output_dir, n_components_to_save=None):
    """
    Save PCA results to files.
    
    Args:
        pca: Fitted PCA model
        X_transformed: Transformed data
        feature_names: List of feature names
        excluded_data: Optional DataFrame with metadata
        output_dir: Directory to save results
        n_components_to_save: Number of components to save (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save transformed data using helper function
    X_df = get_pca_components_as_features(
        X_transformed,
        excluded_data=excluded_data,
        n_components=n_components_to_save
    )
    
    output_path = os.path.join(output_dir, 'pca_transformed_data.csv')
    X_df.to_csv(output_path, index=False)
    print(f"Saved transformed data to {output_path}")
    
    # Save component loadings
    components_df = pd.DataFrame(
        pca.components_[:n_components_to_save].T,
        columns=[f'PC{i+1}' for i in range(n_components_to_save)],
        index=feature_names[:len(pca.components_[0])]
    )
    components_path = os.path.join(output_dir, 'pca_component_loadings.csv')
    components_df.to_csv(components_path)
    print(f"Saved component loadings to {components_path}")
    
    # Save explained variance
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(pca.n_components_)],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance_Ratio': np.cumsum(pca.explained_variance_ratio_),
        'Explained_Variance': pca.explained_variance_
    })
    variance_path = os.path.join(output_dir, 'pca_explained_variance.csv')
    variance_df.to_csv(variance_path, index=False)
    print(f"Saved explained variance to {variance_path}")


def run_pca_analysis(data_path, n_components=None, standardize=True, 
                    output_dir='plots/pca_analysis', exclude_cols=None):
    """
    Run complete PCA analysis pipeline.
    
    Args:
        data_path: Path to prepared dataset CSV
        n_components: Number of PCA components (None = all)
        standardize: Whether to standardize before PCA
        output_dir: Directory to save results and plots
        exclude_cols: Columns to exclude from PCA
        
    Returns:
        Tuple of (pca_model, X_transformed, feature_names, excluded_data)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    df = load_prepared_data(data_path)
    
    # 2. Prepare features
    X, feature_names, excluded_data = prepare_features_for_pca(df, exclude_cols)
    
    # 3. Apply PCA
    pca, X_transformed, scaler = apply_pca(X, n_components=n_components, 
                                          standardize=standardize)
    
    # 4. Create visualizations
    print("\nCreating visualizations...")
    plot_explained_variance(pca, 
                           save_path=os.path.join(output_dir, 'explained_variance.png'))
    plt.close()
    
    plot_component_loadings(pca, feature_names, n_components=5,
                          save_path=os.path.join(output_dir, 'component_loadings.png'))
    plt.close()
    
    plot_pca_components_2d(X_transformed, pca, excluded_data,
                          save_path=os.path.join(output_dir, 'pca_2d_components.png'))
    plt.close()
    
    # 5. Save results
    print("\nSaving results...")
    save_pca_results(pca, X_transformed, feature_names, excluded_data, output_dir)
    
    print(f"\n✓ PCA analysis complete! Results saved to {output_dir}/")
    
    return pca, X_transformed, feature_names, excluded_data


if __name__ == "__main__":
    # Example usage
    data_path = "../data/Track2_QML/df_normalized_lags2_from_pacf.csv"
    output_dir = "../plots/pca_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run PCA analysis
    pca, X_transformed, feature_names, excluded_data = run_pca_analysis(
        data_path=data_path,
        n_components=None,  # Use all components
        standardize=True,
        output_dir=output_dir,
        exclude_cols=['Date']  # Exclude Date column from PCA
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("PCA Analysis Summary")
    print(f"{'='*60}")
    print(f"Original features: {len(feature_names)}")
    print(f"PCA components: {pca.n_components_}")
    print(f"Explained variance (first 10): {pca.explained_variance_ratio_[:10].sum():.4f}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Find optimal number of components
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.argmax(cumsum >= 0.90) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    n_99 = np.argmax(cumsum >= 0.99) + 1
    
    print(f"\nRecommended number of components:")
    print(f"  For 90% variance: {n_90} components")
    print(f"  For 95% variance: {n_95} components")
    print(f"  For 99% variance: {n_99} components")
    
    # Get PCA components as features DataFrame
    print(f"\n{'='*60}")
    print("Extracting PCA Components as Features")
    print(f"{'='*60}")
    
    # Option 1: Get all components
    pca_features_all = get_pca_components_as_features(
        X_transformed, 
        excluded_data=excluded_data
    )
    print(f"\nAll PCA components shape: {pca_features_all.shape}")
    print(f"Columns: {list(pca_features_all.columns[:10])}... (showing first 10)")
    
    # Option 2: Get top N components (e.g., for 95% variance)
    pca_features_95 = get_pca_components_as_features(
        X_transformed,
        excluded_data=excluded_data,
        n_components=n_95  # Use components that explain 95% variance
    )
    print(f"\nTop {n_95} PCA components (95% variance) shape: {pca_features_95.shape}")
    print(f"Columns: {list(pca_features_95.columns)}")
    
    # Option 3: Get top N components (e.g., for 90% variance)
    # pca_features_90 = get_pca_components_as_features(
    #     X_transformed,
    #     excluded_data=excluded_data,
    #     n_components=n_90  # Use components that explain 90% variance
    # )
    # print(f"\nTop {n_90} PCA components (90% variance) shape: {pca_features_90.shape}")
    
    # # Save PCA features to CSV
    # pca_features_path = os.path.join(output_dir, 'pca_features_95_variance.csv')
    # pca_features_95.to_csv(pca_features_path, index=False)
    # print(f"\n✓ Saved PCA features (95% variance) to {pca_features_path}")
    # print(f"  You can now use this file as input features for your models!")
    
    # # Example: Show how to use PCA components
    # print(f"\n{'='*60}")
    # print("Usage Example:")
    # print(f"{'='*60}")
    # print("# To use PCA components as features in your model:")
    # print("# 1. Load the saved CSV:")
    # print(f"#    df_pca = pd.read_csv('{pca_features_path}')")
    # print("# 2. Extract only the PCA component columns (exclude Date if needed):")
    # print("#    pca_cols = [col for col in df_pca.columns if col.startswith('PC')]")
    # print("#    X_pca = df_pca[pca_cols].values")
    # print("# 3. Use X_pca as your feature matrix for training models")
    

