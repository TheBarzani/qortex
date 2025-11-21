"""
Data loading functions for swaptions data.
Extracted from DataLoading_Visualization_example.ipynb
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import torch


def load_swaptions_data(data_path: str) -> pd.DataFrame:
    """
    Load swaptions training data from Excel file.
    
    Args:
        data_path: Path to the train.xlsx file
        
    Returns:
        DataFrame with Date as first column, sorted by date
    """
    df = pd.read_excel(data_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Keep Date as first column
    df = df[['Date'] + [c for c in df.columns if c != 'Date']]
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature columns (all columns except 'Date').
    
    Args:
        df: DataFrame with swaptions data
        
    Returns:
        List of feature column names
    """
    return [c for c in df.columns if c != 'Date']


def prepare_dataset_data(df: pd.DataFrame, feature_cols: list = None) -> tuple:
    """
    Prepare X and y data for dataset creation.
    Uses next-day prediction (y is shifted by -1 day).
    
    Args:
        df: DataFrame with swaptions data
        feature_cols: List of feature column names. If None, auto-detected.
        
    Returns:
        Tuple of (X, y) DataFrames, with last row removed (where y would be NaN)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].copy()
    y = df[feature_cols].shift(-1)
    
    # Drop last row where target is NaN
    mask = ~y.isnull().any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    return X, y


def parse_surface_metadata(df: pd.DataFrame) -> tuple:
    """
    Parse surface metadata from column names.
    Column format: "Tenor: X; Maturity: Y"
    
    Args:
        df: DataFrame with swaptions data
        
    Returns:
        Tuple of (unique_tenors, unique_maturities, mapping)
        where mapping is dict: {column_name: (tenor, maturity)}
    """
    tenors, maturities = [], []
    mapping = {}
    
    for col in df.columns:
        if col == 'Date':
            continue
        parts = col.split(';')
        tenor = float(parts[0].split(':')[1].strip())
        maturity = float(parts[1].split(':')[1].strip())
        tenors.append(tenor)
        maturities.append(maturity)
        mapping[col] = (tenor, maturity)
    
    return sorted(set(tenors)), sorted(set(maturities)), mapping


def surface_for_date(df: pd.DataFrame, idx: int, 
                     unique_tenors: list, unique_maturities: list, 
                     mapping: dict) -> np.ndarray:
    """
    Extract surface data for a specific date index.
    
    Args:
        df: DataFrame with swaptions data
        idx: Row index (date index)
        unique_tenors: Sorted list of unique tenor values
        unique_maturities: Sorted list of unique maturity values
        mapping: Dict mapping column names to (tenor, maturity) tuples
        
    Returns:
        2D numpy array of shape (len(unique_tenors), len(unique_maturities))
    """
    row = df.iloc[idx]
    surface = np.full((len(unique_tenors), len(unique_maturities)), np.nan)
    
    for col, (t, m) in mapping.items():
        t_idx = unique_tenors.index(t)
        m_idx = unique_maturities.index(m)
        surface[t_idx, m_idx] = row[col]
    
    return surface



if __name__ == "__main__":
    data_path = "data/Track2_QML/train.xlsx"
    df = load_swaptions_data(data_path)
    print(df.head())