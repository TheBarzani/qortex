"""
Swaption Volatility Prediction - ML Model Guide
================================================

This guide provides comprehensive approaches for loading your swaption data
and training ML models to predict future swaption volatilities.

Data Structure:
- 224 features: Tenor-Maturity combinations (e.g., Tenor 1Y/Maturity 3M)
- 1 time column: Date
- Target: Predict next day's swaption volatilities

RECOMMENDATIONS SUMMARY
========================

1. DATA LOADING:
   ✓ Use pandas.read_excel() for loading
   ✓ Always sort by date
   ✓ Create target by shifting features forward
   ✓ Use TimeSeriesSplit for cross-validation
   ✓ Scale features with StandardScaler

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(file_path, forecast_horizon=1):
    """
    Load swaption data and prepare it for ML modeling.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    forecast_horizon : int
        Number of days ahead to predict (default=1)
    
    Returns:
    --------
    X : DataFrame with features
    y : DataFrame with targets
    df : Original dataframe with dates
    """
    # Load data
    df = pd.read_excel(file_path)
    
    # Ensure Date is datetime and first column
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    if 'Date' in df.columns:
        ordered_cols = ['Date'] + [c for c in df.columns if c != 'Date']
        df = df[ordered_cols]
    
    # Sort by date (crucial for time series)
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} observations from {df['Date'].min()} to {df['Date'].max()}")
    
    # Separate features and date
    feature_cols = [col for col in df.columns if col != 'Date']
    
    # Create target variables (shifted features)
    # We want to predict the NEXT day's values
    targets = df[feature_cols].shift(-forecast_horizon)
    
    # Remove the last row(s) where we don't have targets
    valid_idx = ~targets.isnull().any(axis=1)
    
    X = df.loc[valid_idx, feature_cols].copy()
    y = targets.loc[valid_idx].copy()
    dates = df.loc[valid_idx, 'Date'].copy()
    
    print(f"Created {len(X)} training samples")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    
    return X, y, dates


def create_lagged_features(df, feature_cols, lags=[1, 2, 3, 5, 10]):
    """
    Create lagged features to capture temporal dependencies.
    
    This adds historical values as features, which can be very powerful
    for time series prediction.
    """
    df_lagged = df[feature_cols].copy()
    
    for lag in lags:
        for col in feature_cols:
            df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Drop rows with NaN (from lagging)
    df_lagged = df_lagged.dropna()
    
    return df_lagged


def create_technical_features(X, dates):
    """
    Create additional features from the swaption surface.
    
    These capture the shape and dynamics of the volatility surface.
    """
    X_enhanced = X.copy()
    
    # 1. Time features
    X_enhanced['day_of_week'] = dates.dt.dayofweek
    X_enhanced['month'] = dates.dt.month
    X_enhanced['quarter'] = dates.dt.quarter
    
    # 2. Surface statistics (across all points)
    X_enhanced['surface_mean'] = X.mean(axis=1)
    X_enhanced['surface_std'] = X.std(axis=1)
    X_enhanced['surface_skew'] = X.skew(axis=1)
    X_enhanced['surface_kurt'] = X.kurtosis(axis=1)
    
    # 3. Rolling statistics (looking back)
    window = 5
    X_enhanced['surface_mean_ma5'] = X_enhanced['surface_mean'].rolling(window).mean()
    X_enhanced['surface_volatility'] = X_enhanced['surface_mean'].rolling(window).std()
    
    # 4. Changes from previous day
    for col in X.columns[:10]:  # First 10 key points
        X_enhanced[f'{col}_change'] = X[col].diff()
    
    # Drop NaN from rolling calculations
    X_enhanced = X_enhanced.dropna()
    
    return X_enhanced

class SwaptionDataset(Dataset):
    """PyTorch Dataset for swaption volatility data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.y = torch.FloatTensor(y.values if isinstance(y, pd.DataFrame) else y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(file_path="/Users/cassandrenotton/Documents/projects/mila-hackathon/QFF-Mila-AMF-Quandela/data_swaptions/train.xlsx"):
    """Load swaption data from the given Excel file."""
    print(f"Loading data from {file_path} ...")
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    if 'Date' in df.columns:
        ordered_cols = ['Date'] + [c for c in df.columns if c != 'Date']
        df = df[ordered_cols]
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} observations")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def prepare_data(df, forecast_horizon=1, sequence_length=None):
    """
    Prepare features and targets.
    
    Args:
        forecast_horizon: Days ahead to predict
        sequence_length: If provided, create sequences for LSTM (e.g., 5)
    """
    if 'Date' in df.columns:
        ordered_cols = ['Date'] + [c for c in df.columns if c != 'Date']
        df = df[ordered_cols]
    
    feature_cols = [col for col in df.columns if col != 'Date']
    
    if sequence_length is None:
        # Simple feed-forward approach
        X = df[feature_cols].copy()
        y = df[feature_cols].shift(-forecast_horizon)
        
        valid_mask = ~y.isnull().any(axis=1)
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        dates = df.loc[valid_mask, 'Date'].reset_index(drop=True)
        
    else:
        # Sequence approach for LSTM
        X_list, y_list, dates_list = [], [], []
        
        for i in range(sequence_length, len(df) - forecast_horizon):
            X_list.append(df[feature_cols].iloc[i-sequence_length:i].values)
            y_list.append(df[feature_cols].iloc[i + forecast_horizon].values)
            dates_list.append(df['Date'].iloc[i])
        
        X = np.array(X_list)
        y = np.array(y_list)
        dates = pd.Series(dates_list)
    
    return X, y, dates


def split_data(X, y, dates, test_size=0.2):
    """Split data maintaining temporal order."""
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test


def build_inference_sequences(df, feature_cols, sequence_length):
    """
    Build rolling sequences for inference so that we can generate one prediction
    per row even when the dataset is short (pads the earliest row when needed).
    """
    values = df[feature_cols].values
    sequences = []
    for i in range(len(values)):
        start = max(0, i - sequence_length + 1)
        window = values[start:i + 1]
        if len(window) < sequence_length:
            pad = np.repeat(values[start:start + 1], sequence_length - len(window), axis=0)
            window = np.vstack([pad, window])
        sequences.append(window)
    return np.array(sequences)
