"""
Data loading functions for swaptions data.
Extracted from DataLoading_Visualization_example.ipynb
"""

from pathlib import Path
import os
import warnings
import pandas as pd
import numpy as np
import torch

# Suppress pandas performance warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


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


# Creating features from the surface
def create_technical_features(X):
    """
    Create additional features from the swaption surface data (X).
    
    Assumes the DataFrame X contains a column named 'date' with datetime objects.
    These features capture the shape and dynamics of the volatility surface.
    """
    X_enhanced = X.copy()
    # print(X_enhanced.columns)
    # Ensure the 'date' column is in datetime format and is present
    if 'Date' not in X_enhanced.columns:
        raise ValueError("DataFrame X must contain a column named 'date'.")
        
    # Convert to datetime objects if they aren't already (important for .dt accessor)
    dates = pd.to_datetime(X_enhanced['Date'])

    # 1. Time features
    # Access the date components directly from the 'date' column
    X_enhanced['day_of_week'] = dates.dt.dayofweek
    X_enhanced['month'] = dates.dt.month
    X_enhanced['quarter'] = dates.dt.quarter
    # print(X_enhanced.columns)
    # print(X_enhanced.head(1))
    # Remove the original 'date' column if it's not needed for the model
    # (Optional, comment out if you need to keep it)
    
    # print(X_enhanced.columns)
    # print(X_enhanced.head(1))
    
    X_enhanced2 = X.copy()
    X_enhanced2 = X_enhanced2.drop(columns=['Date'])
    # print(X_enhanced2.columns)
    # print(X_enhanced2.head(1))
        # 2. Surface statistics (across all points)
    X_enhanced2['surface_mean'] = X_enhanced2.mean(axis=1)
    X_enhanced2['surface_std'] = X_enhanced2.std(axis=1)
    X_enhanced2['surface_skew'] = X_enhanced2.skew(axis=1)
    X_enhanced2['surface_kurt'] = X_enhanced2.kurtosis(axis=1)
    
    # 3. Rolling statistics (looking back)
    window = 5
    X_enhanced2['surface_mean_ma5'] = X_enhanced2['surface_mean'].rolling(window).mean()
    X_enhanced2['surface_volatility'] = X_enhanced2['surface_mean'].rolling(window).std()
    
    # 4. Changes from previous day
    # for col in X.columns[:10]:  # First 10 key points
    #     X_enhanced2[f'{col}_change'] = X[col].diff()

    X_result = pd.concat([X_enhanced, X_enhanced2], axis=1)
    return X_result


def create_lagged_features(df, feature_cols, lags=[1, 2, 3, 5, 10]):
    """
    Create lagged features to capture temporal dependencies.
    
    This adds historical values as features, which can be very powerful
    for time series prediction.
    """
    df_lagged = df[feature_cols].copy()
    
    for lag in lags:
        for col in feature_cols:
            # Ensure we get a Series, not a DataFrame
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame, skipping...")
                continue
            
            # Get the column as a Series explicitly
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                # If it's a DataFrame (shouldn't happen, but handle it), take first column
                col_data = col_data.iloc[:, 0]
            
            # Shift and assign
            df_lagged[f'{col}_lag{lag}'] = col_data.shift(lag)
    
    # Drop rows with NaN (from lagging)
    df_lagged = df_lagged.dropna()
    
    return df_lagged


# Seasonal decomposition feature creation (additive decomposition)
def create_seasonal_decomposition_features(df, feature_cols, period=7, model='additive'):
    """
    Adds seasonal decomposition statistics (trend, seasonal, resid) for each feature column.
    
    Args:
        df: DataFrame, must be sorted in time.
        feature_cols: List of columns to decompose.
        period: Period for decomposition (7 ~ weekly for daily fin data).
        model: 'additive' or 'multiplicative'
    Returns:
        DataFrame with new columns for each component per feature col.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    df_result = df.copy()
    df_result = df_result.dropna()
    
    for col in feature_cols:
        if col not in df_result.columns:
            print(f"Warning: Column '{col}' not found in DataFrame, skipping...")
            continue
        
        try:
            # Get the series for decomposition
            series = df_result[col].copy()
            
            # Ensure we have enough data points for decomposition
            if len(series) < 2 * period:
                print(f"Warning: Not enough data points for column '{col}' (need at least {2*period}, have {len(series)}), skipping...")
                continue
            
            # Perform decomposition - pass Series directly
            decomposition = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
            
            # Extract components - ensure they are Series/arrays, not strings
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            resid = decomposition.resid
            
            # Convert to numpy arrays to avoid index issues, then create Series with correct index
            trend_values = np.array(trend) if hasattr(trend, '__array__') else trend
            seasonal_values = np.array(seasonal) if hasattr(seasonal, '__array__') else seasonal
            resid_values = np.array(resid) if hasattr(resid, '__array__') else resid
            
            # Create Series with the DataFrame's index
            df_result[f'{col}_trend'] = pd.Series(trend_values, index=df_result.index, dtype=float)
            df_result[f'{col}_seasonal'] = pd.Series(seasonal_values, index=df_result.index, dtype=float)
            df_result[f'{col}_resid'] = pd.Series(resid_values, index=df_result.index, dtype=float)
            
        except Exception as e:
            print(f"Error decomposing column '{col}': {e}")
            import traceback
            traceback.print_exc()
            continue

    return df_result




def normalize_features(df, cols):
    """
    Normalize features in the DataFrame.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def prepare_dataset_data_for_forecasting():
    """
    Prepare dataset data for forecasting.
    """
    data_dir_path = "../data/Track2_QML/"
    data_file_name = "train.xlsx"
    data_path = os.path.join(data_dir_path, data_file_name)
    df = load_swaptions_data(str(data_path))
    # print(df.head())

    #FEATURES CREATION
    # Apply feature creation functions on the loaded dataframe
    feature_cols = get_feature_columns(df)

    # Create technical features (this will produce engineered features from the raw inputs)
    df_features = create_technical_features(df)
    print(df_features.shape, "features shape")
    # Optionally, create lagged features if desired:
    lag_input_cols = df_features.columns.to_list()[1:225] #only the tenor and maturity columns
    df_features_lagged = create_lagged_features(df_features, lag_input_cols, lags=[1, 2])
    # Make a new copy containing the enhanced dataset with features

    df_all_features = pd.concat([df_features, df_features_lagged], axis=1)
    print(df_all_features.shape, "all features shape")

    # print(df_all_features.columns.tolist()[224: 250])
    # NOT WORKING
    # df_seasonal_features = create_seasonal_decomposition_features(df_all_features, lag_input_cols, period=7, model='additive')
    # print(df_seasonal_features.shape, "seasonal features shape")
    # print('_trend' in df_seasonal_features.columns.tolist())
    # print('_seasonal' in df_seasonal_features.columns.tolist())
    # print('_resid' in df_seasonal_features.columns.tolist())



    # # #NORMALIZATION
    df_normalized = normalize_features(df_all_features, ["day_of_week", "month", "quarter"])
    print(df_normalized.shape, "normalized shape")
    print(df_normalized[["day_of_week", "month", "quarter"]].head(1))

    # print("null check", any(df_normalized.isnull().sum()), df_normalized.shape)
    df_normalized = df_normalized.dropna()
    print("null check after dropna", any(df_normalized.isnull().sum()), df_normalized.shape)

    df_normalized.to_csv(str(data_dir_path + "df_normalized_lags2_from_pacf.csv"), index=False)
    print(f"Saved normalized data to {data_dir_path + 'df_normalized_lags2_from_pacf.csv'}")

if __name__ == "__main__":

    prepare_dataset_data_for_forecasting()
