# Simple, model-side feature engineering for swaption surfaces.
# These functions operate on already-clean data coming from data_loading.py
# and are meant to be used by forecasting / modeling scripts.

from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd


def build_feature_matrix(
    X_df: pd.DataFrame,
    dates: pd.Series,
) -> Tuple[pd.DataFrame, int]:
    """
    Build an augmented feature matrix from:
      - the raw surface (all original columns in X_df)
      - simple time features (day_of_week, month, quarter)
      - simple surface stats (mean, std across the surface)

    The first `base_feature_count` columns of the returned DataFrame
    are guaranteed to be the original surface columns in the same order.

    Returns
    -------
    X_feat : pd.DataFrame
        Augmented feature matrix.
    base_feature_count : int
        Number of original surface features (used later to know how many
        entries correspond to the surface itself).
    """
    if "Date" in X_df.columns:
        raise ValueError("X_df should NOT contain a 'Date' column.")

    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)

    X_feat = X_df.copy()
    base_feature_count = X_df.shape[1]

    # Time features
    X_feat["day_of_week"] = dates.dt.dayofweek
    X_feat["month"] = dates.dt.month
    X_feat["quarter"] = dates.dt.quarter

    # Surface summary features
    X_feat["surface_mean"] = X_df.mean(axis=1)
    X_feat["surface_std"] = X_df.std(axis=1)

    return X_feat, base_feature_count


def build_features_for_surface(
    surface_vec: np.ndarray,
    date: pd.Timestamp,
) -> np.ndarray:
    """
    Build a single feature vector for a given surface & date, using the
    SAME logic as in build_feature_matrix:

      [surface_values..., day_of_week, month, quarter, surface_mean, surface_std]

    This is used during recursive forecasting, where we only know the
    current surface and date, but still want the same augmented features
    we trained on.
    """
    surface_vec = np.asarray(surface_vec, dtype=float)
    surface_mean = surface_vec.mean()
    surface_std = surface_vec.std()

    day_of_week = date.dayofweek
    month = date.month
    quarter = (date.month - 1) // 3 + 1  # 1..4

    extra = np.array(
        [day_of_week, month, quarter, surface_mean, surface_std],
        dtype=float,
    )

    return np.concatenate([surface_vec, extra])
