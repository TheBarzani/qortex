import os
import math
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from merlin import QuantumLayer, MeasurementStrategy
from merlin.builder import CircuitBuilder

# -------------------------
# Config
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "Track2_QML")

NORMALIZED_FEATURES_PATH = os.path.join(
    DATA_DIR, "df_normalized_lags2_from_pacf.csv"
)
RAW_TRAIN_PATH = os.path.join(DATA_DIR, "train.xlsx")

FORECAST_HORIZON_DAYS = 14
TEST_SIZE = 0.2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of features we actually feed into the quantum layer
N_Q_FEATURES = 6  # small, stable quantum core; classical head expands to 224 dims


# -------------------------
# Utility functions
# -------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_val_split_time_series(X, y, dates, test_size=0.2):
    """Simple chronological split (no shuffling)."""
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    dates_train, dates_val = dates[:split_idx], dates[split_idx:]
    return X_train, X_val, y_train, y_val, dates_train, dates_val


def compute_metrics(y_true, y_pred):
    """Flatten everything and compute global RMSE/MAE/R2."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# -------------------------
# Data loading: engineered dataset
# -------------------------

def load_crafted_dataset():
    """
    Load the engineered / normalized feature file and build
    (X_t, y_{t+1}) pairs on the first 224 columns = normalized surface.
    """
    print(f"Loading engineered dataset from: {NORMALIZED_FEATURES_PATH}")
    df = pd.read_csv(NORMALIZED_FEATURES_PATH)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        dates_all = df["Date"].values
        feat_df = df.drop(columns=["Date"])
    else:
        feat_df = df.copy()
        dates_all = None

    # Use first 224 columns as normalized swaption surface
    surface_cols = feat_df.columns[:224]
    print(f"Using {len(surface_cols)} columns as normalized surface.")
    surf = feat_df[surface_cols].values  # (N, 224)

    # Build 1-step ahead pairs: surf_t -> surf_{t+1}
    X_all_full = surf[:-1]   # full 224-dim surface used for forecasting
    y_all = surf[1:]

    if dates_all is not None:
        dates = dates_all[1:]
    else:
        raw = pd.read_excel(RAW_TRAIN_PATH)
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw.sort_values("Date").reset_index(drop=True)
        dates = raw["Date"].values[-len(y_all):]

    print(f"Dataset shape: X={X_all_full.shape}, y={y_all.shape}")
    return X_all_full, y_all, dates


# -------------------------
# Hybrid QML model
# -------------------------

class HybridQMLRegressor(nn.Module):
    """
    Hybrid model:
      - Inputs to quantum core: first N_Q_FEATURES of normalized surface
      - Quantum core: MerLin QuantumLayer over these features
      - Classical head: maps quantum features -> 224-dim surface
    """

    def __init__(
        self,
        n_q_features: int,
        target_dim: int,
        n_photons: int = 3,
        n_modes: int = 6,
        hidden_dim: int = 128,
    ):
        super().__init__()

        n_modes = max(n_modes, n_q_features)

        builder = CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(
            modes=list(range(n_q_features)),  # one mode per encoded feature
            name="input",
            scale=np.pi,
        )
        builder.add_rotations(trainable=True, name="theta")
        builder.add_superpositions(depth=1, trainable=True)

        self.quantum_core = QuantumLayer(
            input_size=n_q_features,  # <-- must match encoded features (N_Q_FEATURES)
            builder=builder,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )

        self.head = nn.Sequential(
            nn.Linear(self.quantum_core.output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x_q):
        """
        x_q: (..., n_q_features) tensor (already sliced from full surface).
        """
        q_out = self.quantum_core(x_q)
        out = self.head(q_out)
        return out


# -------------------------
# Training loop
# -------------------------

def train_hybrid_qml(
    model,
    X_train_q,
    y_train,
    X_val_q,
    y_val,
    n_epochs: int = 30,
    lr: float = 1e-3,
):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train_q, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val_q, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train_t)
        loss = F.mse_loss(y_pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t).cpu().numpy()
            y_val_true = y_val_t.cpu().numpy()
            rmse, mae, r2 = compute_metrics(y_val_true, y_val_pred)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={loss.item():.6f} | "
            f"val_loss={rmse**2:.6f} | "
            f"RMSE={rmse:.6f} | MAE={mae:.6f} | R2={r2:.6f}"
        )

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# -------------------------
# 14-day autoregressive forecast (normalized space)
# -------------------------

def forecast_14_days(model, last_surface_full, last_date, clip_min, clip_max, n_q_features: int):
    """
    Generate a 14-day trajectory in normalized space.
      - At each step, feed the first n_q_features of the current surface
        into the quantum core.
      - Predict a full 224-dim surface.
    """
    model.eval()
    current_surface = last_surface_full.copy()

    dates = []
    surfaces = []

    for step in range(FORECAST_HORIZON_DAYS):
        x_q = current_surface[:n_q_features].reshape(1, -1)
        x_q_t = torch.tensor(x_q, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            next_surface = model(x_q_t).cpu().numpy().reshape(-1)

        # Clip to training range to avoid explosions
        next_surface = np.clip(next_surface, clip_min, clip_max)

        next_date = last_date + timedelta(days=step + 1)
        dates.append(next_date)
        surfaces.append(next_surface.copy())

        current_surface = next_surface

    surfaces = np.stack(surfaces, axis=0)
    df_forecast = pd.DataFrame(
        surfaces,
        columns=[f"TenorSurface_{i}" for i in range(surfaces.shape[1])],
    )
    df_forecast.insert(0, "Date", dates)
    return df_forecast


# -------------------------
# Main
# -------------------------

def main():
    set_seed(SEED)

    # 1) Load engineered dataset (full 224-dim surfaces)
    X_all_full, y_all, dates_all = load_crafted_dataset()

    # Quantum input: first N_Q_FEATURES from surface_t
    X_all_q = X_all_full[:, :N_Q_FEATURES]

    # Time-series split
    X_train_q, X_val_q, y_train, y_val, dates_train, dates_val = train_val_split_time_series(
        X_all_q, y_all, dates_all, test_size=TEST_SIZE
    )

    print(f"Train samples: {len(X_train_q)}  |  Val samples: {len(X_val_q)}")

    n_q_features = X_train_q.shape[1]
    target_dim = y_train.shape[1]  # 224

    # 2) Build hybrid QML model
    model = HybridQMLRegressor(
        n_q_features=n_q_features,
        target_dim=target_dim,
        n_photons=3,
        n_modes=6,
        hidden_dim=128,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Hybrid QML model parameters: {total_params}")

    # 3) Train
    model = train_hybrid_qml(
        model,
        X_train_q,
        y_train,
        X_val_q,
        y_val,
        n_epochs=30,
        lr=1e-3,
    )

    # Final validation metrics
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val_q, dtype=torch.float32, device=DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
        y_val_pred = model(X_val_t).cpu().numpy()
        y_val_true = y_val_t.cpu().numpy()
        rmse, mae, r2 = compute_metrics(y_val_true, y_val_pred)

    print("\nValidation metrics (1-step ahead hybrid QML, engineered data):")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.6f}")

    # 4) 14-day forecast (autoregressive, normalized)
    print("\nGenerating 14-day autoregressive QML forecast...")

    last_surface_full = X_all_full[-1]  # last 224-dim normalized surface
    last_date = pd.to_datetime(dates_all[-1])

    # Use training targets to set clipping range
    clip_min = float(y_train.min())
    clip_max = float(y_train.max())

    df_forecast = forecast_14_days(
        model,
        last_surface_full,
        last_date,
        clip_min=clip_min,
        clip_max=clip_max,
        n_q_features=n_q_features,
    )

    out_path = os.path.join(DATA_DIR, "qml_merlin_14day_forecast_engineered.csv")
    df_forecast.to_csv(out_path, index=False)

    print("\n14-day QML forecast (engineered features) preview:")
    print(df_forecast.head())
    print(f"\nSaved 14-day QML forecast to: {out_path}")


if __name__ == "__main__":
    main()
