"""
14-day recursive forecast of the swaption surface using
a multi-output PyTorch MLP, with simple extra features:

Inputs (X):
    - full swaption surface (224 points)
    - day_of_week, month, quarter
    - surface_mean, surface_std

Targets (y):
    - next-day full swaption surface (224 points)
"""

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loading import (
    load_swaptions_data,
    prepare_dataset_data,
    get_feature_columns,
)
from features import build_feature_matrix, build_features_for_surface


# -----------------------
# Torch dataset & model
# -----------------------

class SurfaceDataset(Dataset):
    def __init__(self, X, y):
        # X, y are numpy arrays
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------
# Helpers
# -----------------------

def evaluate_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2 on unscaled true/pred arrays."""
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    return rmse, mae, r2


def recursive_forecast_14_days(
    model,
    last_surface_unscaled,
    last_date,
    X_mean,
    X_std,
    y_mean,
    y_std,
    n_days=14,
    device="cpu",
):
    """
    Recursive 14-day forecast using the augmented feature space.

    At each step:
        - build features from (current_surface, current_date)
        - scale with X_mean/X_std
        - predict next-day surface (scaled)
        - unscale using y_mean/y_std
        - move to next day and repeat
    """
    model.eval()
    current_surface = last_surface_unscaled.copy()
    current_date = last_date

    forecasts_unscaled = []

    with torch.no_grad():
        for _ in range(n_days):
            # Build augmented feature vector from current surface & date
            feat_vec_unscaled = build_features_for_surface(current_surface, current_date)

            # Scale input features
            feat_scaled = (feat_vec_unscaled - X_mean) / X_std
            feat_tensor = torch.tensor(
                feat_scaled, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Predict next-day surface (scaled)
            next_scaled = model(feat_tensor).squeeze(0).cpu().numpy()

            # Unscale target (surface)
            next_unscaled = next_scaled * y_std + y_mean
            forecasts_unscaled.append(next_unscaled)

            # Move to next step
            current_surface = next_unscaled
            current_date = current_date + pd.Timedelta(days=1)

    return np.vstack(forecasts_unscaled)


# -----------------------
# Main training & forecast
# -----------------------

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "Track2_QML" / "train.xlsx"

    # 1. Load data
    df = load_swaptions_data(str(data_path))
    feature_cols = get_feature_columns(df)

    # X: today surface, y: next-day surface
    X_df, y_df = prepare_dataset_data(df, feature_cols=feature_cols)

    # Align dates with X/y (prepare_dataset_data drops the last row)
    dates_aligned = df["Date"].iloc[: len(X_df)].reset_index(drop=True)

    # 2. Build augmented feature matrix for inputs
    X_feat_df, base_feature_count = build_feature_matrix(X_df, dates_aligned)

    X_full = X_feat_df.values          # inputs: surface + time + stats
    y_full = y_df.values               # targets: next-day surface only

    n_samples, input_dim = X_full.shape
    output_dim = y_full.shape[1]

    # 3. Scaling (separate for X and y)
    X_mean = X_full.mean(axis=0)
    X_std = X_full.std(axis=0) + 1e-8

    y_mean = y_full.mean(axis=0)
    y_std = y_full.std(axis=0) + 1e-8

    X_scaled = (X_full - X_mean) / X_std
    y_scaled = (y_full - y_mean) / y_std

    # 4. Time-based split (no shuffle)
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

    train_ds = SurfaceDataset(X_train, y_train)
    val_ds = SurfaceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 5. Model, optimizer, loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_sizes=(256, 128)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 6. Training loop
    n_epochs = 60
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # quick validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f}")

    # 7. One-step-ahead metrics on validation set (unscaled)
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_pred_scaled = model(X_val_t).cpu().numpy()

    # unscale
    y_val_true_unscaled = y_val * y_std + y_mean
    y_val_pred_unscaled = y_val_pred_scaled * y_std + y_mean

    rmse, mae, r2 = evaluate_metrics(y_val_true_unscaled, y_val_pred_unscaled)
    print("\nValidation metrics (1-step ahead MLP with extra features):")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.6f}")

    # 8. Recursive 14-day forecast starting from last real surface
    last_surface_unscaled = df[feature_cols].iloc[-1].values  # shape (n_surface_features,)
    last_date = df["Date"].iloc[-1]

    future_surfaces_unscaled = recursive_forecast_14_days(
        model=model,
        last_surface_unscaled=last_surface_unscaled,
        last_date=last_date,
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std,
        n_days=14,
        device=device,
    )

    # 9. Build future dates and DataFrame
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=14, freq="D")

    forecast_df = pd.DataFrame(future_surfaces_unscaled, columns=feature_cols)
    forecast_df.insert(0, "Date", future_dates)

    print("\n14-day MLP forecast (with extra features) preview:")
    print(forecast_df.head())

    # 10. Save to CSV
    out_path = base_dir / "data" / "Track2_QML" / "mlp_14day_forecast_with_features.csv"
    forecast_df.to_csv(out_path, index=False)
    print(f"\nSaved 14-day MLP forecast to: {out_path}")


if __name__ == "__main__":
    main()
