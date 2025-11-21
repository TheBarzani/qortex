"""
Hybrid Quantum–Classical forecast using Quandela MerLin.

- Uses a QuantumLayer as a nonlinear feature extractor on a subset of
  the swaption surface (first n_q_features columns).
- A small classical MLP head maps quantum + classical features to the
  full next-day surface (224 vols).
- Trains 1-step-ahead prediction and reports RMSE / MAE / R2.
- Also generates a 14-day autoregressive forecast from the last
  available date and saves it to CSV.
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MerLin / Quandela
try:
    from merlin import QuantumLayer, MeasurementStrategy
    from merlin.builder import CircuitBuilder
except ImportError as e:
    raise SystemExit(
        "MerLin is not installed. Please install it first, e.g.:\n"
        "  pip install quandela-merlin perceval-quandela\n\n"
        f"Original error: {e}"
    )

# Our own utilities
from data_loading import load_swaptions_data, prepare_dataset_data, get_feature_columns


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class QuantumHybridForecast(nn.Module):
    """
    Hybrid model:
      - QuantumLayer on a subset of features (angle encoding)
      - Concatenate quantum features with full classical features
      - Classical MLP head to predict full next-day surface
    """

    def __init__(
        self,
        n_features: int,
        target_dim: int,
        n_q_features: int = 6,
        latent_n_photons: int = 3,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        assert n_q_features <= n_features, "n_q_features cannot exceed total feature count"

        self.n_features = n_features
        self.target_dim = target_dim
        self.n_q_features = n_q_features

        # Build quantum circuit
        builder = CircuitBuilder(n_modes=n_q_features)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(
            modes=list(range(n_q_features)),  # first n_q_features classical inputs
            name="input",
            scale=np.pi,
        )
        builder.add_rotations(trainable=True, name="theta")
        builder.add_superpositions(depth=1, trainable=True)

        self.quantum = QuantumLayer(
            input_size=n_q_features,
            builder=builder,
            n_photons=latent_n_photons,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
        )

        q_out_dim = self.quantum.output_size
        in_dim_head = n_features + q_out_dim  # classical + quantum features

        layers = []
        prev = in_dim_head
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, target_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_features) scaled classical features
        """
        x_q = x[:, :self.n_q_features]  # (batch, n_q_features)
        q_feats = self.quantum(x_q)     # (batch, q_out_dim)
        x_cat = torch.cat([x, q_feats], dim=1)
        return self.head(x_cat)


# ---------------------------------------------------------------------------
# Training / evaluation utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)

    return total_loss / max(n_samples, 1)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

            preds_list.append(pred.cpu().numpy())
            targets_list.append(yb.cpu().numpy())

    avg_loss = total_loss / max(n_samples, 1)
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds))

    return avg_loss, rmse, mae, r2


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # 1. Load data
    # ----------------------------
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "Track2_QML" / "train.xlsx"

    print(f"Loading data from: {data_path}")
    df = load_swaptions_data(str(data_path))
    feature_cols = get_feature_columns(df)

    X_df, y_df = prepare_dataset_data(df, feature_cols=feature_cols)
    assert X_df.shape == y_df.shape

    n_samples, n_features = X_df.shape
    target_dim = n_features
    print(f"Dataset shape: X={X_df.shape}, y={y_df.shape}")

    # ----------------------------
    # 2. Train / validation split (time ordered)
    # ----------------------------
    split_idx = int(n_samples * 0.8)
    X_train_df = X_df.iloc[:split_idx].reset_index(drop=True)
    y_train_df = y_df.iloc[:split_idx].reset_index(drop=True)
    X_val_df = X_df.iloc[split_idx:].reset_index(drop=True)
    y_val_df = y_df.iloc[split_idx:].reset_index(drop=True)

    print(f"Train samples: {len(X_train_df)}  |  Val samples: {len(X_val_df)}")

    # ----------------------------
    # 3. Scaling (inputs only)
    # ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values.astype(np.float32))
    X_val_scaled = scaler.transform(X_val_df.values.astype(np.float32))

    y_train = y_train_df.values.astype(np.float32)
    y_val = y_val_df.values.astype(np.float32)

    # ----------------------------
    # 4. Build model
    # ----------------------------
    n_q_features = 6  # how many surface points we feed to the quantum layer
    model = QuantumHybridForecast(
        n_features=n_features,
        target_dim=target_dim,
        n_q_features=n_q_features,
        latent_n_photons=3,
        hidden_sizes=(256, 256),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Hybrid QML model parameters: {n_params}")

    # ----------------------------
    # 5. DataLoaders
    # ----------------------------
    batch_size = 64

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_scaled), torch.from_numpy(y_train)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val_scaled), torch.from_numpy(y_val)
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ----------------------------
    # 6. Training loop
    # ----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 30

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, rmse, mae, r2 = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"RMSE={rmse:.6f} | MAE={mae:.6f} | R2={r2:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    val_loss, rmse, mae, r2 = eval_epoch(model, val_loader, device)
    print("\nValidation metrics (1-step ahead hybrid QML):")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.6f}")

    # ----------------------------
    # 7. 14-day autoregressive forecast
    # ----------------------------
    print("\nGenerating 14-day autoregressive QML forecast...")

    # Use the *last* row of X_df as starting context (original scale)
    last_context = X_df.iloc[[-1]].values.astype(np.float32)  # shape (1, n_features)
    last_date = df["Date"].iloc[-1]

    future_rows = []
    future_dates = []

    model.eval()
    with torch.no_grad():
        current_context = last_context.copy()

        for step in range(1, 15):  # 1..14
            # Scale current context for input
            x_scaled = scaler.transform(current_context)  # (1, n_features)
            x_tensor = torch.from_numpy(x_scaled).to(device)

            # Predict next-day surface (original scale)
            y_pred = model(x_tensor).cpu().numpy()[0]  # (target_dim,)

            future_date = last_date + timedelta(days=step)
            future_dates.append(future_date)
            future_rows.append(y_pred)

            # For autoregressive roll-out, treat prediction as next context
            current_context = y_pred.reshape(1, -1).astype(np.float32)

    forecast_array = np.stack(future_rows, axis=0)  # (14, 224)
    forecast_df = pd.DataFrame(forecast_array, columns=feature_cols)
    forecast_df.insert(0, "Date", pd.to_datetime(future_dates))

    out_path = root / "data" / "Track2_QML" / "qml_merlin_14day_forecast.csv"
    forecast_df.to_csv(out_path, index=False)

    print("\n14-day QML forecast preview:")
    print(forecast_df.head())
    print(f"\nSaved 14-day QML forecast to: {out_path}")


if __name__ == "__main__":
    main()
