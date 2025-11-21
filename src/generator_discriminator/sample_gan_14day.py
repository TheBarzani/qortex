"""
sample_gan_14day.py

Use the trained classical Generator (GAN) to generate a 14-day
trajectory of swaption surfaces, starting from the last real surface
in train.xlsx.

This assumes you have already run train_gan.py and that
data/Track2_QML/generator_classical.pt exists.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------
# Make sure we can import from src/ and src/generator_discriminator
# ---------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent          # .../src/generator_discriminator
SRC_DIR = CURRENT_DIR.parent                           # .../src

for p in (CURRENT_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from data_loading import load_swaptions_data, get_feature_columns  # type: ignore
from models import Generator  # type: ignore
from utils import get_device  # type: ignore


# ---------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------


def generate_14day_trajectory(
    gen: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    feature_cols: list,
    last_surface_unscaled: np.ndarray,
    last_date: pd.Timestamp,
    latent_dim: int,
    n_days: int = 14,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """
    Recursively generate a 14-day swaption surface trajectory using the
    trained Generator.

    Args
    ----
    gen : trained Generator model
    mean, std : np.ndarray
        Scaling parameters used during training.
    feature_cols : list
        Names of the surface features (must match training).
    last_surface_unscaled : np.ndarray
        Last real surface from the dataset (unscaled).
    last_date : pd.Timestamp
        Date of the last real surface.
    latent_dim : int
        Dimension of the noise vector used by the generator.
    n_days : int
        Number of future days to generate (default 14).
    device : torch.device or str
        Device where the generator lives.

    Returns
    -------
    forecast_df : pd.DataFrame
        DataFrame with columns: Date + feature_cols, length = n_days.
    """
    gen.eval()

    current_surface_unscaled = last_surface_unscaled.copy()
    current_date = last_date

    surfaces = []
    dates = []

    with torch.no_grad():
        for _ in range(n_days):
            # Scale context
            context_scaled = (current_surface_unscaled - mean) / std
            context_tensor = torch.tensor(
                context_scaled, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Sample noise
            noise = torch.randn(1, latent_dim, device=device)

            # Generate next-day surface (scaled)
            next_scaled = gen(context_tensor, noise).squeeze(0).cpu().numpy()

            # Unscale
            next_unscaled = next_scaled * std + mean

            # Advance date and store
            current_date = current_date + pd.Timedelta(days=1)
            dates.append(current_date)
            surfaces.append(next_unscaled)

            # Feed back as next context
            current_surface_unscaled = next_unscaled

    forecast_df = pd.DataFrame(surfaces, columns=feature_cols)
    forecast_df.insert(0, "Date", dates)
    return forecast_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Paths
    base_dir = SRC_DIR.parent  # go from src/ to repo root
    data_path = base_dir / "data" / "Track2_QML" / "train.xlsx"
    ckpt_path = base_dir / "data" / "Track2_QML" / "generator_classical.pt"

    # Load checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run train_gan.py first to train and save the generator."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    context_dim = checkpoint["context_dim"]
    target_dim = checkpoint["target_dim"]
    latent_dim = checkpoint["latent_dim"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    feature_cols_ckpt = checkpoint["feature_cols"]

    mean = np.asarray(mean)
    std = np.asarray(std)

    print(f"Loaded checkpoint from: {ckpt_path}")
    print(f"  context_dim = {context_dim}")
    print(f"  target_dim  = {target_dim}")
    print(f"  latent_dim  = {latent_dim}")
    print(f"  n_features  = {len(feature_cols_ckpt)}")

    # Rebuild generator architecture (must match train_gan.py settings)
    gen = Generator(
        context_dim=context_dim,
        target_dim=target_dim,
        latent_dim=latent_dim,
        hidden_sizes=(128, 128),  # <- must match training config
    ).to(device)
    gen.load_state_dict(checkpoint["generator_state_dict"])

    # Load real data to get the last real surface + date
    df = load_swaptions_data(str(data_path))
    feature_cols = get_feature_columns(df)

    # Sanity check column alignment
    if list(feature_cols) != list(feature_cols_ckpt):
        raise ValueError(
            "Feature columns in train.xlsx do not match those in the checkpoint.\n"
            f"From data:   {feature_cols[:5]} ...\n"
            f"From ckpt:   {feature_cols_ckpt[:5]} ..."
        )

    last_surface_unscaled = df[feature_cols].iloc[-1].values
    last_date = df["Date"].iloc[-1]

    print(f"Last real date in dataset: {last_date}")
    print("Generating 14-day trajectory using the GAN...")

    forecast_df = generate_14day_trajectory(
        gen=gen,
        mean=mean,
        std=std,
        feature_cols=feature_cols,
        last_surface_unscaled=last_surface_unscaled,
        last_date=last_date,
        latent_dim=latent_dim,
        n_days=14,
        device=device,
    )

    print("\nGAN 14-day sample preview:")
    print(forecast_df.head())

    out_path = base_dir / "data" / "Track2_QML" / "gan_14day_samples.csv"
    forecast_df.to_csv(out_path, index=False)
    print(f"\nSaved GAN 14-day trajectory to: {out_path}")


if __name__ == "__main__":
    main()