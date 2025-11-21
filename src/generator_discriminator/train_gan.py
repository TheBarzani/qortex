"""

Classical conditional GAN for swaption surfaces.

- Uses current-day surface as context
- Generator G(context, noise) -> next-day surface (scaled)
- Discriminator D(context, surface) -> real/fake logit

This is the classical counterpart of a potential quantum GAN architecture.
We also expose parameter counts so they can be matched with the quantum models.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent          # .../src/generator_discriminator
SRC_DIR = CURRENT_DIR.parent                           # .../src

for p in (CURRENT_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from data_loading import (
    load_swaptions_data,
    prepare_dataset_data,
    get_feature_columns,
)
from models import Generator, Discriminator
from dataset import SwaptionGANDataset
from utils import count_parameters, set_seed, get_device


def prepare_gan_data(train_ratio: float = 0.9):
    """
    Load swaption data and prepare (context, target) arrays for GAN training.

    Context: current-day surface
    Target:  next-day surface

    Both are scaled using global mean/std over all surfaces.
    """
    base_dir = Path(__file__).resolve().parents[2]  # go from src/generator_discriminator to repo root
    data_path = base_dir / "data" / "Track2_QML" / "train.xlsx"

    df = load_swaptions_data(str(data_path))
    feature_cols = get_feature_columns(df)

    # X: today surface, y: next-day surface
    X_df, y_df = prepare_dataset_data(df, feature_cols=feature_cols)

    X = X_df.values  # (T-1, n_features)
    y = y_df.values  # (T-1, n_features)

    n_samples, n_features = X.shape

    # Use combined surfaces (X and y) to compute scaling stats
    all_surfaces = np.vstack([X, y])
    mean = all_surfaces.mean(axis=0)
    std = all_surfaces.std(axis=0) + 1e-8

    X_scaled = (X - mean) / std
    y_scaled = (y - mean) / std

    # Time-based split
    split_idx = int(train_ratio * n_samples)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        mean,
        std,
        feature_cols,
        df["Date"].iloc[: len(X_df)].reset_index(drop=True),
    )


def train_gan(
    latent_dim: int = 16,
    hidden_sizes_gen=(128, 128),
    hidden_sizes_disc=(128, 128),
    batch_size: int = 64,
    n_epochs: int = 50,
    lr_gen: float = 1e-4,
    lr_disc: float = 4e-4,
):
    """
    Main training loop for the conditional GAN.
    """
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        mean,
        std,
        feature_cols,
        dates_aligned,
    ) = prepare_gan_data(train_ratio=0.9)

    context_dim = X_train.shape[1]
    target_dim = y_train.shape[1]

    # Datasets and loaders
    train_ds = SwaptionGANDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Models
    gen = Generator(
        context_dim=context_dim,
        target_dim=target_dim,
        latent_dim=latent_dim,
        hidden_sizes=hidden_sizes_gen,
    ).to(device)

    disc = Discriminator(
        context_dim=context_dim,
        target_dim=target_dim,
        hidden_sizes=hidden_sizes_disc,
    ).to(device)

    print(f"Generator parameters:   {count_parameters(gen)}")
    print(f"Discriminator parameters: {count_parameters(disc)}")

    # Optimizers & loss
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    # Adversarial training
    for epoch in range(1, n_epochs + 1):
        gen.train()
        disc.train()
        d_losses = []
        g_losses = []

        for context_batch, real_surface_batch in train_loader:
            context_batch = context_batch.to(device)
            real_surface_batch = real_surface_batch.to(device)
            batch_size_curr = context_batch.size(0)

            # -----------------------
            # Train Discriminator
            # -----------------------
            opt_disc.zero_grad()

            # Real samples
            real_labels = torch.ones(batch_size_curr, 1, device=device)
            fake_labels = torch.zeros(batch_size_curr, 1, device=device)

            logits_real = disc(context_batch, real_surface_batch)
            loss_real = criterion(logits_real, real_labels)

            # Fake samples
            noise = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_surface = gen(context_batch, noise)
            logits_fake = disc(context_batch, fake_surface.detach())
            loss_fake = criterion(logits_fake, fake_labels)

            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            opt_disc.step()
            d_losses.append(loss_disc.item())

            # -----------------------
            # Train Generator
            # -----------------------
            opt_gen.zero_grad()
            noise = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_surface = gen(context_batch, noise)
            logits_fake_for_gen = disc(context_batch, fake_surface)
            # Generator tries to fool D -> want labels "real"
            loss_gen = criterion(logits_fake_for_gen, real_labels)
            loss_gen.backward()
            opt_gen.step()
            g_losses.append(loss_gen.item())

        avg_d = float(np.mean(d_losses)) if d_losses else 0.0
        avg_g = float(np.mean(g_losses)) if g_losses else 0.0

        print(f"Epoch {epoch:03d} | D_loss={avg_d:.4f} | G_loss={avg_g:.4f}")

    # Save generator for later comparison / sampling
    base_dir = Path(__file__).resolve().parents[2]
    out_path = base_dir / "data" / "Track2_QML" / "generator_classical.pt"
    torch.save(
        {
            "generator_state_dict": gen.state_dict(),
            "context_dim": context_dim,
            "target_dim": target_dim,
            "latent_dim": latent_dim,
            "mean": mean,
            "std": std,
            "feature_cols": feature_cols,
        },
        out_path,
    )
    print(f"\nSaved trained generator to: {out_path}")


if __name__ == "__main__":
    train_gan()
