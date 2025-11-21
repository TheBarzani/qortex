"""

Create a time-series plot comparing:
- historical swaption surface values
- 14-day MLP forecast
- 14-day GAN trajectory

Output: a PNG you can embed in your README.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent        # .../src/visualization
SRC_DIR = CURRENT_DIR.parent                         # .../src
ROOT_DIR = SRC_DIR.parent                            # repo root

for p in (SRC_DIR,):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from data_loading import load_swaptions_data, get_feature_columns  # type: ignore


def main():
    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------
    data_path = ROOT_DIR / "data" / "Track2_QML" / "train.xlsx"
    mlp_forecast_path = ROOT_DIR / "data" / "Track2_QML" / "mlp_14day_forecast_with_features.csv"
    gan_forecast_path = ROOT_DIR / "data" / "Track2_QML" / "gan_14day_samples.csv"

    out_dir = ROOT_DIR / "data" / "Track2_QML" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    df_hist = load_swaptions_data(str(data_path))
    feature_cols = get_feature_columns(df_hist)

    # Choose one surface point to visualize.
    # e.g. idx = 0 (first column) or pick a mid/long tenor.
    idx = 50  # tweak this as you like
    col_name = feature_cols[idx]

    print(f"Using feature column: {col_name}")

    df_mlp = pd.read_csv(mlp_forecast_path)
    df_mlp["Date"] = pd.to_datetime(df_mlp["Date"])

    df_gan = pd.read_csv(gan_forecast_path)
    df_gan["Date"] = pd.to_datetime(df_gan["Date"])

    # -----------------------------------------------------------------
    # Build combined time series
    # -----------------------------------------------------------------
    # Last N historical days for context
    N_HIST = 60
    df_hist_tail = df_hist[["Date", col_name]].tail(N_HIST).copy()
    df_hist_tail["Source"] = "Historical"

    df_mlp_series = df_mlp[["Date", col_name]].copy()
    df_mlp_series["Source"] = "MLP 14-day forecast"

    df_gan_series = df_gan[["Date", col_name]].copy()
    df_gan_series["Source"] = "GAN 14-day sample"

    df_all = pd.concat([df_hist_tail, df_mlp_series, df_gan_series], ignore_index=True)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    # Historical
    mask_hist = df_all["Source"] == "Historical"
    plt.plot(
        df_all.loc[mask_hist, "Date"],
        df_all.loc[mask_hist, col_name],
        label="Historical",
        linewidth=2,
    )

    # MLP forecast
    mask_mlp = df_all["Source"] == "MLP 14-day forecast"
    plt.plot(
        df_all.loc[mask_mlp, "Date"],
        df_all.loc[mask_mlp, col_name],
        linestyle="--",
        label="MLP forecast",
        linewidth=2,
    )

    # GAN sample
    mask_gan = df_all["Source"] == "GAN 14-day sample"
    plt.plot(
        df_all.loc[mask_gan, "Date"],
        df_all.loc[mask_gan, col_name],
        linestyle=":",
        label="GAN sample",
        linewidth=2,
    )

    plt.title(f"Time series for {col_name}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Make filename safe
    safe_col = col_name.replace(" ", "_").replace(";", "_").replace(":", "").replace("/", "_")
    out_path = out_dir / f"time_series_mlp_vs_gan_{safe_col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved time series plot to: {out_path}")


if __name__ == "__main__":
    main()