# Qortex – Quantum & Classical Models for Swaption Surface Forecasting

This repository contains our work for **Qiskit Fall Fest / Mil’Haq – Track 2 (Quantum Machine Learning)**.

We model the **evolution of swaption implied volatility surfaces** and compare:

- A **Qiskit-based variational quantum model** (Alexis’ VQA-style circuit)
- A **strong classical MLP baseline** with engineered features
- A **classical conditional GAN** for trajectory sampling
- A **hybrid photonic QML model** built with **MerLin / Perceval**

Our **official submission** for Track 2 is based on **Alexis’ Qiskit model**, which we package into the final Excel file.

---

## 1. Installation & Setup

### Prerequisites

- Python **3.11**
- `pip` (or [`uv`](https://github.com/astral-sh/uv) if you prefer)
- Git

The repo already contains all the organizer data under `data/Track2_QML`.

---

### Option A – Using `pip` (standard)

```bash
git clone https://github.com/TheBarzani/qortex.git
cd qortex

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
````

### Option B – Using `uv` (fast)

```bash
git clone https://github.com/TheBarzani/qortex.git
cd qortex

uv venv --python 3.11
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

uv pip install -r requirements.txt
# or
uv pip sync requirements.txt
```

### Quick import check

In a Python shell:

```python
import numpy as np
import torch
import qiskit
import perceval
import merlin
print("All core packages import correctly.")
```

---

## 2. Repository Structure

```text
qortex/
├── data/
│   ├── Track1_Walk/                      # Track 1 (Black-Scholes) – not central here
│   └── Track2_QML/
│       ├── csv/
│       │   ├── train.csv
│       │   ├── template_track2_results.csv
│       │   ├── forecast.csv
│       │   ├── final_dataset.csv         # final QML submission (Alexis)
│       ├── xlsx/
│       │   ├── train.xlsx
│       │   ├── template_track2_results.xlsx
│       │   ├── forecast.xlsx
│       │   ├── final_dataset.xlsx        # final QML submission (Alexis)
│       ├── df_normalized_lags2_from_pacf.csv
│       ├── gan_14day_samples.csv
│       ├── mlp_14day_forecast.csv
│       ├── mlp_14day_forecast_with_features.csv
│       ├── qml_merlin_14day_forecast.csv
│       ├── qml_merlin_14day_forecast_engineered.csv
│       ├── generator_classical.pt        # trained classical GAN generator
│       └── plots/
│           ├── evolution_snapshots.png
│           ├── heatmap_latest.png
│           ├── swaption_surface_evolution.gif
│           ├── time_series.png
│           └── time_series_mlp_vs_gan_Tenor__9__Maturity__0.75.png
├── docs/                                 # Challenge prompt & instructions
├── notebooks/                            # Exploration notebooks (MerLin, Qiskit, PyTorch…)
└── src/
    ├── alexis_vqa.py                     # Qiskit variational model (official submission)
    ├── classical_vqa.py                  # 24/66-parameter classical analogue of Alexis’ model
    ├── data_loading.py                   # Feature engineering, lags, normalization
    ├── features.py                       # Additional feature utilities
    ├── forecast_mlp.py                   # Strong MLP baseline (14-day forecast)
    ├── qml_merlin_forecast.py            # Hybrid MerLin QML model (14-day forecast)
    ├── generator_discriminator/
    │   ├── models.py                     # Classical GAN: Generator & Discriminator
    │   ├── dataset.py
    │   ├── train_gan.py                  # Train conditional GAN
    │   └── sample_gan_14day.py           # 14-day GAN trajectory sampling
    └── visualization/
        └── plot_time_series.py           # Time-series viz (MLP vs GAN vs QML)
```

---

## 3. Data

Track 2 swaptions data lives under:

* `data/Track2_QML/xlsx/train.xlsx` – **training surfaces**
* `data/Track2_QML/xlsx/template_track2_results.xlsx` – **organizer template**
* `data/Track2_QML/xlsx/final_dataset.xlsx` – **our submitted forecast**
* `data/Track2_QML/csv/*` – CSV mirrors of the above + our model outputs

All scripts assume this folder layout; you shouldn’t need to move anything.

---

## 4. Main Models

### 4.1 Alexis’ Qiskit VQA Model (Submission)

**Script:** `src/alexis_vqa.py`
**Goal:** Learn low-dimensional dynamics of the volatility surface using a small variational quantum circuit in **PCA space**, then reconstruct full surfaces.

Pipeline:

1. Load `train.xlsx`, drop the `Date` column.
2. Apply **PCA (2 components)** to compress each full surface.
3. Build **time windows**:

   * Input: 3 consecutive PCA points
   * Target: next 3 PCA points
4. Qiskit variational layer:

   * Parametric circuit with Ry rotations + CZ entanglers
   * Expectation values of Z on each qubit → classical features
5. Linear head maps quantum features back to 3×2 PCA points.
6. Autoregressive generation in PCA space, then inverse PCA to recover surfaces.
7. Format the result on the Tenor × Maturity grid and save:

   * `data/Track2_QML/csv/final_dataset.csv`
   * `data/Track2_QML/xlsx/final_dataset.xlsx`

Run:

```bash
cd src
python alexis_vqa.py
```

> This is the model we use to generate the **Excel file submitted to the organizers**.

---

### 4.2 24/66-Parameter Classical Analogue

**Script:** `src/classical_vqa.py`

We mirror Alexis’ architecture with a **tiny fully classical network** that has the **same parameter budget** as the quantum model (24 or 66 parameters, depending on the config).

* Same **PCA** preprocessing and **3 → 3 windowing**.
* Replace the Qiskit circuit by a small `ClassicalObsLayer`:

  * Input: flattened 3×2 window
  * Parameters: `theta` with shape `(reps, n)` → **same count as Alexis’ θs**.
  * Nonlinear mixing (`tanh` over “phases”) imitates encoding + entangling + measurement.
* Reshape back to 3×2 PCA predictions.

Run:

```bash
cd src
python classical_vqa.py
```

This gives us a **fair classical benchmark** for “same architecture, same number of trainable parameters”.

---

### 4.3 MLP Baseline with Engineered Features

**Script:** `src/forecast_mlp.py`
**Feature engineering:** `src/data_loading.py`, `src/features.py`

Steps:

1. Build an engineered dataset:

   * Time features: day of week, month, quarter
   * Global surface stats: mean, std, skew, kurtosis
   * Rolling features (e.g. 5-day moving average, volatility)
   * Lagged values of key surface points

   Saved to: `data/Track2_QML/df_normalized_lags2_from_pacf.csv`.

2. Train a **simple MLP** on the engineered features:

   * Input: engineered features at day *t*
   * Target: full surface at day *t+1*
   * Gets a very strong fit (R² ≈ 0.998 on validation).

3. Autoregressive **14-day forecast** in surface space:

   * Output: `mlp_14day_forecast_with_features.csv`
   * Also a version with plain data: `mlp_14day_forecast.csv`.

Run:

```bash
cd src
python data_loading.py      # build engineered dataset
python forecast_mlp.py      # train + 14-day forecast
```

---

### 4.4 Classical Conditional GAN

**Folder:** `src/generator_discriminator/`

We implement a **classical conditional GAN** that maps:

* Context: current-day surface
* Noise: latent vector
* Output: next-day surface

Files:

* `models.py` – Generator & Discriminator (MLPs)
* `dataset.py` – wraps `(context, target)` pairs
* `train_gan.py` – train loop (conditional GAN)
* `sample_gan_14day.py` – autoregressive 14-day sampling

Basic usage:

```bash
cd src/generator_discriminator

# 1. Train GAN
python train_gan.py   # saves generator to data/Track2_QML/generator_classical.pt

# 2. Sample a 14-day trajectory
python sample_gan_14day.py
# → data/Track2_QML/gan_14day_samples.csv
```

We also provide **plots** comparing the GAN and MLP on selected Tenor/Maturity points, e.g.:

* `data/Track2_QML/plots/time_series_mlp_vs_gan_Tenor__9__Maturity__0.75.png`
* `data/Track2_QML/plots/time_series.png`

---

### 4.5 Hybrid QML with MerLin / Perceval

**Script:** `src/qml_merlin_forecast.py`
**Notebook:** `notebooks/FirstQuantumLayers_with_MerLin.ipynb`

This model wraps a **MerLin `QuantumLayer`** (photonic circuit) inside a classical head:

* Input: first 6 PCA/engineered features
* Core: MerLin `QuantumLayer` with entangling layers, angle encoding, rotations, and superpositions
* Output head: dense network mapping quantum features → full 224-dim surface

We support:

* Training on plain surfaces
* Training on the engineered dataset (`df_normalized_lags2_from_pacf.csv`)
* 14-day **autoregressive forecast**:

  * `qml_merlin_14day_forecast.csv`
  * `qml_merlin_14day_forecast_engineered.csv`

Run:

```bash
cd src
python qml_merlin_forecast.py
```

---

## 5. Visualization

Basic visualizations live in `data/Track2_QML/plots/` and `src/visualization/`:

* **Surface evolution** gifs:

  * `swaption_surface_evolution.gif`
  * `plotsevolution.gif`
* **Time series** at specific Tenor/Maturity:

  * `time_series.png`
  * `time_series_mlp_vs_gan_Tenor__9__Maturity__0.75.png`

You can regenerate / extend plots via:

```bash
cd src
python visualization/plot_time_series.py
```

---

## 6. Reproducing the Submission Artifacts

For the **Track 2 submission**, the organizers ask for:

1. **Excel results** following the template
2. **Code** (this repo)
3. **Short presentation** (3 slides / 3 minutes)

In this repo:

* Final QML forecast (Alexis’ model):

  * `data/Track2_QML/xlsx/final_dataset.xlsx`
  * `data/Track2_QML/csv/final_dataset.csv`

* Classical baselines & QML variants:

  * `mlp_14day_forecast_with_features.csv`
  * `gan_14day_samples.csv`
  * `qml_merlin_14day_forecast_engineered.csv`

To regenerate Alexis’ final dataset:

```bash
cd src
python alexis_vqa.py
# Rewrites final_dataset.(csv|xlsx) under data/Track2_QML
```

You can then copy `final_dataset.xlsx` into the organizer’s **Track 2 template** as required.

---

## 7. Team

* **Timothy Roch**
* **Alexis Vieloszynski**
* **Jérôme Francis** 
* **Ismaël Barzani** 
* **Rani Naaman**

---

## 8. Notes & Limitations

* The Qiskit and MerLin models currently use **statevector simulators**, not noisy hardware.
* Some scripts can be slow for large epochs due to exact simulation.
* The small VQA and its tiny classical analogue are intentionally parameter-limited to study the effect of **circuit expressivity vs parameter count**.
