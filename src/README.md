# Track 2 – Classical Baseline & 14-Day Forecast

This is the **classical (non-quantum)** side of our solution for Track 2 (swaption prices).  
The main goal: build a **strong, fast baseline** to compare against the QML models.

---

## 1. Quick experiments (`exp.py`)

We first ran some simple models to see what works well on **next-day surface prediction**:

- **Linear regression** (PolynomialFeatures degree 1)
- **Polynomial regression** (degree 2)
- **XGBoost** (multi-output: one regressor per surface point)

All models use the same data preparation from `data_loading.py`, where:
- `X` = today’s swaption surface  
- `y` = next day’s surface (shifted by one day)

**Results (1-step ahead):**

| Model       | RMSE      | MAE       | R²        |
|------------|-----------|-----------|-----------|
| Linear     | ~0.129    | ~0.019    | ~-0.69    |
| Poly deg2  | ~0.220    | ~0.033    | ~-3.92    |
| **XGBoost** | **~0.0045** | **~0.0032** | **~0.998** |

**Why we moved on:**

- Linear / polynomial models clearly **don’t fit** the surface dynamics.
- XGBoost gives **excellent accuracy**, but as a multi-output model it becomes heavy to use repeatedly (e.g. for long recursive forecasts or quick iteration).

That motivated a model that is:
- Multi-output by design  
- Fast to train and evaluate  
- Still competitive with XGBoost  

→ enter the **MLP**.

---

## 2. Main model: multi-output MLP (`forecast_mlp.py`)

`forecast_mlp.py` is our **main classical forecasting script**.

What it does:

1. **Loads and prepares data** via `data_loading.py`  
   - Uses `load_swaptions_data` and `prepare_dataset_data`
   - All non-`Date` columns are treated as surface points
   - Builds `(X, y)` where:
     - `X[t]` = surface at day *t*
     - `y[t]` = surface at day *t+1* (next day)

2. **Scales the surfaces**  
   - Computes feature-wise mean and std on `X`
   - Applies the same scaling to both `X` and `y`  
   (since input and target are the same type of object: swaption surface)

3. **Splits data in time order**  
   - First 80% of samples: train  
   - Last 20%: validation  
   - No shuffling (we respect temporal structure)

4. **Trains a multi-output MLP (PyTorch)**  
   - Input dimension = number of surface points  
   - Hidden layers: `256 → 128` with ReLU  
   - Output dimension = number of surface points (full next-day surface)  
   - Trained for 60 epochs with Adam + MSE loss

5. **Evaluates 1-step ahead performance** on the validation set  
   - Predictions are unscaled back to the original space
   - Metrics computed on the entire surface (flattened):

   **Validation metrics (1-step ahead MLP):**

   - RMSE ≈ **0.00418**  
   - MAE  ≈ **0.00316**  
   - R²   ≈ **0.99806**

   → This is basically **on par with XGBoost**, but with a model that is:
   - native multi-output  
   - simpler to control  
   - fast enough for repeated use.

6. **Performs a 14-day recursive forecast**  
   - Starts from the **last real surface** in `train.xlsx`
   - Applies the trained MLP 14 times:
     - feed surface at *t* → predict *t+1*
     - use prediction as input for the next step
   - Unscales predictions back to the original price space
   - Generates future dates D+1 … D+14
   - Builds a DataFrame with:

     - `Date` column (future dates)
     - All tenor/maturity columns

   - Saves the result to:

     ```text
     data/Track2_QML/mlp_14day_forecast.csv
     ```

   A preview of the output looks like:

   - 14 rows  
   - 1 `Date` column  
   - 224 surface columns (e.g. `Tenor : X; Maturity : Y`)

---

## 3. How this connects to the QML work

- `forecast_mlp.py` provides a **strong classical baseline** for:
  - 1-step ahead prediction
  - 14-day recursive forecast of the full swaption surface

- The QML models can be evaluated on the **same tasks** and compared against:
  - MLP metrics (above)
  - Previously measured XGBoost metrics from `exp.py`

The idea is:
- Use XGBoost + MLP to show what good **classical** models can already do.
- Then see whether QML can **match or improve** on these baselines, or offer advantages in other aspects (e.g. structure, robustness, interpretability, etc.).
