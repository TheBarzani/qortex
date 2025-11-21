"""
exp.py

Safe experiment script comparing:
- Linear Regression (Poly degree 1)
- Polynomial Regression (Poly degree 2)
- XGBoost Regressor

"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb

from data_loading import load_swaptions_data, prepare_dataset_data


def get_train_test_data(test_size: float = 0.2, random_state: int = 42):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "Track2_QML" / "train.xlsx"

    df = load_swaptions_data(str(data_path))
    X_df, y_df = prepare_dataset_data(df)

    X = X_df.values
    y = y_df.values

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_poly_model(degree: int = 1) -> Pipeline:
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )


def build_xgb_model() -> MultiOutputRegressor:
    base = xgb.XGBRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )
    return MultiOutputRegressor(base)


def evaluate_regression(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    return {
        "RMSE": np.sqrt(mean_squared_error(yt, yp)),
        "MAE": mean_absolute_error(yt, yp),
        "R2": r2_score(yt, yp),
    }


def main():
    X_train, X_test, y_train, y_test = get_train_test_data()

    experiments = {
        "Linear": build_poly_model(degree=1),
        "Poly_deg2": build_poly_model(degree=2),
        "XGBoost": build_xgb_model(),
    }

    results = []

    for name, model in experiments.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
        metrics["Model"] = name
        results.append(metrics)

        print(f"{name} metrics:")
        for k, v in metrics.items():
            if k != "Model":
                print(f"  {k}: {v:.6f}")

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n=== Metrics comparison matrix ===")
    print(results_df)


if __name__ == "__main__":
    main()
