import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def get_latest_metrics(df: pd.DataFrame):
    latest_row = df.sort_values("date").iloc[-1]
    latest_actual = latest_row["y_true"]
    latest_pred = latest_row["y_pred"]
    latest_error = latest_actual - latest_pred
    return latest_actual, latest_pred, latest_error


def build_metrics_table(all_results: dict) -> pd.DataFrame:
    rows = []

    for stock_name, df in all_results.items():
        rmse, mae, r2 = calculate_metrics(df["y_true"], df["y_pred"])
        rows.append({
            "Stock": stock_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    return pd.DataFrame(rows)