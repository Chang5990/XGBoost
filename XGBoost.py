# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


RAW_PATH = "all_stocks_5yr.csv"
SELECTED_STOCKS = ["AAPL", "MSFT", "XOM"]
OUTPUT_DIR = "stock_dashboard/results"

TRAIN_SIZE = 252
TEST_SIZE = 20
RANDOM_STATE = 42


def clean_stock_data(file_path, selected_stocks=None):
    df = pd.read_csv(file_path)

    print("Original dataset shape:", df.shape)
    print("Original columns:", df.columns.tolist())

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Name"] = df["Name"].astype(str).str.strip().str.upper()

    print("\nMissing values:")
    print(df.isna().sum())

    before_dropna = len(df)
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume", "Name"])
    print(f"\nAfter dropping missing values: {before_dropna} -> {len(df)}")

    before_dup = len(df)
    df = df.drop_duplicates()
    print(f"After dropping full-row duplicates: {before_dup} -> {len(df)}")

    before_name_date_dup = len(df)
    df = df.drop_duplicates(subset=["Name", "date"], keep="first")
    print(f"After dropping Name+date duplicates: {before_name_date_dup} -> {len(df)}")

    price_logic_ok = (
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    )

    non_negative_ok = (
        (df["open"] >= 0) &
        (df["high"] >= 0) &
        (df["low"] >= 0) &
        (df["close"] >= 0) &
        (df["volume"] >= 0)
    )

    invalid_rows = df[~(price_logic_ok & non_negative_ok)]
    print(f"Invalid rows removed: {len(invalid_rows)}")

    df = df[price_logic_ok & non_negative_ok].copy()
    print("Shape after cleaning invalid rows:", df.shape)

    if selected_stocks is not None:
        selected_stocks = [s.strip().upper() for s in selected_stocks]
        df = df[df["Name"].isin(selected_stocks)].copy()
        print(f"\nShape after filtering {selected_stocks}:", df.shape)
        print("Samples per stock:")
        print(df["Name"].value_counts())

    df = df.sort_values(["Name", "date"]).reset_index(drop=True)
    return df


def build_features(df):
    df_feat = df.copy()
    grp = df_feat.groupby("Name")

    df_feat["daily_return"] = grp["close"].pct_change()
    df_feat["price_range"] = df_feat["high"] - df_feat["low"]
    df_feat["close_open_diff"] = df_feat["close"] - df_feat["open"]

    df_feat["close_lag1"] = grp["close"].shift(1)
    df_feat["close_lag2"] = grp["close"].shift(2)
    df_feat["return_lag1"] = grp["daily_return"].shift(1)
    df_feat["return_lag2"] = grp["daily_return"].shift(2)

    df_feat["ma_5"] = grp["close"].transform(lambda s: s.rolling(5).mean())
    df_feat["ma_10"] = grp["close"].transform(lambda s: s.rolling(10).mean())
    df_feat["ma_20"] = grp["close"].transform(lambda s: s.rolling(20).mean())

    df_feat["volatility_5"] = grp["daily_return"].transform(lambda s: s.rolling(5).std())
    df_feat["volatility_10"] = grp["daily_return"].transform(lambda s: s.rolling(10).std())

    df_feat["target"] = grp["close"].shift(-1)

    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat


FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "daily_return", "price_range", "close_open_diff",
    "close_lag1", "close_lag2",
    "return_lag1", "return_lag2",
    "ma_5", "ma_10", "ma_20",
    "volatility_5", "volatility_10"
]


PARAM_GRID = [
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 80, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 80, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 80, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 120, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 3, "learning_rate": 0.08, "n_estimators": 60, "subsample": 0.8, "colsample_bytree": 0.8},
]


def build_model(params):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4
    )


def rolling_train_test(group, feature_cols, model_params, train_size=252, test_size=20):
    g = group.sort_values("date").reset_index(drop=True)

    preds = []
    trues = []
    dates = []

    for start in range(0, len(g) - train_size - test_size + 1, test_size):
        train_df = g.iloc[start:start + train_size]
        test_df = g.iloc[start + train_size:start + train_size + test_size]

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test = test_df[feature_cols]
        y_test = test_df["target"]

        model = build_model(model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        preds.extend(y_pred)
        trues.extend(y_test.values)
        dates.extend(test_df["date"].values)

    result_df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "y_true": trues,
        "y_pred": preds
    })

    if not result_df.empty:
        result_df["residual"] = result_df["y_true"] - result_df["y_pred"]
        result_df["abs_error"] = np.abs(result_df["residual"])

    return result_df


def calculate_metrics(result_df):
    mse = mean_squared_error(result_df["y_true"], result_df["y_pred"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(result_df["y_true"], result_df["y_pred"])
    r2 = r2_score(result_df["y_true"], result_df["y_pred"])
    return rmse, mae, r2


def tune_hyperparameters(group, stock_name, feature_cols, param_grid, train_size=252, test_size=20):
    print(f"\n{'=' * 60}")
    print(f"Tuning hyperparameters for {stock_name}")
    print(f"{'=' * 60}")

    tuning_records = []

    for i, params in enumerate(param_grid, start=1):
        result_df = rolling_train_test(
            group,
            feature_cols=feature_cols,
            model_params=params,
            train_size=train_size,
            test_size=test_size
        )

        if result_df.empty:
            print(f"Trial {i}: skipped because there is not enough data.")
            continue

        rmse, mae, r2 = calculate_metrics(result_df)

        record = {
            "trial": i,
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
        tuning_records.append(record)

        print(
            f"Trial {i} | "
            f"max_depth={params['max_depth']}, "
            f"learning_rate={params['learning_rate']}, "
            f"n_estimators={params['n_estimators']}, "
            f"subsample={params['subsample']}, "
            f"colsample_bytree={params['colsample_bytree']} | "
            f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
        )

    tuning_df = pd.DataFrame(tuning_records).sort_values("RMSE").reset_index(drop=True)

    print(f"\nTop tuning results for {stock_name}:")
    print(tuning_df.to_string(index=False))

    best_row = tuning_df.iloc[0]
    best_params = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "n_estimators": int(best_row["n_estimators"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"])
    }

    print(f"\nBest parameters for {stock_name}: {best_params}")
    print(f"Best validation RMSE: {best_row['RMSE']:.4f}")

    return best_params, tuning_df


def plot_predictions(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)

    if n == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, all_results.items()):
        df_plot = result.copy().sort_values("date")
        ax.plot(df_plot["date"], df_plot["y_true"], label="Actual")
        ax.plot(df_plot["date"], df_plot["y_pred"], label="Predicted")
        ax.set_title(f"{name} - Actual vs Predicted Close Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_scatter(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, all_results.items()):
        y_true = result["y_true"]
        y_pred = result["y_pred"]

        ax.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        ax.set_title(f"{name} - Actual vs Predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_residuals_over_time(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)

    if n == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, all_results.items()):
        df_plot = result.copy().sort_values("date")
        df_plot["residual"] = df_plot["y_true"] - df_plot["y_pred"]

        ax.plot(df_plot["date"], df_plot["residual"])
        ax.axhline(0, linestyle="--")
        ax.set_title(f"{name} - Residuals Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Residual")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = clean_stock_data(RAW_PATH, selected_stocks=SELECTED_STOCKS)
    df_feat = build_features(df)

    print("\nFeature dataset shape:", df_feat.shape)
    print("Feature columns:")
    print(FEATURE_COLS)

    all_results = {}
    metrics_rows = []
    best_params_rows = []

    for name, group in df_feat.groupby("Name"):
        best_params, tuning_df = tune_hyperparameters(
            group,
            stock_name=name,
            feature_cols=FEATURE_COLS,
            param_grid=PARAM_GRID,
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE
        )

        final_result = rolling_train_test(
            group,
            feature_cols=FEATURE_COLS,
            model_params=best_params,
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE
        )

        if final_result.empty:
            print(f"{name} skipped because there is not enough data.")
            continue

        rmse, mae, r2 = calculate_metrics(final_result)

        all_results[name] = final_result
        metrics_rows.append({
            "Stock": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })
        best_params_rows.append({
            "Stock": name,
            **best_params
        })

        print(f"\nFinal evaluation for {name} using best parameters")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)
    best_params_df = pd.DataFrame(best_params_rows)

    print(f"\n{'=' * 60}")
    print("Final metrics summary")
    print(f"{'=' * 60}")
    print(metrics_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("Best parameters summary")
    print(f"{'=' * 60}")
    print(best_params_df.to_string(index=False))

    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)

    for name, result in all_results.items():
        result.to_csv(os.path.join(OUTPUT_DIR, f"{name}_predictions.csv"), index=False)
        print(f"{name} predictions exported to {os.path.join(OUTPUT_DIR, f'{name}_predictions.csv')}")

    plot_predictions(all_results)
    plot_scatter(all_results)
    plot_residuals_over_time(all_results)


if __name__ == "__main__":
    main()