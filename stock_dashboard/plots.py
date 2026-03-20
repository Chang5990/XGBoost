import matplotlib.pyplot as plt


def plot_actual_vs_pred(df, stock_name):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["y_true"], label="Actual")
    ax.plot(df["date"], df["y_pred"], label="Predicted")
    ax.set_title(f"{stock_name} - Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_scatter(df, stock_name):
    fig, ax = plt.subplots(figsize=(6, 5))

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    ax.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_title(f"{stock_name} - Actual vs Predicted Scatter")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_residual_hist(df, stock_name):
    residuals = df["y_true"] - df["y_pred"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=30)
    ax.set_title(f"{stock_name} - Residual Distribution")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_residual_over_time(df, stock_name):
    df_plot = df.copy()
    df_plot["residual"] = df_plot["y_true"] - df_plot["y_pred"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_plot["date"], df_plot["residual"])
    ax.axhline(0, linestyle="--")
    ax.set_title(f"{stock_name} - Residuals Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_metrics_bar(metrics_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(metrics_df["Stock"], metrics_df["RMSE"])
    axes[0].set_title("RMSE by Stock")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y")

    axes[1].bar(metrics_df["Stock"], metrics_df["MAE"])
    axes[1].set_title("MAE by Stock")
    axes[1].set_ylabel("MAE")
    axes[1].grid(axis="y")

    axes[2].bar(metrics_df["Stock"], metrics_df["R2"])
    axes[2].set_title("R² by Stock")
    axes[2].set_ylabel("R²")
    axes[2].grid(axis="y")

    fig.tight_layout()
    return fig