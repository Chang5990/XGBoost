import os
import streamlit as st
from data_loader import load_all_results, load_single_result
from metrics_utils import calculate_metrics, get_latest_metrics, build_metrics_table
from plots import (
    plot_actual_vs_pred,
    plot_scatter,
    plot_residual_hist,
    plot_residual_over_time,
    plot_metrics_bar
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Price Prediction Dashboard")
st.caption("Model: XGBoost Regressor | Strategy: Rolling Window Split")

all_results = load_all_results(RESULTS_DIR)

if not os.path.exists(RESULTS_DIR):
    st.error(f"Results folder not found: {RESULTS_DIR}")
    st.stop()

if not all_results:
    st.error("No result files were found. Please make sure the results folder contains *_predictions.csv files.")
    st.stop()

stock_names = list(all_results.keys())
selected_stock = st.sidebar.selectbox("Select Stock", stock_names)

st.sidebar.markdown("### Methodology")
st.sidebar.write(
    """
    - Model: XGBoost Regressor  
    - Features: lag variables, moving averages, volatility, and volume-related features  
    - Split strategy: rolling-window time-series split  
    - Reason: a fixed split may cause distribution shift in stock prices
    """
)

df_stock = load_single_result(RESULTS_DIR, selected_stock)

rmse, mae, r2 = calculate_metrics(df_stock["y_true"], df_stock["y_pred"])
latest_actual, latest_pred, latest_error = get_latest_metrics(df_stock)

st.subheader(f"{selected_stock} Prediction Table")

df_display = df_stock.copy()
df_display["error"] = df_display["y_true"] - df_display["y_pred"]

df_display["date"] = df_display["date"].dt.strftime("%Y-%m-%d")
df_display["y_true"] = df_display["y_true"].round(2)
df_display["y_pred"] = df_display["y_pred"].round(2)
df_display["error"] = df_display["error"].round(2)

df_display = df_display.rename(columns={
    "date": "Date",
    "y_true": "Actual Price",
    "y_pred": "Predicted Price",
    "error": "Prediction Error"
})

st.dataframe(
    df_display[["Date", "Actual Price", "Predicted Price", "Prediction Error"]]
    .sort_values("Date", ascending=False)
    .reset_index(drop=True),
    use_container_width=True,
    height=320
)

st.markdown("---")

st.subheader(f"{selected_stock} Performance Summary")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("RMSE", f"{rmse:.4f}")
c2.metric("MAE", f"{mae:.4f}")
c3.metric("R²", f"{r2:.4f}")
c4.metric("Latest Actual", f"{latest_actual:.2f}")
c5.metric("Latest Predicted", f"{latest_pred:.2f}")
c6.metric("Latest Error", f"{latest_error:.2f}")

st.markdown("---")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Actual vs Predicted")
    fig1 = plot_actual_vs_pred(df_stock, selected_stock)
    st.pyplot(fig1, use_container_width=True)

with row1_col2:
    st.subheader("Actual vs Predicted Scatter")
    fig2 = plot_scatter(df_stock, selected_stock)
    st.pyplot(fig2, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Residual Distribution")
    fig3 = plot_residual_hist(df_stock, selected_stock)
    st.pyplot(fig3, use_container_width=True)

with row2_col2:
    st.subheader("Residuals Over Time")
    fig4 = plot_residual_over_time(df_stock, selected_stock)
    st.pyplot(fig4, use_container_width=True)

st.markdown("---")

st.subheader("Overall Metric Comparison")
metrics_df = build_metrics_table(all_results)
fig5 = plot_metrics_bar(metrics_df)
st.pyplot(fig5, use_container_width=True)