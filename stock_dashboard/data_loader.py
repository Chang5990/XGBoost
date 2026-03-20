import os
import glob
import pandas as pd


def load_single_result(results_dir: str, stock_name: str) -> pd.DataFrame:
    """
    Read the prediction result of a single stock
   File name format：AAPL_predictions.csv
    """
    file_path = os.path.join(results_dir, f"{stock_name}_predictions.csv")
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_all_results(results_dir: str) -> dict:
    """
   Read all *_predictions.csv files in the results folder
  Return:
        {
            "AAPL": df,
            "MSFT": df,
            ...
        }
    """
    pattern = os.path.join(results_dir, "*_predictions.csv")
    files = glob.glob(pattern)

    results = {}
    for file_path in files:
        file_name = os.path.basename(file_path)
        stock_name = file_name.replace("_predictions.csv", "")

        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        results[stock_name] = df

    return results