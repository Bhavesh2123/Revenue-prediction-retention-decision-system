import sys
import os
from pathlib import Path

# FIX: BASE_DIR must resolve correctly whether this file lives in root or scripts/
# Using __file__ here is correct — but the scripts/ version should use parent.parent
BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import joblib

from src.data_preprocessing import load_data, add_total_price, time_based_split, auto_detect_dates
from src.feature_engineering import (
    build_rfm,
    build_additional_features,
    build_future_revenue,      # FIX: was build_Future_Revenue (wrong capitalisation)
    merge_all_features,
)
from src.ltv_model import train_model


def run_revenue_training(data_path=None, cutoff_date=None, prediction_end=None):
    """
    Full revenue model training pipeline.

    Args:
        data_path: Path to CSV. Defaults to Dataset/Raw_Data.csv.
        cutoff_date: pd.Timestamp for end of observation window.
                     If None, auto-detected from data.
        prediction_end: pd.Timestamp for end of prediction window.
                        If None, auto-detected from data.

    Returns:
        tuple: (best_model, metrics)
    """
    if data_path is None:
        data_path = BASE_DIR / "Dataset" / "Raw_Data.csv"

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load_data(data_path)
    df = add_total_price(df)

    # FIX: auto-detect dates so real client data works — no more hardcoded 2011 dates
    if cutoff_date is None or prediction_end is None:
        cutoff_date, prediction_end = auto_detect_dates(df, obs_months=6)

    past_data, future_data = time_based_split(df, cutoff_date, prediction_end)

    rfm = build_rfm(past_data, cutoff_date)
    extra_features = build_additional_features(past_data, cutoff_date)
    future_revenue = build_future_revenue(future_data)
    model_df = merge_all_features(rfm, extra_features, future_revenue)

    # FIX: train_model returns (model, metrics) — must unpack both
    model, metrics = train_model(model_df)

    model_path = BASE_DIR / "models" / "revenue_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # FIX: save only the model, not the (model, metrics) tuple
    joblib.dump(model, model_path)
    print(f"[revenue] Model saved to {model_path}")

    # FIX: was a plain string missing the f prefix — metrics['R2'] was never evaluated
    print(f"Revenue training complete. R² = {metrics['R2']}")

    return model, metrics


if __name__ == "__main__":
    run_revenue_training()
