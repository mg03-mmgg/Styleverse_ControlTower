"""
Benchmark Validation — proves your modeling METHODOLOGY generalizes
beyond your own synthetic StyleVerse data, using the REAL, publicly
available Rossmann Store Sales dataset (1,115 real German drugstores,
~1M real daily sales records, 2013-2015).

This directly answers PwC's feedback: instead of merging unrelated
company data into your own tables (which wouldn't make sense — different
products, currency, scale), this trains the SAME pipeline (XGBoost,
chronological split, same evaluation approach) on real external data and
reports the accuracy as independent proof the approach works.

BEFORE RUNNING:
  1. Download train.csv from a Kaggle Rossmann Store Sales dataset page
     (not the competition page, to skip rules-acceptance):
     e.g. https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales
  2. Place it at:  benchmark_data/rossmann_train.csv

Run with:  python benchmark_validation.py
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb

DATA_PATH = os.path.join(os.path.dirname(__file__), "benchmark_data", "train.csv")


def load_and_clean(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df)} raw rows loaded.")

    df["Date"] = pd.to_datetime(df["Date"])
    # only open days have meaningful sales — matches real business logic
    df = df[df["Open"] == 1].copy()
    df = df[df["Sales"] > 0]
    print(f"  {len(df)} rows after filtering to open days with sales.")

    return df


def engineer_features(df):
    print("Engineering features...")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # sort for lag features (same pattern as your StyleVerse pipeline —
    # only use PAST information to predict, no leakage)
    df = df.sort_values(["Store", "Date"])
    df["Sales_lag_7"] = df.groupby("Store")["Sales"].shift(7)
    df["Sales_rolling_7"] = df.groupby("Store")["Sales"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )

    df["StateHoliday"] = df["StateHoliday"].astype(str).astype("category")
    df["Store"] = df["Store"].astype("category")

    df = df.dropna(subset=["Sales_lag_7", "Sales_rolling_7"])
    print(f"  {len(df)} rows after feature engineering (some early rows dropped for lag features).")
    return df


def chronological_split(df, test_fraction=0.2):
    df = df.sort_values("Date")
    unique_dates = sorted(df["Date"].unique())
    split_idx = int(len(unique_dates) * (1 - test_fraction))
    split_date = unique_dates[split_idx]

    train = df[df["Date"] < split_date]
    test = df[df["Date"] >= split_date]
    print(f"Train: {len(train)} rows (before {split_date.date()}) | "
          f"Test: {len(test)} rows (from {split_date.date()})")
    return train, test


def train_and_evaluate(train, test, feature_cols, target_col="Sales"):
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    print("Training XGBoost (same methodology as StyleVerse model)...")
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", enable_categorical=True, random_state=42,
    )
    model.fit(X_train, y_train)

    preds = np.maximum(model.predict(X_test), 0)
    mae = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    wape = np.sum(np.abs(preds - y_test)) / np.sum(np.abs(y_test))

    print(f"\n=== Benchmark results (Rossmann Store Sales, real data) ===")
    print(f"  Test rows: {len(y_test)}")
    print(f"  MAE:  {mae:.1f} units")
    print(f"  RMSE: {rmse:.1f} units")
    print(f"  WAPE: {wape*100:.1f}%")
    print(f"  Forecast accuracy: {max(0, 100*(1-wape)):.1f}%")
    print(f"\n  (Published Rossmann Kaggle solutions typically achieve ~10-15% RMSPE "
          f"with more extensive feature engineering — this is a fair, comparable "
          f"same-methodology run, not a leaderboard attempt.)")

    print("\n  Top 10 feature importances:")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    for idx in order:
        print(f"    {feature_cols[idx]:<20} {importances[idx]:.4f}")

    return {"mae": mae, "rmse": rmse, "wape": wape, "accuracy": max(0, 100*(1-wape))}


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"File not found: {DATA_PATH}\n"
            "Download rossmann train.csv from Kaggle and place it there first."
        )

    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)

    feature_cols = ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
                     "Month", "WeekOfYear", "Sales_lag_7", "Sales_rolling_7"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    train, test = chronological_split(df)
    metrics = train_and_evaluate(train, test, feature_cols)

    print("\nDone. Use this accuracy figure alongside your StyleVerse model's "
          "accuracy in the presentation as independent, real-data validation "
          "that the methodology generalizes.")
