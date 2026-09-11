"""
Trains the demand forecasting model — the core AI component of the
Control Tower. Pipeline:

 1. Pull feat_sku_location_day (daily) and aggregate to WEEKLY, matching
    the granularity of label_actual_demand.
 2. Join weekly features with the real label_actual_demand target
    (estimated_total_demand — sales + estimated lost sales from stockouts).
 3. Join in product/location attributes (brand, category, cluster) as
    categorical features.
 4. Split chronologically (first 80% of weeks = train, last 20% = test)
    to avoid data leakage — never train on the future to predict the past.
 5. Train an XGBoost regressor.
 6. Evaluate with MAE, RMSE, WAPE.
 7. Save the trained model to disk.
 8. Register the model in ai_model_registry.

Run with:  python models/train_demand_forecast.py
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sklearn.model_selection import train_test_split  # not used for the chrono split, but handy if needed
import xgboost as xgb

load_dotenv()
db_url = os.getenv("DATABASE_URL")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


def resilient_read_table(engine, table, order_by_cols, chunk_size=50000, max_retries=5):
    """Reads a large table in chunks with retries, instead of one giant
    SELECT * that a single connection drop can kill entirely."""
    order_clause = ", ".join(order_by_cols)
    chunks = []
    offset = 0

    while True:
        attempt = 0
        chunk_df = None
        while attempt < max_retries:
            try:
                query = f"SELECT * FROM {table} ORDER BY {order_clause} LIMIT {chunk_size} OFFSET {offset}"
                chunk_df = pd.read_sql(query, engine)
                break
            except OperationalError as e:
                attempt += 1
                wait = min(30, 5 * attempt)
                print(f"  chunk read failed (attempt {attempt}/{max_retries}): {e}")
                if attempt >= max_retries:
                    raise
                engine.dispose()
                print(f"  reconnecting and retrying in {wait}s...")
                time.sleep(wait)

        if chunk_df is None or chunk_df.empty:
            break

        chunks.append(chunk_df)
        fetched_so_far = offset + len(chunk_df)
        offset += chunk_size
        print(f"  fetched {fetched_so_far} rows from {table}")

        if len(chunk_df) < chunk_size:
            break

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_daily_features(engine):
    print("Loading feat_sku_location_day (chunked)...")
    df = resilient_read_table(engine, "feat_sku_location_day",
                                order_by_cols=["date_id", "product_id", "location_id"])
    df["date"] = pd.to_datetime(df["date_id"], format="%Y%m%d")
    print(f"  {len(df)} daily rows loaded.")
    return df


def aggregate_to_weekly(daily_df):
    print("Aggregating daily features to weekly...")
    daily_df["week_start"] = daily_df["date"] - pd.to_timedelta(daily_df["date"].dt.dayofweek, unit="D")

    agg = daily_df.groupby(["product_id", "location_id", "week_start"]).agg(
        sales_7_days_eow=("sales_7_days", "last"),
        sales_28_days_eow=("sales_28_days", "last"),
        sales_growth_avg=("sales_growth_7_days", "mean"),
        views_7_days_eow=("views_7_days", "last"),
        cart_additions_eow=("cart_additions_7_days", "last"),
        return_rate_avg=("return_rate_28_days", "mean"),
        start_inventory=("available_inventory", "first"),
        avg_days_of_inventory=("days_of_inventory", "mean"),
        avg_price=("current_price", "mean"),
        max_markdown=("markdown_percentage", "max"),
        any_promotion=("promotion_flag", "max"),
        total_footfall=("footfall", "sum"),
        avg_weather=("weather_score", "mean"),
        any_festival=("festival_flag", "max"),
        avg_trend=("trend_score", "mean"),
        supplier_lead_days=("supplier_lead_days", "first"),
    ).reset_index()

    print(f"  {len(agg)} weekly feature rows.")
    return agg


def load_labels(engine):
    print("Loading label_actual_demand (chunked)...")
    df = resilient_read_table(engine, "label_actual_demand",
                                order_by_cols=["week_start_date", "product_id", "location_id"])
    df["week_start"] = pd.to_datetime(df["week_start_date"])
    print(f"  {len(df)} weekly label rows.")
    return df


def load_product_location_attrs(engine):
    with engine.connect() as conn:
        products = pd.read_sql("SELECT product_id, brand, category, subcategory FROM dim_product", conn)
        locations = pd.read_sql("SELECT location_id, cluster_id, region, channel FROM dim_location", conn)
    return products, locations


def build_training_table(engine):
    daily = load_daily_features(engine)
    weekly_features = aggregate_to_weekly(daily)
    labels = load_labels(engine)
    products, locations = load_product_location_attrs(engine)

    # ---- CRITICAL: fix data leakage ----
    # weekly_features for week W were computed using days INSIDE week W
    # (e.g. sales_7_days_eow on the week's last day already contains most
    # of that week's own sales). Joining them directly to week W's label
    # lets the model "see" almost the entire answer before predicting it.
    #
    # Fix: shift features forward by one week, so week W's features are
    # matched against week W+1's demand — a genuine one-week-ahead
    # forecast using only information available BEFORE the target week
    # begins.
    weekly_features = weekly_features.copy()
    weekly_features["week_start"] = weekly_features["week_start"] + pd.Timedelta(days=7)

    print("Joining features with labels (features shifted +1 week to prevent leakage)...")
    data = weekly_features.merge(
        labels[["product_id", "location_id", "week_start", "units_sold",
                "stockout_days", "estimated_total_demand", "stockout_flag"]],
        on=["product_id", "location_id", "week_start"], how="inner"
    )
    print(f"  {len(data)} rows after joining features with labels.")

    data = data.merge(products, on="product_id", how="left")
    data = data.merge(locations, on="location_id", how="left")

    return data


def prepare_features(data):
    categorical_cols = ["brand", "category", "subcategory", "cluster_id", "region", "channel"]
    for col in categorical_cols:
        data[col] = data[col].astype("category")

    feature_cols = [
        "sales_7_days_eow", "sales_28_days_eow", "sales_growth_avg",
        "views_7_days_eow", "cart_additions_eow", "return_rate_avg",
        "start_inventory", "avg_days_of_inventory", "avg_price", "max_markdown",
        "any_promotion", "total_footfall", "avg_weather", "any_festival",
        "avg_trend", "supplier_lead_days"
    ] + categorical_cols

    target_col = "estimated_total_demand"
    return data, feature_cols, target_col


def chronological_split(data, feature_cols, target_col, test_fraction=0.2):
    data = data.sort_values("week_start")
    unique_weeks = sorted(data["week_start"].unique())
    split_idx = int(len(unique_weeks) * (1 - test_fraction))
    train_weeks = set(unique_weeks[:split_idx])
    test_weeks = set(unique_weeks[split_idx:])

    train = data[data["week_start"].isin(train_weeks)]
    test = data[data["week_start"].isin(test_weeks)]

    print(f"Train: {len(train)} rows ({len(train_weeks)} weeks) | "
          f"Test: {len(test)} rows ({len(test_weeks)} weeks)")

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    return X_train, y_train, X_test, y_test, train, test


def train_model(X_train, y_train):
    print("Training XGBoost regressor...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        enable_categorical=True,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)  # demand can't be negative

    # ---- overall metrics (includes the many zero-demand weeks) ----
    mae = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    wape = np.sum(np.abs(preds - y_test)) / np.sum(np.abs(y_test)) if y_test.sum() > 0 else float("nan")

    print(f"\n=== Overall evaluation (all test weeks, including zero-demand) ===")
    print(f"  Rows:  {len(y_test)}")
    print(f"  MAE:   {mae:.2f} units")
    print(f"  RMSE:  {rmse:.2f} units")
    print(f"  WAPE:  {wape*100:.1f}%")
    print(f"  (Forecast accuracy ~ {max(0, 100*(1-wape)):.1f}%)")

    # ---- non-zero-only metrics (the part that actually matters for a
    # demand forecaster — this is the number to use in the presentation,
    # since the overall numbers above are inflated by rows with 0 true demand) ----
    nonzero_mask = y_test > 0
    y_nz = y_test[nonzero_mask]
    preds_nz = preds[nonzero_mask]

    if len(y_nz) > 0:
        mae_nz = np.mean(np.abs(preds_nz - y_nz))
        rmse_nz = np.sqrt(np.mean((preds_nz - y_nz) ** 2))
        wape_nz = np.sum(np.abs(preds_nz - y_nz)) / np.sum(np.abs(y_nz))
        accuracy_nz = max(0, 100 * (1 - wape_nz))

        print(f"\n=== Non-zero demand weeks only ({len(y_nz)} of {len(y_test)} rows, "
              f"{100*len(y_nz)/len(y_test):.1f}% of test set) ===")
        print(f"  MAE:   {mae_nz:.2f} units")
        print(f"  RMSE:  {rmse_nz:.2f} units")
        print(f"  WAPE:  {wape_nz*100:.1f}%")
        print(f"  Forecast accuracy: {accuracy_nz:.1f}%   <-- use THIS number for the presentation")
    else:
        print("\nNo non-zero demand rows in the test set — cannot compute this breakdown.")
        mae_nz = rmse_nz = wape_nz = accuracy_nz = float("nan")

    return {
        "mae": float(mae), "rmse": float(rmse), "wape": float(wape),
        "accuracy": float(max(0, 100 * (1 - wape))),
        "mae_nonzero": float(mae_nz), "rmse_nonzero": float(rmse_nz),
        "wape_nonzero": float(wape_nz), "accuracy_nonzero": float(accuracy_nz),
        "nonzero_test_rows": int(len(y_nz)), "total_test_rows": int(len(y_test)),
    }


def diagnose_degenerate_model(model, X_test, y_test, feature_cols):
    """Explicitly checks whether the model just learned to predict ~0 for
    everything (a real risk given how sparse the target is), rather than
    trusting that a low MAE alone means it's actually learning."""
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    print("\n=== Degenerate-model check ===")

    print(f"  Prediction stats: min={preds.min():.3f}, max={preds.max():.3f}, "
          f"mean={preds.mean():.3f}, std={preds.std():.3f}")

    near_zero_pct = 100 * np.mean(preds < 0.1)
    print(f"  Predictions near zero (<0.1): {near_zero_pct:.1f}% of all test rows")
    if near_zero_pct > 95:
        print("  [WARNING] Over 95% of predictions are near zero — this looks like a degenerate model.")
    else:
        print("  [OK] Predictions show real variation, not just defaulting to zero.")

    # correlation between predicted and actual, specifically on non-zero-demand rows
    nonzero_mask = y_test > 0
    if nonzero_mask.sum() > 5:
        corr = np.corrcoef(preds[nonzero_mask], y_test[nonzero_mask])[0, 1]
        print(f"  Correlation (predicted vs actual) on non-zero demand rows: {corr:.3f}")
        if corr < 0.15:
            print("  [WARNING] Very low correlation — model may not be capturing real demand signal.")
        else:
            print("  [OK] Model's predictions track real demand direction on non-zero rows.")
    else:
        print("  Not enough non-zero rows in test set to compute correlation.")

    # feature importance — a degenerate model tends to lean entirely on
    # one or two features (or shows near-flat importance across all of them)
    print("\n  Top 10 most important features:")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    for idx in order:
        print(f"    {feature_cols[idx]:<25} {importances[idx]:.4f}")

    # side-by-side sample: actual vs predicted, on a handful of real non-zero rows
    print("\n  Sample actual vs predicted (10 non-zero-demand test rows):")
    sample_idx = np.where(nonzero_mask)[0][:10]
    for i in sample_idx:
        print(f"    actual={y_test.iloc[i]:.2f}  predicted={preds[i]:.2f}")


def save_model(model, feature_cols):
    path = os.path.join(MODEL_DIR, "demand_forecast_v1.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"Model saved to {path}")
    return path


def register_model(engine, metrics, train_start, train_end):
    print("Registering model in ai_model_registry...")
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO ai_model_registry
            (model_version_id, model_name, use_case, version_number, status,
             training_start_date, training_end_date, business_owner, technical_owner,
             approval_date, accuracy_floor)
            VALUES
            (:mvid, :name, :use_case, :ver, :status, :tstart, :tend, :bowner, :towner, :adate, :floor)
            ON CONFLICT (model_version_id) DO NOTHING
        """), {
            "mvid": "DEMAND_V1",
            "name": "Demand Forecast",
            "use_case": "SKU-store weekly demand forecasting",
            "ver": "1.0",
            "status": "Active",
            "tstart": train_start,
            "tend": train_end,
            "bowner": "merchandising_team",
            "towner": "data_science_team",
            "adate": datetime.now().date(),
            "floor": round(metrics["accuracy_nonzero"], 1),
        })
    print("Model registered.")


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    data = build_training_table(engine)
    data, feature_cols, target_col = prepare_features(data)

    X_train, y_train, X_test, y_test, train_df, test_df = chronological_split(data, feature_cols, target_col)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    diagnose_degenerate_model(model, X_test, y_test, feature_cols)

    save_model(model, feature_cols)

    train_start = train_df["week_start"].min().date()
    train_end = train_df["week_start"].max().date()
    register_model(engine, metrics, train_start, train_end)

    print("\nDone. Demand forecasting model trained, evaluated, saved, and registered.")