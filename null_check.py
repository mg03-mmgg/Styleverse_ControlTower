"""
Pre-training null check — replicates the exact aggregation/join used in
train_demand_forecast.py, then reports null counts per FEATURE COLUMN
(not the whole raw table). This tells you exactly what the model will
actually see, since XGBoost handles NaNs natively but you still want to
know how much missingness exists before trusting the results.

Run with:  python pipeline/check_training_nulls.py
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

load_dotenv()
db_url = os.getenv("DATABASE_URL")


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
        offset += chunk_size
        print(f"  fetched {offset if len(chunk_df) == chunk_size else offset - chunk_size + len(chunk_df)} rows from {table}")

        if len(chunk_df) < chunk_size:
            break  # last page

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def build_training_table(engine):
    print("Loading feat_sku_location_day (chunked)...")
    daily = resilient_read_table(engine, "feat_sku_location_day",
                                   order_by_cols=["date_id", "product_id", "location_id"])
    daily["date"] = pd.to_datetime(daily["date_id"], format="%Y%m%d")
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.dayofweek, unit="D")

    print("Aggregating to weekly...")
    weekly = daily.groupby(["product_id", "location_id", "week_start"]).agg(
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

    print("Loading label_actual_demand (chunked)...")
    labels = resilient_read_table(engine, "label_actual_demand",
                                    order_by_cols=["week_start_date", "product_id", "location_id"])
    labels["week_start"] = pd.to_datetime(labels["week_start_date"])

    print("Joining features with labels...")
    data = weekly.merge(
        labels[["product_id", "location_id", "week_start", "units_sold",
                "stockout_days", "estimated_total_demand", "stockout_flag"]],
        on=["product_id", "location_id", "week_start"], how="inner"
    )

    print("Joining product/location attributes...")
    products = pd.read_sql("SELECT product_id, brand, category, subcategory FROM dim_product", engine)
    locations = pd.read_sql("SELECT location_id, cluster_id, region, channel FROM dim_location", engine)
    data = data.merge(products, on="product_id", how="left")
    data = data.merge(locations, on="location_id", how="left")

    return data


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    data = build_training_table(engine)
    print(f"\nFinal training table: {data.shape[0]} rows, {data.shape[1]} columns.\n")

    print("=== Null counts per column (what XGBoost will actually see) ===")
    nulls = data.isnull().sum()
    pct = (nulls / len(data) * 100).round(2)
    summary = pd.DataFrame({"null_count": nulls, "null_pct": pct})
    summary = summary[summary["null_count"] > 0].sort_values("null_pct", ascending=False)

    if summary.empty:
        print("None — every column is fully populated. Clean to train on.")
    else:
        print(summary.to_string())
        print("\nNote: XGBoost handles NaNs natively and won't crash on these.")
        print("But if any column above is >20-30% null, consider whether that")
        print("feature is reliable enough to trust, or should be dropped/imputed.")

    print("\n=== Target variable sanity check ===")
    print(data["estimated_total_demand"].describe())
    negative_target = (data["estimated_total_demand"] < 0).sum()
    print(f"\nNegative demand values (should be 0): {negative_target}")