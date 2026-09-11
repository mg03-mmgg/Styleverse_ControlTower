"""
Decision Engine — Replenishment recommendations.

Combines:
  - ML: real demand predictions from your trained XGBoost model
  - Rules/formulas: safety stock theory, inventory gap calculation,
    state classification (ported from the HTML mockup's logic)

For each product-store combination:
  1. Build this week's real features (same shape the model was trained on)
  2. Predict next week's demand using the trained model
  3. Compute safety stock (z x sigma x sqrt(lead_weeks + 1))
  4. Compute the inventory gap (how much to order)
  5. Classify the position (OUT / RISK / HEALTHY)
  6. If at risk, write a real recommendation to ai_recommendation

Populates: ai_demand_forecast, ai_recommendation

Run with:  python build_replenishment_recommendations.py
"""

import os
import pickle
import math
import numpy as np
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "demand_forecast_v1.pkl")

ZTAB = {90: 1.2816, 91: 1.3408, 92: 1.4051, 93: 1.4758, 94: 1.5548,
        95: 1.6449, 96: 1.7507, 97: 1.8808, 98: 2.0537, 99: 2.3263}

SERVICE_LEVEL = 95  # target service level %
CATEGORY_CV = {  # demand variability by category (higher = less predictable)
    "Dress": 0.46, "Shirt": 0.35, "T-Shirt": 0.30, "Trousers": 0.32,
    "Jeans": 0.30, "Jacket": 0.42, "Skirt": 0.44, "Kurta": 0.38,
    "Footwear": 0.28, "Accessories": 0.50,
}


def load_model():
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["feature_cols"]


def load_latest_features(engine):
    """Pulls the most recent ~7 real days per product-location and
    aggregates them EXACTLY the same way train_demand_forecast.py did
    (same column names: sales_7_days_eow, avg_price, total_footfall,
    etc.) — the model can only accept features shaped the way it was
    trained on, not raw daily column names."""
    print("Loading the most recent week of daily features per product-location...")
    query = """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY product_id, location_id ORDER BY date_id DESC) AS rn
            FROM feat_sku_location_day
        )
        SELECT * FROM ranked WHERE rn <= 7
    """
    daily = pd.read_sql(query, engine)
    print(f"  {len(daily)} raw daily rows loaded (last 7 days per position).")

    daily = daily.sort_values(["product_id", "location_id", "date_id"])

    print("Aggregating to match the model's expected weekly feature shape...")
    weekly = daily.groupby(["product_id", "location_id"]).agg(
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
        available_inventory=("available_inventory", "last"),  # keep the RAW current inventory too, for the gap formula
    ).reset_index()

    print("Joining product/location attributes...")
    with engine.connect() as conn:
        products = pd.read_sql("SELECT product_id, brand, category, subcategory, mrp FROM dim_product", conn)
        locations = pd.read_sql("SELECT location_id, cluster_id, region, channel FROM dim_location", conn)

    weekly = weekly.merge(products, on="product_id", how="left")
    weekly = weekly.merge(locations, on="location_id", how="left")

    print(f"  {len(weekly)} product-location positions ready.")
    return weekly


def load_supplier_lead_times(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT product_id, lead_days FROM bridge_product_supplier
            WHERE is_primary_supplier = true
        """))
        return dict(result.fetchall())


def predict_demand(model, feature_cols, df):
    X = df[feature_cols].copy()
    categorical_cols = ["brand", "category", "subcategory", "cluster_id", "region", "channel"]
    for col in categorical_cols:
        X[col] = X[col].astype("category")
    preds = model.predict(X)
    return np.maximum(preds, 0)


def classify_and_build_recommendations(df, lead_time_map):
    z = ZTAB[SERVICE_LEVEL]
    recommendations = []
    forecasts = []

    for _, row in df.iterrows():
        forecast = max(0.05, row["predicted_demand"])
        cv = CATEGORY_CV.get(row["category"], 0.35)
        sigma = forecast * cv

        lead_days = lead_time_map.get(row["product_id"], 30)
        lead_weeks = lead_days / 7

        safety_stock = z * sigma * math.sqrt(lead_weeks + 1)
        on_hand = float(row["available_inventory"])
        cover_weeks = on_hand / forecast if forecast > 0 else 999

        gap = forecast * (lead_weeks + 1) + safety_stock - on_hand

        if on_hand <= 0:
            state = "OUT"
        elif cover_weeks < lead_weeks + 1:
            state = "RISK"
        else:
            state = "HEALTHY"

        forecasts.append({
            "product_id": row["product_id"],
            "location_id": row["location_id"],
            "forecast_quantity": round(float(forecast), 2),
            "lower_estimate": round(max(0, float(forecast - 1.28 * sigma)), 2),
            "upper_estimate": round(float(forecast + 1.28 * sigma), 2),
            "confidence_score": round(max(0.3, min(0.95, 1 - cv)), 2),
            "state": state,
        })

        if state in ("OUT", "RISK") and gap > 0.5:
            qty = math.ceil(gap)
            unit_margin = float(row["mrp"]) * 0.5
            recommendations.append({
                "product_id": row["product_id"],
                "location_id": row["location_id"],
                "recommendation_type": "Replenish",
                "recommended_quantity": qty,
                "recommended_price": None,
                "expected_revenue": round(qty * float(row["mrp"]), 2),
                "expected_margin": round(qty * unit_margin, 2),
                "risk_score": round(min(1.0, 1 - cover_weeks / (lead_weeks + 1)) if (lead_weeks + 1) > 0 else 1.0, 2),
                "recommendation_reason": (
                    f"Cover is {cover_weeks:.1f} weeks against a {lead_weeks+1:.1f} week horizon "
                    f"(lead time {lead_days}d + review). Safety stock {safety_stock:.0f} units at {SERVICE_LEVEL}% service level."
                ),
                "recommendation_status": "Pending",
            })

    return forecasts, recommendations


def insert_forecasts(engine, forecasts, model_version_id):
    print(f"Inserting {len(forecasts)} demand forecasts...")
    run_ts = datetime.now()
    rows = []
    fid_counter = 1
    forecast_id_map = {}

    for f in forecasts:
        fid = f"FC{fid_counter:07d}"
        forecast_id_map[(f["product_id"], f["location_id"])] = fid
        rows.append({
            "forecast_id": fid,
            "model_version_id": model_version_id,
            "run_timestamp": run_ts,
            "product_id": f["product_id"],
            "location_id": f["location_id"],
            "forecast_period": "Next week",
            "forecast_quantity": f["forecast_quantity"],
            "lower_estimate": f["lower_estimate"],
            "upper_estimate": f["upper_estimate"],
            "confidence_score": f["confidence_score"],
        })
        fid_counter += 1

    columns = rows[0].keys()
    col_list = ", ".join(columns)
    val_list = ", ".join(f":{c}" for c in columns)
    stmt = text(f"INSERT INTO ai_demand_forecast ({col_list}) VALUES ({val_list}) ON CONFLICT DO NOTHING")

    batch_size = 2000
    with engine.begin() as conn:
        for i in range(0, len(rows), batch_size):
            conn.execute(stmt, rows[i:i+batch_size])
            print(f"  inserted {min(i+batch_size, len(rows))}/{len(rows)}")

    return forecast_id_map


def insert_recommendations(engine, recommendations, forecast_id_map, model_version_id):
    print(f"Inserting {len(recommendations)} recommendations...")
    rows = []
    rid_counter = 1
    for r in recommendations:
        rid = f"REC{rid_counter:07d}"
        fid = forecast_id_map.get((r["product_id"], r["location_id"]))
        rows.append({
            "recommendation_id": rid,
            "forecast_id": fid,
            "model_version_id": model_version_id,
            "product_id": r["product_id"],
            "location_id": r["location_id"],
            "supplier_id": None,
            "recommendation_type": r["recommendation_type"],
            "recommended_quantity": r["recommended_quantity"],
            "recommended_price": r["recommended_price"],
            "expected_revenue": r["expected_revenue"],
            "expected_margin": r["expected_margin"],
            "risk_score": r["risk_score"],
            "recommendation_reason": r["recommendation_reason"],
            "recommendation_status": r["recommendation_status"],
        })
        rid_counter += 1

    if not rows:
        print("  No recommendations to insert.")
        return

    columns = rows[0].keys()
    col_list = ", ".join(columns)
    val_list = ", ".join(f":{c}" for c in columns)
    stmt = text(f"INSERT INTO ai_recommendation ({col_list}) VALUES ({val_list}) ON CONFLICT DO NOTHING")

    batch_size = 2000
    with engine.begin() as conn:
        for i in range(0, len(rows), batch_size):
            conn.execute(stmt, rows[i:i+batch_size])
            print(f"  inserted {min(i+batch_size, len(rows))}/{len(rows)}")


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    print("Loading trained model...")
    model, feature_cols = load_model()

    df = load_latest_features(engine)
    lead_time_map = load_supplier_lead_times(engine)

    print("Predicting demand for every position...")
    df["predicted_demand"] = predict_demand(model, feature_cols, df)

    print("Applying safety stock / gap / classification logic...")
    forecasts, recommendations = classify_and_build_recommendations(df, lead_time_map)

    state_counts = pd.Series([f["state"] for f in forecasts]).value_counts()
    print(f"\nPosition states:\n{state_counts}")
    print(f"\nGenerated {len(recommendations)} replenishment recommendations.")

    forecast_id_map = insert_forecasts(engine, forecasts, model_version_id="DEMAND_V1")
    insert_recommendations(engine, recommendations, forecast_id_map, model_version_id="DEMAND_V1")

    print("\nDone. ai_demand_forecast and ai_recommendation now populated with real model-driven data.")