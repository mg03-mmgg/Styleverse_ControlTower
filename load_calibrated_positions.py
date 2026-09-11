"""
Loads the 932 externally-sourced, fashion-calibrated positions from
scored_styleverse_positions.csv into feat_sku_location_day.

WHY: the synthetic inventory generator produced badly-calibrated stock
levels (84% of positions sat above 40 weeks of cover), which made the
dashboard's state grid render as a single colour. This CSV's positions
are properly calibrated for fashion retail:
    base_velocity  median 1.45 units/week  (range 0.38 - 2.60)
    on_hand        median 8 units          (range 0 - 82)
  -> cover ratio   median ~5.5 weeks, a realistic fashion retail band

WHAT IT CHANGES: only available_inventory and the sales columns on the
LATEST snapshot date per product-location. Prices, markdown, weather,
trend and everything else are left exactly as they are (markdown in the
CSV is 0 for all rows, so your real fact_price_history values are kept).

Stores are matched by city name. The CSV's 10 stores map onto your 10
real stores; where a city has no direct match (Chennai, Lucknow) it is
assigned to a remaining unused store slot.

Run with:  python load_calibrated_positions.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

CSV_PATH = "scored_styleverse_positions.csv"


def build_store_mapping(engine, csv_stores):
    """Match CSV store names to real location_ids by city, then fill any
    unmatched CSV stores into whatever real stores are still unused."""
    with engine.connect() as conn:
        db_stores = conn.execute(text(
            "SELECT location_id, city FROM dim_location WHERE location_type = 'Store' ORDER BY location_id"
        )).fetchall()

    city_to_id = {city: sid for sid, city in db_stores}
    mapping = {}
    used = set()

    # first pass: exact city match
    for cs in csv_stores:
        city = cs.split("·")[0].strip()
        if city in city_to_id:
            mapping[cs] = city_to_id[city]
            used.add(city_to_id[city])

    # second pass: assign leftovers to unused store slots
    leftover_ids = [sid for sid, _ in db_stores if sid not in used]
    for cs in csv_stores:
        if cs not in mapping and leftover_ids:
            mapping[cs] = leftover_ids.pop(0)

    print("Store mapping:")
    for cs, sid in mapping.items():
        print(f"  {cs:<30} -> {sid}")
    return mapping


def build_product_mapping(engine, csv_skus):
    """The CSV's product names (Ridge Parka, Vesper Midi Dress...) don't
    exist in dim_product, so map them positionally onto the real top
    products already used in feat_sku_location_day."""
    with engine.connect() as conn:
        db_products = [r[0] for r in conn.execute(text("""
            SELECT DISTINCT product_id FROM feat_sku_location_day ORDER BY product_id
        """))]

    unique_csv_skus = sorted(set(csv_skus))
    print(f"\n{len(unique_csv_skus)} unique CSV products -> {len(db_products)} real products in the feature table")

    mapping = {}
    for i, sku in enumerate(unique_csv_skus):
        if i < len(db_products):
            mapping[sku] = db_products[i]
    print(f"  Mapped {len(mapping)} products positionally.")
    return mapping


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"{CSV_PATH} not found — run train_on_real_dataset.py first.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} calibrated positions from {CSV_PATH}\n")

    store_map = build_store_mapping(engine, df["store_name"].unique())
    product_map = build_product_mapping(engine, df["sku_name"].tolist())

    with engine.connect() as conn:
        latest_date = conn.execute(text(
            "SELECT MAX(date_id) FROM feat_sku_location_day"
        )).scalar()
    print(f"\nUpdating the latest snapshot date: {latest_date}")

    updates = []
    skipped = 0
    for _, row in df.iterrows():
        pid = product_map.get(row["sku_name"])
        lid = store_map.get(row["store_name"])
        if not pid or not lid:
            skipped += 1
            continue

        weekly_velocity = float(row["base_velocity"])
        updates.append({
            "date_id": latest_date,
            "product_id": pid,
            "location_id": lid,
            "available_inventory": int(row["on_hand"]),
            "sales_7_days": round(weekly_velocity, 2),
            "sales_28_days": round(weekly_velocity * 4, 2),
            "days_of_inventory": round(int(row["on_hand"]) / weekly_velocity * 7, 1) if weekly_velocity > 0 else 999,
        })

    print(f"Prepared {len(updates)} updates ({skipped} skipped — no mapping)")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE _calibrated (
                date_id VARCHAR(10), product_id VARCHAR(20), location_id VARCHAR(20),
                available_inventory INTEGER, sales_7_days NUMERIC(10,2),
                sales_28_days NUMERIC(10,2), days_of_inventory NUMERIC(6,2)
            ) ON COMMIT DROP
        """))
        conn.execute(text("""
            INSERT INTO _calibrated
            (date_id, product_id, location_id, available_inventory, sales_7_days, sales_28_days, days_of_inventory)
            VALUES (:date_id, :product_id, :location_id, :available_inventory,
                    :sales_7_days, :sales_28_days, :days_of_inventory)
        """), updates)

        result = conn.execute(text("""
            UPDATE feat_sku_location_day f
            SET available_inventory = c.available_inventory,
                sales_7_days = c.sales_7_days,
                sales_28_days = c.sales_28_days,
                days_of_inventory = c.days_of_inventory
            FROM _calibrated c
            WHERE f.date_id = c.date_id
              AND f.product_id = c.product_id
              AND f.location_id = c.location_id
        """))
        print(f"Updated {result.rowcount} rows in feat_sku_location_day.")

    # show the resulting cover distribution, so you can see immediately
    # whether the grid will now have real variety
    with engine.connect() as conn:
        dist = conn.execute(text("""
            SELECT
              CASE
                WHEN available_inventory = 0 THEN '0 (OUT)'
                WHEN available_inventory / GREATEST(sales_28_days/4.0, 0.3) < 4 THEN '1-4 weeks'
                WHEN available_inventory / GREATEST(sales_28_days/4.0, 0.3) < 12 THEN '4-12 weeks'
                WHEN available_inventory / GREATEST(sales_28_days/4.0, 0.3) < 24 THEN '12-24 weeks'
                WHEN available_inventory / GREATEST(sales_28_days/4.0, 0.3) < 40 THEN '24-40 weeks'
                ELSE '40+ weeks'
              END AS bucket,
              COUNT(*) AS n
            FROM feat_sku_location_day
            WHERE date_id = :d
            GROUP BY 1 ORDER BY 1
        """), {"d": latest_date}).fetchall()

    print("\nCover distribution on the latest snapshot (what the grid will show):")
    for bucket, n in dist:
        print(f"  {bucket:<14} {n}")

    print("\nDone. Restart uvicorn and refresh the dashboard.")