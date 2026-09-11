"""
Generates fact_external_signal — weather, festival, trend, and local
event signals per date/region/category, linked to real products and
locations where relevant.

~2,000-2,500 rows across 180 days.

Run with:  python generate_fact_external_signal.py
"""

import os
import time
import random
from datetime import date, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(66)

START = date(2026, 3, 1)
TOTAL_DAYS = 180

SIGNAL_TYPES = ["Weather", "Festival", "Trend", "Event"]
SOURCES = ["Weather API", "Social Listening", "Google Trends", "Local Events Calendar"]


def fetch_locations_and_products(engine):
    with engine.connect() as conn:
        locations = conn.execute(text(
            "SELECT location_id, region FROM dim_location WHERE region IS NOT NULL"
        )).fetchall()
        top_products = [r[0] for r in conn.execute(text("""
            SELECT product_id FROM fact_sales_line
            GROUP BY product_id ORDER BY SUM(units_sold) DESC LIMIT 100
        """))]
        categories = [r[0] for r in conn.execute(text(
            "SELECT DISTINCT category FROM dim_product"
        ))]
    return locations, top_products, categories


def generate_signals(locations, top_products, categories):
    dates = [START + timedelta(days=i) for i in range(TOTAL_DAYS)]
    rows = []
    sid = 1

    for d in dates:
        n_signals = random.randint(10, 16)
        for _ in range(n_signals):
            stype = random.choice(SIGNAL_TYPES)
            loc = random.choice(locations) if locations else (None, None)
            # some signals are product-specific (trend), others are broader (weather/festival)
            product_id = random.choice(top_products) if stype == "Trend" and random.random() < 0.6 else None

            rows.append({
                "signal_id": f"SIG{sid:06d}",
                "date_id": d.strftime("%Y%m%d"),
                "location_id": loc[0],
                "product_id": product_id,
                "category": random.choice(categories) if categories else None,
                "signal_type": stype,
                "signal_value": round(random.uniform(0.2, 1.0), 2),
                "source": random.choice(SOURCES),
                "quality_score": round(random.uniform(0.6, 0.98), 2),
            })
            sid += 1

    return rows


def insert_rows_batched(engine, table, rows, batch_size=1000, max_retries=5):
    if not rows:
        print(f"No rows for {table}.")
        return

    columns = rows[0].keys()
    col_list = ", ".join(columns)
    val_list = ", ".join(f":{c}" for c in columns)
    stmt = text(f"INSERT INTO {table} ({col_list}) VALUES ({val_list})")

    total = len(rows)
    i = 0
    while i < total:
        batch = rows[i:i + batch_size]
        attempt = 0
        while attempt < max_retries:
            try:
                with engine.begin() as conn:
                    conn.execute(stmt, batch)
                print(f"  inserted {min(i + batch_size, total)}/{total}")
                break
            except OperationalError as e:
                attempt += 1
                wait = min(30, 5 * attempt)
                print(f"  batch failed (attempt {attempt}/{max_retries}): {e}")
                if attempt >= max_retries:
                    raise
                engine.dispose()
                time.sleep(wait)
        i += batch_size

    print(f"Inserted {total} rows into {table}")


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    locations, top_products, categories = fetch_locations_and_products(engine)
    print(f"Loaded {len(locations)} locations, {len(top_products)} top products, {len(categories)} categories.")

    rows = generate_signals(locations, top_products, categories)
    print(f"Generated {len(rows)} external signal rows.")

    insert_rows_batched(engine, "fact_external_signal", rows)

    print("Done.")