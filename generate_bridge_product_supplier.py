"""
Generates bridge_product_supplier by reading the ACTUAL products already
inserted into dim_product (so it never invents product_ids that don't
exist), and links each one to 1-2 of the 5 suppliers with a unit_cost
derived from that product's real unit_cost (+/- variation), not a random
unrelated number.

Run with:  python generate_bridge_product_supplier.py
"""

import os
import random
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(21)

SUPPLIER_IDS = ["SUP001", "SUP002", "SUP003", "SUP004", "SUP005"]


def fetch_products(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT product_id, unit_cost FROM dim_product"))
        return [(row[0], float(row[1])) for row in result]


def generate_bridge(products):
    rows = []
    for pid, base_cost in products:
        # each product gets 1 primary supplier, 40% chance of a second backup supplier
        chosen = random.sample(SUPPLIER_IDS, k=1 if random.random() > 0.4 else 2)
        for idx, sid in enumerate(chosen):
            is_primary = (idx == 0)
            # supplier's cost varies +/-15% around the product's actual unit_cost
            variation = random.uniform(0.85, 1.15)
            supplier_cost = round(base_cost * variation, 2)

            rows.append({
                "product_id": pid,
                "supplier_id": sid,
                "is_primary_supplier": is_primary,
                "unit_cost": supplier_cost,
                "lead_days": random.randint(15, 45) if is_primary else random.randint(25, 55),
                "minimum_order_qty": random.choice([200, 300, 500, 750]),
                "capacity_share": round(random.uniform(0.5, 1.0), 2) if is_primary
                                   else round(random.uniform(0.1, 0.4), 2),
            })
    return rows


def insert_rows(engine, table, rows):
    if not rows:
        return
    columns = rows[0].keys()
    col_list = ", ".join(columns)
    val_list = ", ".join(f":{c}" for c in columns)
    stmt = text(f"INSERT INTO {table} ({col_list}) VALUES ({val_list}) ON CONFLICT DO NOTHING")
    with engine.begin() as conn:
        conn.execute(stmt, rows)
    print(f"Inserted {len(rows)} rows into {table}")


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)


    products = fetch_products(engine)
    if not products:
        raise SystemExit("No rows found in dim_product — run the product generator first.")

    print(f"Found {len(products)} products in dim_product.")

    bridge_rows = generate_bridge(products)
    insert_rows(engine, "bridge_product_supplier", bridge_rows)

    print("Done.")