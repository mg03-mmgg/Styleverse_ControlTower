"""
REBUILT inventory generator — restock quantities are now PROPORTIONAL to
each product-location's REAL average weekly demand (computed from actual
fact_sales_line data), instead of flat random 20-80 unit batches applied
regardless of how fast something actually sells.

This fixes the root cause of the 84% "40+ weeks of cover" overstock skew
found in the previous version — that wasn't a display bug, it was a real
consequence of restocking slow and fast movers with the same random
quantities. Real inventory policy scales replenishment to demand; this
version does too.

Run with:  python generate_fact_inventory_snapshot.py
(TRUNCATE TABLE fact_inventory_snapshot; first — this REPLACES the
previous version's data.)
"""

import os
import io
import csv
import time
import random
from datetime import date, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(77)
TOP_N_PRODUCTS = 250


# ------------------------------------------------------------------
# COPY-based bulk insert
# ------------------------------------------------------------------

def records_to_csv_buffer(records, columns):
    buf = io.StringIO()
    writer = csv.writer(buf)
    for r in records:
        row = []
        for col in columns:
            v = r[col]
            if v is None:
                row.append("")
            elif isinstance(v, bool):
                row.append("true" if v else "false")
            else:
                row.append(str(v))
        writer.writerow(row)
    buf.seek(0)
    return buf


def copy_insert_via_staging(engine, table, records, columns, conflict_cols, chunk_size=20000, max_retries=5):
    if not records:
        print(f"No records for {table}.")
        return
    total = len(records)
    col_list = ", ".join(columns)
    staging_table = f"_staging_{table}"

    i = 0
    while i < total:
        chunk = records[i:i + chunk_size]
        attempt = 0
        while attempt < max_retries:
            try:
                raw_conn = engine.raw_connection()
                try:
                    cur = raw_conn.cursor()
                    cur.execute(f"CREATE TEMP TABLE {staging_table} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
                    csv_buf = records_to_csv_buffer(chunk, columns)
                    cur.copy_expert(
                        f"COPY {staging_table} ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '')",
                        csv_buf
                    )
                    cur.execute(
                        f"INSERT INTO {table} ({col_list}) "
                        f"SELECT {col_list} FROM {staging_table} "
                        f"ON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING"
                    )
                    raw_conn.commit()
                    cur.close()
                finally:
                    raw_conn.close()
                print(f"  inserted {min(i + chunk_size, total)}/{total}")
                break
            except Exception as e:
                attempt += 1
                wait = min(30, 5 * attempt)
                print(f"  chunk failed (attempt {attempt}/{max_retries}): {e}")
                if attempt >= max_retries:
                    raise
                engine.dispose()
                print(f"  reconnecting and retrying in {wait}s...")
                time.sleep(wait)
        i += chunk_size

    print(f"Done inserting into {table} (via COPY, conflicts skipped).")


# ------------------------------------------------------------------
# generation logic — DEMAND-PROPORTIONAL
# ------------------------------------------------------------------

def fetch_top_products_and_locations(engine):
    with engine.connect() as conn:
        products = [r[0] for r in conn.execute(text("""
            SELECT product_id FROM fact_sales_line
            GROUP BY product_id ORDER BY SUM(units_sold) DESC LIMIT :n
        """), {"n": TOP_N_PRODUCTS})]
        locations = [r[0] for r in conn.execute(text(
            "SELECT location_id FROM dim_location WHERE location_type != 'Online'"
        ))]
    return products, locations


def fetch_daily_sales_and_weekly_rate(engine, products):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT product_id, location_id, date_id, SUM(units_sold)
            FROM fact_sales_line
            WHERE product_id = ANY(:products)
            GROUP BY product_id, location_id, date_id
        """), {"products": products})
        rows = result.fetchall()

    sales_map = {}
    total_by_pl = {}
    for pid, lid, date_id, total in rows:
        sales_map[(pid, lid, date_id)] = int(total)
        key = (pid, lid)
        total_by_pl[key] = total_by_pl.get(key, 0) + int(total)

    weekly_rate = {key: max(0.3, total / (180 / 7)) for key, total in total_by_pl.items()}
    return sales_map, weekly_rate


def generate_snapshots(products, locations, sales_map, weekly_rate,
                        start=date(2026, 3, 1), total_days=180, step=14):
    snapshot_dates = [start + timedelta(days=i) for i in range(0, total_days, step)]
    rows = []

    for pid in products:
        for lid in locations:
            demand = weekly_rate.get((pid, lid), 0.3)

            target_cover = random.uniform(3, 7)
            on_hand = round(demand * random.uniform(2, 4))
            safety_stock = round(demand * random.uniform(1, 2))
            restock_target = round(demand * target_cover)

            last_date = start

            for snap_date in snapshot_dates:
                period_sales = 0
                d = last_date
                while d <= snap_date:
                    period_sales += sales_map.get((pid, lid, d.strftime("%Y%m%d")), 0)
                    d += timedelta(days=1)

                on_hand = max(0, on_hand - period_sales)

                if on_hand < safety_stock and random.random() < 0.7:
                    on_hand = max(on_hand, restock_target)
                elif random.random() < 0.15:
                    on_hand += round(demand * random.uniform(1, 3))

                reserved = random.randint(0, min(3, on_hand))
                available = max(0, on_hand - reserved)
                in_transit = round(demand * random.uniform(0, 1)) if random.random() < 0.2 else 0
                damaged = random.randint(0, 2) if random.random() < 0.08 else 0

                rows.append({
                    "snapshot_timestamp": f"{snap_date.isoformat()} 08:00:00",
                    "date_id": snap_date.strftime("%Y%m%d"),
                    "product_id": pid,
                    "location_id": lid,
                    "on_hand_quantity": on_hand,
                    "reserved_quantity": reserved,
                    "available_quantity": available,
                    "in_transit_quantity": in_transit,
                    "damaged_quantity": damaged,
                    "safety_stock": safety_stock,
                })

                last_date = snap_date + timedelta(days=1)

    return rows


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    products, locations = fetch_top_products_and_locations(engine)
    print(f"Loaded top {len(products)} products, {len(locations)} inventory-holding locations.")

    print("Computing real weekly demand rate per product-location...")
    sales_map, weekly_rate = fetch_daily_sales_and_weekly_rate(engine, products)
    print(f"Loaded {len(sales_map)} sales aggregates, {len(weekly_rate)} demand rates.")

    if not products or not locations:
        raise SystemExit("Missing master data — run the earlier generator scripts first.")

    rows = generate_snapshots(products, locations, sales_map, weekly_rate)
    print(f"Generated {len(rows)} inventory snapshot rows.")

    stockouts = sum(1 for r in rows if r["available_quantity"] == 0)
    print(f"Stockout snapshots: {stockouts} ({100*stockouts/len(rows):.1f}%)")

    cols = ["snapshot_timestamp", "date_id", "product_id", "location_id", "on_hand_quantity",
            "reserved_quantity", "available_quantity", "in_transit_quantity", "damaged_quantity", "safety_stock"]
    copy_insert_via_staging(engine, "fact_inventory_snapshot", rows, cols,
                             conflict_cols=["product_id", "location_id", "snapshot_timestamp"])

    print("Done.")