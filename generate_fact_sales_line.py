"""
RETUNED sales generator — increases concentration on fast-moving products
(the ones your ML models actually train on) and overall transaction
volume, so weekly demand per top-seller product-store cell is dense
enough to carry real, learnable signal (previous version averaged
~0.4 units/week per cell — too sparse for any model to learn from).

Uses the same real products, locations, calendar, customers, and
promotions as before. Uses a fast COPY-based bulk insert.

IMPORTANT: run this only AFTER truncating fact_return_line,
fact_sales_line, fact_inventory_snapshot, feat_sku_location_day, and
label_actual_demand — otherwise the fixed random seed will regenerate
identical IDs and ON CONFLICT DO NOTHING will silently skip everything.

Run with:  python generate_fact_sales_line.py
"""

import os
import io
import csv
import time
import random
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(55)


# ------------------------------------------------------------------
# COPY-based bulk insert (self-contained)
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
# fetch real master data
# ------------------------------------------------------------------

def fetch_all(engine):
    with engine.connect() as conn:
        products = conn.execute(text(
            "SELECT product_id, mrp, unit_cost FROM dim_product"
        )).fetchall()
        locations = conn.execute(text(
            "SELECT location_id, channel FROM dim_location WHERE location_type != 'Distribution Centre'"
        )).fetchall()
        calendar = conn.execute(text(
            "SELECT date_id, holiday_flag, festival_flag FROM dim_calendar ORDER BY date_id"
        )).fetchall()
        customers = conn.execute(text(
            "SELECT customer_id FROM dim_customer"
        )).fetchall()
        promotions = conn.execute(text(
            "SELECT promotion_id, start_date, end_date, discount_percentage FROM dim_promotion"
        )).fetchall()
    return products, locations, calendar, customers, promotions


def build_promotion_lookup(promotions):
    import datetime
    lookup = {}
    for promo_id, start, end, pct in promotions:
        d = start
        while d <= end:
            key = d.strftime("%Y%m%d")
            lookup.setdefault(key, []).append((promo_id, float(pct)))
            d += datetime.timedelta(days=1)
    return lookup


# ------------------------------------------------------------------
# RETUNED generation logic
# ------------------------------------------------------------------

def generate_sales(products, locations, calendar, customers, promo_lookup):
    customer_ids = [c[0] for c in customers]

    # assign popularity tier — same 20/50/30 split, but the WEIGHTS
    # below are sharply retuned to concentrate volume on fast movers
    popularity = {}
    for pid, _, _ in products:
        popularity[pid] = random.choices(["fast", "medium", "slow"], weights=[20, 50, 30])[0]

    # RETUNED: fast movers now dominate far more heavily (was 5/2/0.5)
    tier_weight = {"fast": 18.0, "medium": 2.0, "slow": 0.3}
    product_weights = [tier_weight[popularity[p[0]]] for p in products]

    rows = []
    sale_line_counter = 1
    transaction_counter = 1

    for date_id, holiday_flag, festival_flag in calendar:
        day_multiplier = 1.0
        if holiday_flag:
            day_multiplier *= 1.4
        if festival_flag:
            day_multiplier *= 2.0

        active_promos = promo_lookup.get(date_id, [])

        for location_id, channel in locations:
            # RETUNED: higher base transaction volume (was 20/12)
            base_txn = 35 if channel == "Online" else 22
            n_txn = max(1, int(random.gauss(base_txn * day_multiplier, base_txn * 0.2)))

            for _ in range(n_txn):
                transaction_id = f"TXN{transaction_counter:08d}"
                transaction_counter += 1

                n_lines = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
                chosen = random.choices(products, weights=product_weights, k=n_lines)
                customer_id = random.choice(customer_ids) if random.random() < 0.75 else None

                promo_id, promo_pct = (None, 0)
                if active_promos and random.random() < 0.6:
                    promo_id, promo_pct = random.choice(active_promos)

                for pid, mrp, unit_cost in chosen:
                    mrp = float(mrp)
                    unit_cost = float(unit_cost)
                    units = random.choices([1, 2, 3], weights=[70, 25, 5])[0]

                    gross = round(units * mrp, 2)
                    discount_amt = round(gross * promo_pct / 100, 2) if promo_id else 0.0
                    net = round(gross - discount_amt, 2)
                    cogs = round(units * unit_cost, 2)

                    rows.append({
                        "sale_line_id": f"SL{sale_line_counter:08d}",
                        "transaction_id": transaction_id,
                        "date_id": date_id,
                        "event_timestamp": f"{date_id[:4]}-{date_id[4:6]}-{date_id[6:]} "
                                             f"{random.randint(9,21):02d}:{random.randint(0,59):02d}:00",
                        "product_id": pid,
                        "location_id": location_id,
                        "customer_id": customer_id,
                        "promotion_id": promo_id,
                        "channel": channel,
                        "units_sold": units,
                        "gross_sales": gross,
                        "discount_amount": discount_amt,
                        "net_sales": net,
                        "cogs": cogs,
                        "source_system": "synthetic_generator",
                    })
                    sale_line_counter += 1

    return rows


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    products, locations, calendar, customers, promotions = fetch_all(engine)
    print(f"Loaded {len(products)} products, {len(locations)} locations, "
          f"{len(calendar)} calendar days, {len(customers)} customers, "
          f"{len(promotions)} promotions.")

    if not products or not locations or not calendar:
        raise SystemExit("Missing master data — make sure earlier generator scripts ran successfully.")

    promo_lookup = build_promotion_lookup(promotions)

    sales_rows = generate_sales(products, locations, calendar, customers, promo_lookup)
    print(f"Generated {len(sales_rows)} sale lines (retuned for higher density).")

    cols = ["sale_line_id", "transaction_id", "date_id", "event_timestamp", "product_id",
            "location_id", "customer_id", "promotion_id", "channel", "units_sold",
            "gross_sales", "discount_amount", "net_sales", "cogs", "source_system"]
    copy_insert_via_staging(engine, "fact_sales_line", sales_rows, cols, conflict_cols=["sale_line_id"])

    print("Done.")