"""
Generates the final 4 operational tables, using REAL products, stores,
and customers already in the database:
 - fact_stock_transfer   (~800 rows)
 - fact_price_history     (~2,000 rows, scaled to product count)
 - fact_store_traffic     (1,800 rows: 180 days x 10 stores)
 - fact_ecommerce_event   (~50,000 rows)

Run with:  python generate_remaining_operational.py
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

random.seed(44)

START = date(2026, 3, 1)
TOTAL_DAYS = 180


def fetch_masters(engine):
    with engine.connect() as conn:
        products = [r[0] for r in conn.execute(text("SELECT product_id FROM dim_product"))]
        stores = [r[0] for r in conn.execute(text(
            "SELECT location_id FROM dim_location WHERE location_type = 'Store'"
        ))]
        customers = [r[0] for r in conn.execute(text("SELECT customer_id FROM dim_customer"))]
    return products, stores, customers


# ------------------------------------------------------------------
# fact_stock_transfer
# ------------------------------------------------------------------

def generate_transfers(products, stores, n=800):
    rows = []
    for i in range(1, n + 1):
        pid = random.choice(products)
        from_s, to_s = random.sample(stores, 2)
        req = START + timedelta(days=random.randint(0, TOTAL_DAYS - 10))
        dispatch = req + timedelta(days=random.randint(1, 3))
        arrival = dispatch + timedelta(days=random.randint(1, 4))
        qty = random.choice([10, 20, 30, 50])

        rows.append({
            "transfer_id": f"TRF{i:05d}",
            "product_id": pid,
            "from_location_id": from_s,
            "to_location_id": to_s,
            "request_timestamp": f"{req.isoformat()} 09:00:00",
            "dispatch_timestamp": f"{dispatch.isoformat()} 09:00:00",
            "arrival_timestamp": f"{arrival.isoformat()} 09:00:00",
            "transfer_quantity": qty,
            "transfer_cost": round(qty * random.uniform(15, 40), 2),
            "status": "Received",
        })
    return rows


# ------------------------------------------------------------------
# fact_price_history
# ------------------------------------------------------------------

def generate_price_history(products):
    rows = []
    price_id_counter = 1
    for pid in products:
        mrp = round(random.uniform(500, 5000), -1) + 9
        n_segments = random.choice([1, 2, 3])
        seg_start = START
        end_boundary = START + timedelta(days=TOTAL_DAYS)

        for seg in range(n_segments):
            if seg_start > end_boundary:
                break
            markdown = 0 if seg == 0 else random.choice([10, 15, 20, 30])
            selling_price = round(mrp * (1 - markdown / 100), 2)
            seg_len = random.randint(30, 90)
            seg_end = min(seg_start + timedelta(days=seg_len), end_boundary)

            rows.append({
                "price_record_id": f"PRC{price_id_counter:07d}",
                "product_id": pid,
                "location_id": None,  # national pricing
                "valid_from": seg_start.isoformat(),
                "valid_to": seg_end.isoformat(),
                "mrp": mrp,
                "selling_price": selling_price,
                "markdown_percentage": markdown,
                "price_change_reason": "Initial Launch" if seg == 0 else "Clearance",
                "approved_by": "pricing_team",
            })
            price_id_counter += 1
            seg_start = seg_end + timedelta(days=1)

    return rows


# ------------------------------------------------------------------
# fact_store_traffic
# ------------------------------------------------------------------

def generate_store_traffic(stores):
    rows = []
    for d in range(TOTAL_DAYS):
        day = START + timedelta(days=d)
        is_weekend = day.weekday() >= 5
        for sid in stores:
            footfall = max(50, int(random.gauss(400 if is_weekend else 250, 40)))
            transactions = int(footfall * random.uniform(0.15, 0.3))
            rows.append({
                "date_id": day.strftime("%Y%m%d"),
                "location_id": sid,
                "footfall": footfall,
                "transactions": transactions,
                "conversion_rate": round(transactions / footfall, 3),
                "average_bill_value": round(random.uniform(1200, 3500), 2),
                "operating_hours": 10.0,
            })
    return rows


# ------------------------------------------------------------------
# fact_ecommerce_event
# ------------------------------------------------------------------

def generate_ecommerce_events(products, customers):
    event_types = ["search", "view", "cart", "wishlist", "checkout"]
    event_weights = [30, 35, 15, 10, 10]
    devices = ["Mobile", "Desktop", "Tablet"]
    sources = ["Direct", "Advertisement", "Social", "Organic Search", "Email"]

    rows = []
    eid = 1
    for d in range(TOTAL_DAYS):
        day = START + timedelta(days=d)
        n_events = random.randint(200, 350)
        for _ in range(n_events):
            pid = random.choice(products)
            etype = random.choices(event_types, weights=event_weights)[0]
            rows.append({
                "event_id": f"EV{eid:08d}",
                "event_timestamp": f"{day.isoformat()} {random.randint(0,23):02d}:{random.randint(0,59):02d}:00",
                "date_id": day.strftime("%Y%m%d"),
                "session_id": f"SESS{random.randint(1,999999):06d}",
                "customer_id": random.choice(customers) if random.random() < 0.6 else None,
                "product_id": pid,
                "event_type": etype,
                "search_term": None,
                "device_type": random.choice(devices),
                "traffic_source": random.choice(sources),
            })
            eid += 1
    return rows


def insert_rows_batched(engine, table, rows, batch_size=500, max_retries=5):
    if not rows:
        print(f"No rows generated for {table}.")
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
                print(f"  reconnecting and retrying in {wait}s...")
                time.sleep(wait)
        i += batch_size

    print(f"Inserted {total} rows into {table}")


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    products, stores, customers = fetch_masters(engine)
    print(f"Loaded {len(products)} products, {len(stores)} stores, {len(customers)} customers.")

    if not products or not stores:
        raise SystemExit("Missing master data — run the earlier generator scripts first.")

    # transfers, prices, traffic already fully inserted in an earlier run — skip regenerating/reinserting
    events = generate_ecommerce_events(products, customers)
    print(f"Generated {len(events)} ecommerce events (full set).")

    # RESUME LOGIC: skip rows already successfully inserted in a previous
    # (interrupted) run, instead of restarting from scratch every time.
    with engine.connect() as conn:
        already_inserted = conn.execute(text("SELECT COUNT(*) FROM fact_ecommerce_event")).scalar()
    print(f"{already_inserted} rows already in the table — resuming from there.")

    remaining_events = events[already_inserted:]
    print(f"{len(remaining_events)} rows left to insert.")

    insert_rows_batched(engine, "fact_ecommerce_event", remaining_events, batch_size=500)

    print("Done.")