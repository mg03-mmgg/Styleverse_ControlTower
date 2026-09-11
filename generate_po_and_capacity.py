"""
Generates fact_purchase_order_line and fact_supplier_capacity using the
REAL bridge_product_supplier relationships (so every PO references a
product-supplier pair that genuinely exists, at that supplier's real
unit_cost and lead_days).

fact_purchase_order_line: ~1,500-2,500 rows (weekly reorder cycles)
fact_supplier_capacity:   ~4,000-5,200 rows (weekly capacity per supplier's product mix)

Run with:  python generate_po_and_capacity.py
"""

import os
import random
from datetime import date, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(33)


def fetch_bridge(engine):
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT product_id, supplier_id, is_primary_supplier, unit_cost, lead_days "
            "FROM bridge_product_supplier"
        ))
        return result.fetchall()


def generate_purchase_orders(bridge_rows, start=date(2026, 3, 1), weeks=26):
    # only reorder from each product's PRIMARY supplier, for realism
    primary = {}
    for pid, sid, is_primary, unit_cost, lead_days in bridge_rows:
        if is_primary:
            primary[pid] = {"supplier_id": sid, "unit_cost": float(unit_cost), "lead_days": lead_days}

    all_products = list(primary.keys())
    order_dates = [start + timedelta(days=7 * w) for w in range(weeks)]

    rows = []
    po_counter = 1
    po_line_counter = 1

    for od in order_dates:
        reorder_products = random.sample(all_products, k=min(len(all_products), random.randint(40, 70)))
        for pid in reorder_products:
            info = primary[pid]
            po_id = f"PO{po_counter:06d}"
            ordered_qty = random.choice([100, 150, 200, 300, 500])
            promised = od + timedelta(days=info["lead_days"])
            actual = promised + timedelta(days=random.randint(1, 10)) if random.random() < 0.2 else promised
            received_qty = ordered_qty if random.random() > 0.1 else int(ordered_qty * random.uniform(0.8, 0.98))
            status = "Completed" if actual <= od + timedelta(days=60) else "In Production"

            rows.append({
                "po_line_id": f"POL{po_line_counter:07d}",
                "po_id": po_id,
                "product_id": pid,
                "supplier_id": info["supplier_id"],
                "order_date_id": od.strftime("%Y%m%d"),
                "promised_delivery_date": promised.isoformat(),
                "actual_delivery_date": actual.isoformat(),
                "ordered_quantity": ordered_qty,
                "received_quantity": received_qty,
                "unit_cost": info["unit_cost"],
                "status": status,
            })
            po_counter += 1
            po_line_counter += 1

    return rows


def generate_supplier_capacity(bridge_rows, start=date(2026, 3, 1), weeks=26):
    by_supplier = {}
    for pid, sid, is_primary, unit_cost, lead_days in bridge_rows:
        by_supplier.setdefault(sid, []).append(pid)

    rows = []
    for sid, products in by_supplier.items():
        sample = random.sample(products, k=min(40, len(products)))
        for pid in sample:
            for w in range(weeks):
                week_start = start + timedelta(days=7 * w)
                total_cap = random.randint(500, 2000)
                reserved = int(total_cap * random.uniform(0.3, 0.6))
                used = int(reserved * random.uniform(0.5, 1.0))

                rows.append({
                    "supplier_id": sid,
                    "product_id": pid,
                    "week_start_date": week_start.isoformat(),
                    "total_capacity": total_cap,
                    "reserved_capacity": reserved,
                    "used_capacity": used,
                    "expedite_capacity": int(total_cap * 0.15),
                    "estimated_lead_days": random.randint(15, 45),
                    "raw_material_available": random.random() > 0.1,
                })
    return rows



import time
from sqlalchemy import text
from sqlalchemy.exc import OperationalError


def insert_rows_batched(engine, table, rows, batch_size=1000, max_retries=3):
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
                # each batch gets its OWN short-lived transaction
                with engine.begin() as conn:
                    conn.execute(stmt, batch)
                print(f"  inserted {min(i + batch_size, total)}/{total}")
                break  # success, move to next batch
            except OperationalError as e:
                attempt += 1
                print(f"  batch failed (attempt {attempt}/{max_retries}): {e}")
                if attempt >= max_retries:
                    raise
                time.sleep(3)  # brief pause before retrying, lets connection recover
        i += batch_size

    print(f"Inserted {total} rows into {table}")



if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)


    print("Fetching bridge_product_supplier...")
    bridge_rows = fetch_bridge(engine)
    print(f"Loaded {len(bridge_rows)} product-supplier relationships.")

    if not bridge_rows:
        raise SystemExit("bridge_product_supplier is empty — run that generator first.")

    po_rows = generate_purchase_orders(bridge_rows)
    print(f"Generated {len(po_rows)} purchase order lines.")
    late_pct = 100 * sum(1 for r in po_rows if r["actual_delivery_date"] > r["promised_delivery_date"]) / len(po_rows)
    print(f"  ({late_pct:.1f}% delivered late)")

    cap_rows = generate_supplier_capacity(bridge_rows)
    print(f"Generated {len(cap_rows)} supplier capacity rows.")

    insert_rows_batched(engine, "fact_purchase_order_line", po_rows)
    insert_rows_batched(engine, "fact_supplier_capacity", cap_rows)

    print("Done.")