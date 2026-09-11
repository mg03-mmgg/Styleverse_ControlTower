"""
Generates fact_return_line using REAL sale lines, clamping return dates
to stay within dim_calendar's range. Uses COPY-based bulk insert for
speed and resilience.

Run with:  python generate_fact_return_line.py
"""

import os
import io
import csv
import time
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(99)

RETURN_RATE_BY_BRAND = {
    "Maison Luxe": 0.08,
    "UrbanEdge": 0.15,
    "SpeedStyle": 0.28,
    "EcoWeave": 0.12,
    "ThreadBasics": 0.18,
}

REASONS = ["Size/Fit issue", "Quality issue", "Colour mismatch",
           "Changed mind", "Late delivery", "Wrong item"]
CONDITIONS = ["Resellable", "Resellable", "Resellable", "Damaged", "Defective"]


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
# generation logic
# ------------------------------------------------------------------

def fetch_calendar_max_date(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(date_id) FROM dim_calendar"))
        return result.scalar()


def fetch_sales_with_brand(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT s.sale_line_id, s.product_id, s.location_id, s.customer_id,
                   s.date_id, s.units_sold, s.net_sales, p.brand
            FROM fact_sales_line s
            JOIN dim_product p ON s.product_id = p.product_id
        """))
        return result.fetchall()


def generate_returns(sales_rows, max_date_id):
    max_date = datetime.strptime(max_date_id, "%Y%m%d")
    rows = []
    return_id_counter = 1

    for sale_line_id, product_id, location_id, customer_id, date_id, units_sold, net_sales, brand in sales_rows:
        rate = RETURN_RATE_BY_BRAND.get(brand, 0.15)
        if random.random() >= rate:
            continue

        sale_date = datetime.strptime(date_id, "%Y%m%d")
        return_date = sale_date + timedelta(days=random.randint(2, 21))

        if return_date > max_date:
            return_date = max_date

        return_date_id = return_date.strftime("%Y%m%d")

        rows.append({
            "return_id": f"RET{return_id_counter:08d}",
            "original_sale_line_id": sale_line_id,
            "date_id": return_date_id,
            "product_id": product_id,
            "location_id": location_id,
            "customer_id": customer_id,
            "units_returned": units_sold,
            "refund_amount": float(net_sales),
            "return_reason": random.choice(REASONS),
            "product_condition": random.choices(CONDITIONS, weights=[50, 20, 15, 10, 5])[0],
        })
        return_id_counter += 1

    return rows


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    max_date_id = fetch_calendar_max_date(engine)
    print(f"Calendar max date: {max_date_id}")

    print("Fetching real sales lines with brand info...")
    sales_rows = fetch_sales_with_brand(engine)
    print(f"Loaded {len(sales_rows)} sales lines.")

    if not sales_rows:
        raise SystemExit("No rows found in fact_sales_line — run that generator first.")

    return_rows = generate_returns(sales_rows, max_date_id)
    print(f"Generated {len(return_rows)} return rows "
          f"({100*len(return_rows)/len(sales_rows):.1f}% overall return rate).")

    cols = ["return_id", "original_sale_line_id", "date_id", "product_id", "location_id",
            "customer_id", "units_returned", "refund_amount", "return_reason", "product_condition"]
    copy_insert_via_staging(engine, "fact_return_line", return_rows, cols, conflict_cols=["return_id"])

    print("Done.")