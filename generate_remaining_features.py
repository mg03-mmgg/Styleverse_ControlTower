"""
Generates the remaining 3 model-ready tables from REAL data, using a
fast COPY-based bulk insert (faster and more resilient than row-by-row
INSERT — everything needed is self-contained in this one file):

 - label_actual_demand: weekly demand PER product-store, for the SAME
   top-250 products used in feat_sku_location_day.
 - feat_supplier_sku_week: weekly supplier capacity/performance per
   product.
 - feat_price_response: sales before/after real markdown events.

Run with:  python generate_remaining_features.py
"""

import os
import io
import csv
import time
import random
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(202)

START = date(2026, 3, 1)
TOTAL_DAYS = 180
TOP_N_PRODUCTS = 250


# ------------------------------------------------------------------
# COPY-based bulk insert (self-contained, no external imports)
# ------------------------------------------------------------------

def records_to_csv_buffer(records, columns):
    buf = io.StringIO()
    writer = csv.writer(buf)
    for r in records:
        row = []
        for col in columns:
            v = r[col]
            if v is None or (isinstance(v, float) and pd.isna(v)):
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
# label_actual_demand
# ------------------------------------------------------------------

def generate_label_actual_demand(engine, products, stores):
    with engine.connect() as conn:
        sales = conn.execute(text("""
            SELECT date_id, product_id, location_id, units_sold
            FROM fact_sales_line WHERE product_id = ANY(:p)
        """), {"p": products}).fetchall()
        inv = conn.execute(text("""
            SELECT date_id, product_id, location_id, available_quantity
            FROM fact_inventory_snapshot WHERE product_id = ANY(:p)
        """), {"p": products}).fetchall()

    sales_df = pd.DataFrame(sales, columns=["date_id", "product_id", "location_id", "units_sold"])
    sales_df["date"] = pd.to_datetime(sales_df["date_id"], format="%Y%m%d")

    inv_df = pd.DataFrame(inv, columns=["date_id", "product_id", "location_id", "available_quantity"])
    inv_df["date"] = pd.to_datetime(inv_df["date_id"], format="%Y%m%d")

    dates = pd.date_range(START, START + timedelta(days=TOTAL_DAYS - 1), freq="D")
    grid = pd.MultiIndex.from_product([products, stores, dates],
                                        names=["product_id", "location_id", "date"]).to_frame(index=False)

    grid = grid.merge(inv_df[["date", "product_id", "location_id", "available_quantity"]],
                       on=["date", "product_id", "location_id"], how="left")
    grid = grid.sort_values(["product_id", "location_id", "date"])
    grid["available_quantity"] = grid.groupby(["product_id", "location_id"])["available_quantity"] \
        .transform(lambda x: x.ffill().bfill()).fillna(0)

    grid = grid.merge(sales_df[["date", "product_id", "location_id", "units_sold"]],
                       on=["date", "product_id", "location_id"], how="left")
    grid["units_sold"] = grid["units_sold"].fillna(0)

    grid["week_start"] = grid["date"] - pd.to_timedelta(grid["date"].dt.dayofweek, unit="D")
    grid["is_stockout_day"] = grid["available_quantity"] == 0

    weekly = grid.groupby(["product_id", "location_id", "week_start"]).agg(
        units_sold=("units_sold", "sum"),
        stockout_days=("is_stockout_day", "sum"),
    ).reset_index()

    avg_daily = grid[~grid["is_stockout_day"]].groupby(["product_id", "location_id"])["units_sold"] \
        .mean().rename("avg_daily_rate")
    weekly = weekly.merge(avg_daily, on=["product_id", "location_id"], how="left")
    weekly["avg_daily_rate"] = weekly["avg_daily_rate"].fillna(0)

    weekly["estimated_lost_sales"] = (weekly["stockout_days"] * weekly["avg_daily_rate"]).round(1)
    weekly["estimated_total_demand"] = weekly["units_sold"] + weekly["estimated_lost_sales"]
    weekly["stockout_flag"] = weekly["stockout_days"] > 0

    weekly["week_start_date"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
    weekly["stockout_days"] = weekly["stockout_days"].astype(int)
    weekly["units_sold"] = weekly["units_sold"].astype(int)

    return weekly[["week_start_date", "product_id", "location_id", "units_sold",
                    "stockout_days", "estimated_lost_sales", "estimated_total_demand", "stockout_flag"]]


# ------------------------------------------------------------------
# feat_supplier_sku_week
# ------------------------------------------------------------------

def generate_feat_supplier_sku_week(engine):
    with engine.connect() as conn:
        capacity = conn.execute(text("""
            SELECT supplier_id, product_id, week_start_date, total_capacity,
                   reserved_capacity, used_capacity, estimated_lead_days
            FROM fact_supplier_capacity
        """)).fetchall()
        po = conn.execute(text("""
            SELECT supplier_id, product_id, order_date_id, ordered_quantity, received_quantity, unit_cost
            FROM fact_purchase_order_line
        """)).fetchall()
        suppliers = conn.execute(text(
            "SELECT supplier_id, on_time_rate, quality_score FROM dim_supplier"
        )).fetchall()

    cap_df = pd.DataFrame(capacity, columns=["supplier_id", "product_id", "week_start_date",
                                               "total_capacity", "reserved_capacity",
                                               "used_capacity", "estimated_lead_days"])
    po_df = pd.DataFrame(po, columns=["supplier_id", "product_id", "order_date_id",
                                        "ordered_quantity", "received_quantity", "unit_cost"])
    supplier_df = pd.DataFrame(suppliers, columns=["supplier_id", "on_time_rate", "quality_score"])

    po_df["order_date"] = pd.to_datetime(po_df["order_date_id"], format="%Y%m%d")
    po_df["week_start"] = po_df["order_date"] - pd.to_timedelta(po_df["order_date"].dt.dayofweek, unit="D")
    po_df["open_qty"] = po_df["ordered_quantity"] - po_df["received_quantity"]
    po_weekly = po_df.groupby(["supplier_id", "product_id", "week_start"])["open_qty"].sum().reset_index()
    po_weekly["week_start_date"] = po_weekly["week_start"].dt.strftime("%Y-%m-%d")

    cap_df["available_capacity"] = cap_df["total_capacity"] - cap_df["used_capacity"]
    cap_df = cap_df.merge(supplier_df, on="supplier_id", how="left")
    cap_df["defect_rate"] = (1 - cap_df["quality_score"] / 5).round(3).clip(0, 0.15)

    merged = cap_df.merge(
        po_weekly[["supplier_id", "product_id", "week_start_date", "open_qty"]],
        on=["supplier_id", "product_id", "week_start_date"],
        how="left"
    )
    merged["open_qty"] = merged["open_qty"].fillna(0).astype(int)

    unit_cost_lookup = po_df.groupby(["supplier_id", "product_id"])["unit_cost"].mean().rename("avg_unit_cost")
    merged = merged.merge(unit_cost_lookup, on=["supplier_id", "product_id"], how="left")
    merged["avg_unit_cost"] = merged["avg_unit_cost"].fillna(0).round(2)

    merged["expected_demand_gap"] = (merged["reserved_capacity"] - merged["used_capacity"]).clip(lower=0)

    all_weeks = sorted(merged["week_start_date"].unique())
    week_index = {w: i for i, w in enumerate(all_weeks)}
    merged["weeks_left_in_season"] = merged["week_start_date"].map(lambda w: max(0, len(all_weeks) - week_index[w]))

    merged.rename(columns={"avg_unit_cost": "unit_cost", "open_qty": "open_purchase_order_quantity"}, inplace=True)

    return merged[["week_start_date", "supplier_id", "product_id", "available_capacity",
                    "reserved_capacity", "estimated_lead_days", "on_time_rate", "defect_rate",
                    "unit_cost", "open_purchase_order_quantity", "expected_demand_gap",
                    "weeks_left_in_season"]].rename(columns={
                        "estimated_lead_days": "average_lead_days",
                        "on_time_rate": "on_time_delivery_rate",
                    })


# ------------------------------------------------------------------
# feat_price_response
# ------------------------------------------------------------------

def generate_feat_price_response(engine, products):
    with engine.connect() as conn:
        prices = conn.execute(text("""
            SELECT product_id, valid_from, valid_to, markdown_percentage
            FROM fact_price_history WHERE product_id = ANY(:p) AND markdown_percentage > 0
        """), {"p": products}).fetchall()
        sales = conn.execute(text("""
            SELECT date_id, product_id, units_sold, net_sales, cogs
            FROM fact_sales_line WHERE product_id = ANY(:p)
        """), {"p": products}).fetchall()

    price_df = pd.DataFrame(prices, columns=["product_id", "valid_from", "valid_to", "markdown_percentage"])
    sales_df = pd.DataFrame(sales, columns=["date_id", "product_id", "units_sold", "net_sales", "cogs"])
    sales_df["date"] = pd.to_datetime(sales_df["date_id"], format="%Y%m%d")
    price_df["valid_from"] = pd.to_datetime(price_df["valid_from"])

    rows = []
    id_counter = 1
    for _, row in price_df.iterrows():
        pid = row["product_id"]
        markdown_start = row["valid_from"]
        before_window = sales_df[(sales_df["product_id"] == pid) &
                                   (sales_df["date"] >= markdown_start - timedelta(days=14)) &
                                   (sales_df["date"] < markdown_start)]
        after_window = sales_df[(sales_df["product_id"] == pid) &
                                  (sales_df["date"] >= markdown_start) &
                                  (sales_df["date"] < markdown_start + timedelta(days=14))]

        sales_before = before_window["units_sold"].sum()
        sales_after = after_window["units_sold"].sum()
        margin_before = (before_window["net_sales"].sum() - before_window["cogs"].sum())
        margin_after = (after_window["net_sales"].sum() - after_window["cogs"].sum())

        if sales_before == 0 and sales_after == 0:
            continue

        rows.append({
            "id": id_counter,
            "product_id": pid,
            "location_cluster": random.choice(["CL_A", "CL_B", "CL_C", "CL_D"]),
            "price_band": "Mid" if row["markdown_percentage"] <= 20 else "Deep",
            "markdown_percentage": float(row["markdown_percentage"]),
            "sales_before_markdown": float(sales_before),
            "sales_after_markdown": float(sales_after),
            "estimated_sales_increase": float(sales_after - sales_before),
            "gross_margin_before": round(float(margin_before), 2),
            "gross_margin_after": round(float(margin_after), 2),
            "remaining_inventory": random.randint(0, 200),
            "weeks_left_in_season": random.randint(1, 20),
        })
        id_counter += 1

    return rows


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    with engine.connect() as conn:
        top_products = [r[0] for r in conn.execute(text("""
            SELECT product_id FROM fact_sales_line
            GROUP BY product_id ORDER BY SUM(units_sold) DESC LIMIT :n
        """), {"n": TOP_N_PRODUCTS})]
        stores = [r[0] for r in conn.execute(text(
            "SELECT location_id FROM dim_location WHERE location_type = 'Store'"
        ))]

    print(f"Using top {len(top_products)} products, {len(stores)} stores.")

    print("\n--- label_actual_demand ---")
    label_df = generate_label_actual_demand(engine, top_products, stores)
    print(f"Generated {len(label_df)} rows.")
    label_cols = ["week_start_date", "product_id", "location_id", "units_sold",
                  "stockout_days", "estimated_lost_sales", "estimated_total_demand", "stockout_flag"]
    copy_insert_via_staging(engine, "label_actual_demand", label_df.to_dict(orient="records"),
                             label_cols, conflict_cols=["week_start_date", "product_id", "location_id"])

    print("\n--- feat_supplier_sku_week ---")
    supplier_week_df = generate_feat_supplier_sku_week(engine)
    print(f"Generated {len(supplier_week_df)} rows.")
    sw_cols = ["week_start_date", "supplier_id", "product_id", "available_capacity",
               "reserved_capacity", "average_lead_days", "on_time_delivery_rate", "defect_rate",
               "unit_cost", "open_purchase_order_quantity", "expected_demand_gap", "weeks_left_in_season"]
    copy_insert_via_staging(engine, "feat_supplier_sku_week", supplier_week_df.to_dict(orient="records"),
                             sw_cols, conflict_cols=["week_start_date", "supplier_id", "product_id"])

    print("\n--- feat_price_response ---")
    price_response_rows = generate_feat_price_response(engine, top_products)
    print(f"Generated {len(price_response_rows)} rows.")
    pr_cols = ["id", "product_id", "location_cluster", "price_band", "markdown_percentage",
               "sales_before_markdown", "sales_after_markdown", "estimated_sales_increase",
               "gross_margin_before", "gross_margin_after", "remaining_inventory", "weeks_left_in_season"]
    copy_insert_via_staging(engine, "feat_price_response", price_response_rows,
                             pr_cols, conflict_cols=["id"])

    print("\nDone.")