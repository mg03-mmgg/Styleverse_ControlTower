"""
Generates feat_sku_location_day — the main ML training table — by
aggregating REAL data already in the database:
 - fact_sales_line       -> sales_1/7/28_days, sales_growth_7_days
 - fact_inventory_snapshot -> available_inventory, days_of_inventory
 - fact_price_history     -> current_price, markdown_percentage
 - dim_promotion          -> promotion_flag
 - fact_store_traffic     -> footfall
 - fact_ecommerce_event   -> views_7_days, cart_additions_7_days (product-level, applied across its stores)
 - fact_return_line       -> return_rate_28_days
 - dim_calendar           -> festival_flag
 - bridge_product_supplier -> supplier_lead_days
 - weather_score, trend_score -> synthetic (no real external signal source in this prototype)

Scoped to the TOP 250 products by total units sold, across all 10 stores,
for all 180 days (~450,000 rows) — matching the "pilot on top SKUs"
approach from the case study itself, and keeping this within a
reasonable size for the free-tier database.

Run with:  python generate_feat_sku_location_day.py
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
from sqlalchemy.exc import OperationalError

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(101)

START = date(2026, 3, 1)


def records_to_csv_buffer(records, columns):
    """Convert a list of dicts into an in-memory CSV buffer for COPY,
    correctly handling None -> NULL, booleans, and Decimal/numeric types."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    for r in records:
        row = []
        for col in columns:
            v = r[col]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                row.append("")  # empty field -> NULL (with NULL '' option below)
            elif isinstance(v, bool):
                row.append("true" if v else "false")
            else:
                row.append(str(v))
        writer.writerow(row)
    buf.seek(0)
    return buf


def copy_insert_via_staging(engine, table, records, columns, conflict_cols, chunk_size=50000, max_retries=5):
    """Bulk-loads records using COPY (fast, single stream) into a temp
    staging table, then merges into the real table with ON CONFLICT DO
    NOTHING. Chunked with retries so a network drop only costs one chunk,
    not the whole table."""
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
                    # temp table lives only for this connection's session
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
TOTAL_DAYS = 180
TOP_N_PRODUCTS = 250


def get_top_products(engine, n):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT product_id, SUM(units_sold) as total_units
            FROM fact_sales_line
            GROUP BY product_id
            ORDER BY total_units DESC
            LIMIT :n
        """), {"n": n})
        return [r[0] for r in result]


def fetch_sales(engine, products):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT date_id, product_id, location_id, units_sold
            FROM fact_sales_line
            WHERE product_id = ANY(:products)
        """), {"products": products})
        df = pd.DataFrame(result.fetchall(), columns=["date_id", "product_id", "location_id", "units_sold"])
    df["date"] = pd.to_datetime(df["date_id"], format="%Y%m%d")
    return df


def fetch_inventory(engine, products):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT date_id, product_id, location_id, available_quantity
            FROM fact_inventory_snapshot
            WHERE product_id = ANY(:products)
        """), {"products": products})
        df = pd.DataFrame(result.fetchall(),
                           columns=["date_id", "product_id", "location_id", "available_quantity"])
    df["date"] = pd.to_datetime(df["date_id"], format="%Y%m%d")
    return df


def fetch_prices(engine, products):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT product_id, valid_from, valid_to, selling_price, markdown_percentage
            FROM fact_price_history
            WHERE product_id = ANY(:products)
        """), {"products": products})
        return pd.DataFrame(result.fetchall(),
                             columns=["product_id", "valid_from", "valid_to", "selling_price", "markdown_percentage"])


def fetch_promotions(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT start_date, end_date FROM dim_promotion"))
        return result.fetchall()


def fetch_traffic(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT date_id, location_id, footfall FROM fact_store_traffic"))
        df = pd.DataFrame(result.fetchall(), columns=["date_id", "location_id", "footfall"])
    df["date"] = pd.to_datetime(df["date_id"], format="%Y%m%d")
    return df


def fetch_ecommerce(engine, products):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT date_id, product_id, event_type
            FROM fact_ecommerce_event
            WHERE product_id = ANY(:products)
        """), {"products": products})
        df = pd.DataFrame(result.fetchall(), columns=["date_id", "product_id", "event_type"])
    df["date"] = pd.to_datetime(df["date_id"], format="%Y%m%d")
    return df


def fetch_returns(engine, products):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT date_id, product_id, location_id, units_returned
            FROM fact_return_line
            WHERE product_id = ANY(:products)
        """), {"products": products})
        df = pd.DataFrame(result.fetchall(),
                           columns=["date_id", "product_id", "location_id", "units_returned"])
    df["date"] = pd.to_datetime(df["date_id"], format="%Y%m%d")
    return df


def fetch_calendar(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT date_id, festival_flag FROM dim_calendar"))
        df = pd.DataFrame(result.fetchall(), columns=["date_id", "festival_flag"])
    df["date"] = pd.to_datetime(df["date_id"], format="%Y%m%d")
    return df


def fetch_supplier_lead(engine, products):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT product_id, lead_days FROM bridge_product_supplier
            WHERE product_id = ANY(:products) AND is_primary_supplier = true
        """), {"products": products})
        return dict(result.fetchall())


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    print(f"Selecting top {TOP_N_PRODUCTS} products by total units sold...")
    top_products = get_top_products(engine, TOP_N_PRODUCTS)
    print(f"Selected {len(top_products)} products.")

    with engine.connect() as conn:
        stores = [r[0] for r in conn.execute(text(
            "SELECT location_id FROM dim_location WHERE location_type = 'Store'"
        ))]

    dates = pd.date_range(START, START + timedelta(days=TOTAL_DAYS - 1), freq="D")

    print("Building base grid (product x store x day)...")
    grid = pd.MultiIndex.from_product(
        [top_products, stores, dates], names=["product_id", "location_id", "date"]
    ).to_frame(index=False)
    print(f"Base grid size: {len(grid)} rows.")

    print("Fetching sales...")
    sales_df = fetch_sales(engine, top_products)
    daily_sales = sales_df.groupby(["date", "product_id", "location_id"])["units_sold"].sum().reset_index()

    grid = grid.merge(daily_sales, on=["date", "product_id", "location_id"], how="left")
    grid["units_sold"] = grid["units_sold"].fillna(0)
    grid = grid.sort_values(["product_id", "location_id", "date"])

    print("Computing rolling sales windows...")
    grp = grid.groupby(["product_id", "location_id"])["units_sold"]
    grid["sales_1_day"] = grid["units_sold"]
    grid["sales_7_days"] = grp.transform(lambda x: x.rolling(7, min_periods=1).sum())
    grid["sales_28_days"] = grp.transform(lambda x: x.rolling(28, min_periods=1).sum())

    def growth(x):
        prior = x.shift(7).rolling(7, min_periods=1).sum()
        curr = x.rolling(7, min_periods=1).sum()
        g = (curr - prior) / prior.replace(0, pd.NA)
        return g.fillna(0)
    grid["sales_growth_7_days"] = grp.transform(growth).astype(float)

    print("Fetching and merging inventory...")
    inv_df = fetch_inventory(engine, top_products)
    inv_df = inv_df.sort_values(["product_id", "location_id", "date"])
    grid = grid.merge(
        inv_df[["date", "product_id", "location_id", "available_quantity"]],
        on=["date", "product_id", "location_id"], how="left"
    )
    # forward-fill inventory between the biweekly snapshots (per product-location)
    grid["available_inventory"] = grid.groupby(["product_id", "location_id"])["available_quantity"] \
        .transform(lambda x: x.ffill().bfill()).fillna(0)
    grid.drop(columns=["available_quantity"], inplace=True)

    grid["days_of_inventory"] = grid.apply(
        lambda r: round(r["available_inventory"] / r["sales_1_day"], 1) if r["sales_1_day"] > 0
        else (999 if r["available_inventory"] > 0 else 0), axis=1
    )

    print("Fetching and merging prices...")
    price_df = fetch_prices(engine, top_products)
    price_df["valid_from"] = pd.to_datetime(price_df["valid_from"])
    price_df["valid_to"] = pd.to_datetime(price_df["valid_to"])

    # merge_asof needs sorted data per product
    grid = grid.sort_values(["product_id", "date"])
    price_lookup_rows = []
    for pid, pgroup in price_df.groupby("product_id"):
        pgroup = pgroup.sort_values("valid_from")
        price_lookup_rows.append(pgroup)
    price_all = pd.concat(price_lookup_rows) if price_lookup_rows else price_df

    merged_prices = []
    for pid, pgrid in grid.groupby("product_id"):
        pprices = price_all[price_all["product_id"] == pid].sort_values("valid_from")
        if pprices.empty:
            pgrid = pgrid.copy()
            pgrid["current_price"] = None
            pgrid["markdown_percentage"] = 0
        else:
            pgrid = pd.merge_asof(pgrid.sort_values("date"), pprices[["valid_from", "selling_price", "markdown_percentage"]],
                                   left_on="date", right_on="valid_from", direction="backward")
            pgrid.drop(columns=["valid_from"], inplace=True)
            pgrid.rename(columns={"selling_price": "current_price"}, inplace=True)
        merged_prices.append(pgrid)
    grid = pd.concat(merged_prices).sort_values(["product_id", "location_id", "date"])

    print("Flagging promotion days...")
    promos = fetch_promotions(engine)
    promo_dates = set()
    for start_d, end_d in promos:
        d = start_d
        while d <= end_d:
            promo_dates.add(d)
            d += timedelta(days=1)
    grid["promotion_flag"] = grid["date"].dt.date.isin(promo_dates)

    print("Merging store traffic...")
    traffic_df = fetch_traffic(engine)
    grid = grid.merge(traffic_df[["date", "location_id", "footfall"]],
                       on=["date", "location_id"], how="left")
    grid["footfall"] = grid["footfall"].fillna(0)

    print("Aggregating ecommerce signals (product-level, applied across its stores)...")
    ecom_df = fetch_ecommerce(engine, top_products)
    ecom_daily = ecom_df.groupby(["date", "product_id", "event_type"]).size().unstack(fill_value=0).reset_index()
    for col in ["view", "cart"]:
        if col not in ecom_daily.columns:
            ecom_daily[col] = 0
    ecom_daily = ecom_daily.sort_values(["product_id", "date"])
    ecom_daily["views_7_days"] = ecom_daily.groupby("product_id")["view"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    ecom_daily["cart_additions_7_days"] = ecom_daily.groupby("product_id")["cart"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    grid = grid.merge(ecom_daily[["date", "product_id", "views_7_days", "cart_additions_7_days"]],
                       on=["date", "product_id"], how="left")
    grid["views_7_days"] = grid["views_7_days"].fillna(0)
    grid["cart_additions_7_days"] = grid["cart_additions_7_days"].fillna(0)

    print("Computing return rate...")
    returns_df = fetch_returns(engine, top_products)
    returns_daily = returns_df.groupby(["date", "product_id", "location_id"])["units_returned"].sum().reset_index()
    grid = grid.merge(returns_daily, on=["date", "product_id", "location_id"], how="left")
    grid["units_returned"] = grid["units_returned"].fillna(0)
    grp2 = grid.groupby(["product_id", "location_id"])
    grid["returns_28_days"] = grp2["units_returned"].transform(lambda x: x.rolling(28, min_periods=1).sum())
    grid["return_rate_28_days"] = (grid["returns_28_days"] / grid["sales_28_days"].replace(0, pd.NA)).fillna(0).clip(0, 1)
    grid.drop(columns=["units_returned", "returns_28_days"], inplace=True)

    print("Merging calendar (festival flag)...")
    cal_df = fetch_calendar(engine)
    grid = grid.merge(cal_df[["date", "festival_flag"]], on="date", how="left")

    print("Fetching supplier lead days...")
    lead_map = fetch_supplier_lead(engine, top_products)
    grid["supplier_lead_days"] = grid["product_id"].map(lead_map).fillna(30).astype(int)

    print("Generating synthetic weather and trend scores...")
    unique_dates = grid["date"].dt.date.unique()
    weather_by_date = {d: round(random.uniform(0.3, 1.0), 2) for d in unique_dates}
    grid["weather_score"] = grid["date"].dt.date.map(weather_by_date)

    trend_by_product = {pid: round(random.uniform(0.2, 1.0), 2) for pid in top_products}
    grid["trend_score"] = grid["product_id"].map(trend_by_product) * grid["views_7_days"].apply(
        lambda v: min(1.5, 0.7 + v / 500)
    )
    grid["trend_score"] = grid["trend_score"].round(2)

    print("Finalizing columns...")
    grid["date_id"] = grid["date"].dt.strftime("%Y%m%d")

    # these columns are INTEGER in the schema — pandas rolling/fillna operations
    # left them as floats (e.g. 0.0), which COPY rejects (unlike INSERT, which
    # silently casts). Cast explicitly before writing.
    int_cols = ["views_7_days", "cart_additions_7_days", "available_inventory",
                "footfall", "supplier_lead_days"]
    for col in int_cols:
        grid[col] = grid[col].fillna(0).astype(int)

    final_cols = ["date_id", "product_id", "location_id", "sales_1_day", "sales_7_days",
                  "sales_28_days", "sales_growth_7_days", "views_7_days", "cart_additions_7_days",
                  "return_rate_28_days", "available_inventory", "days_of_inventory", "current_price",
                  "markdown_percentage", "promotion_flag", "footfall", "weather_score",
                  "festival_flag", "trend_score", "supplier_lead_days"]
    grid = grid[final_cols]

    print(f"Final feature table shape: {grid.shape}")
    print(grid.head(3).to_string())

    print("Inserting into feat_sku_location_day via COPY (fast, resilient)...")
    records = grid.to_dict(orient="records")
    copy_insert_via_staging(
        engine, "feat_sku_location_day", records, final_cols,
        conflict_cols=["date_id", "product_id", "location_id"],
        chunk_size=50000
    )

    print("Done.")