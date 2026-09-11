"""
Comprehensive data quality validation across ALL 21 populated tables:
 - row counts
 - duplicate primary key checks
 - null checks on key columns
 - plus targeted logical checks (impossible values, date ordering)
   for the tables where those apply.

Run with:  python pipeline/validate_data.py
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

# table -> (primary key columns, key columns that should never be null)
TABLE_SPEC = {
    "dim_product":              (["product_id"], ["brand", "category", "mrp", "unit_cost"]),
    "dim_location":              (["location_id"], ["location_type"]),
    "dim_supplier":               (["supplier_id"], ["supplier_name"]),
    "dim_customer":                (["customer_id"], ["loyalty_tier"]),
    "dim_promotion":                (["promotion_id"], ["start_date", "end_date"]),
    "dim_calendar":                  (["date_id"], ["full_date"]),
    "bridge_product_supplier":        (["product_id", "supplier_id"], ["unit_cost"]),
    "fact_sales_line":                 (["sale_line_id"], ["product_id", "location_id", "date_id", "units_sold"]),
    "fact_return_line":                  (["return_id"], ["original_sale_line_id", "product_id"]),
    "fact_inventory_snapshot":             (["snapshot_id"], ["product_id", "location_id", "available_quantity"]),
    "fact_ecommerce_event":                  (["event_id"], ["product_id", "event_type"]),
    "fact_purchase_order_line":                (["po_line_id"], ["product_id", "supplier_id", "ordered_quantity"]),
    "fact_supplier_capacity":                    (["capacity_id"], ["supplier_id", "total_capacity"]),
    "fact_stock_transfer":                         (["transfer_id"], ["product_id", "from_location_id", "to_location_id"]),
    "fact_price_history":                            (["price_record_id"], ["product_id", "mrp", "selling_price"]),
    "fact_store_traffic":                              (["date_id", "location_id"], ["footfall"]),
    "fact_external_signal":                              (["signal_id"], ["signal_type", "signal_value"]),
    "feat_sku_location_day":                               (["date_id", "product_id", "location_id"], ["current_price"]),
    "feat_supplier_sku_week":                                (["week_start_date", "supplier_id", "product_id"], ["available_capacity"]),
    "feat_price_response":                                     (["id"], ["product_id", "markdown_percentage"]),
    "label_actual_demand":                                       (["week_start_date", "product_id", "location_id"], ["units_sold", "estimated_total_demand"]),
}


def run_generic_checks(engine, table, pk_cols, notnull_cols):
    with engine.connect() as conn:
        row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

        pk_list = ", ".join(pk_cols)
        dup_count = conn.execute(text(
            f"SELECT COUNT(*) FROM (SELECT {pk_list} FROM {table} GROUP BY {pk_list} HAVING COUNT(*) > 1) x"
        )).scalar()

        null_issues = []
        for col in notnull_cols:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL")).scalar()
            if n > 0:
                null_issues.append(f"{col}={n}")

    status = "OK" if row_count > 0 and dup_count == 0 and not null_issues else "ISSUE"
    null_str = f", nulls: {', '.join(null_issues)}" if null_issues else ""
    print(f"  [{status}] {table:<30} rows={row_count:<8} dup_keys={dup_count}{null_str}")

    return {"table": table, "rows": row_count, "duplicates": dup_count, "null_issues": null_issues}


def run_logical_checks(engine):
    print("\n=== Targeted logical checks ===")

    def check(name, query):
        with engine.connect() as conn:
            n = conn.execute(text(query)).scalar()
        print(f"  [{'OK' if n == 0 else 'ISSUE'}] {name}: {n}")

    check("negative on_hand_quantity",
          "SELECT COUNT(*) FROM fact_inventory_snapshot WHERE on_hand_quantity < 0")
    check("negative available_quantity",
          "SELECT COUNT(*) FROM fact_inventory_snapshot WHERE available_quantity < 0")
    check("negative units_sold",
          "SELECT COUNT(*) FROM fact_sales_line WHERE units_sold < 0")
    check("selling_price > mrp (impossible markup)",
          "SELECT COUNT(*) FROM fact_price_history WHERE selling_price > mrp")
    check("cogs > gross_sales (cost exceeds price)",
          "SELECT COUNT(*) FROM fact_sales_line WHERE cogs > gross_sales")
    check("return before original sale (impossible)", """
        SELECT COUNT(*) FROM fact_return_line r
        JOIN fact_sales_line s ON r.original_sale_line_id = s.sale_line_id
        WHERE r.date_id < s.date_id
    """)
    check("PO actual_delivery before order_date", """
        SELECT COUNT(*) FROM fact_purchase_order_line
        WHERE actual_delivery_date < TO_DATE(order_date_id, 'YYYYMMDD')
    """)
    check("sales referencing a product not in dim_product (should be impossible — FK enforced)", """
        SELECT COUNT(*) FROM fact_sales_line s
        LEFT JOIN dim_product p ON s.product_id = p.product_id
        WHERE p.product_id IS NULL
    """)


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    print("=== Generic checks across all 21 tables ===")
    results = []
    for table, (pk_cols, notnull_cols) in TABLE_SPEC.items():
        results.append(run_generic_checks(engine, table, pk_cols, notnull_cols))

    run_logical_checks(engine)

    issues = [r for r in results if r["rows"] == 0 or r["duplicates"] > 0 or r["null_issues"]]
    print(f"\n=== Summary: {len(TABLE_SPEC)} tables checked, {len(issues)} with issues ===")
    for r in issues:
        print(f"  - {r['table']}: rows={r['rows']}, duplicates={r['duplicates']}, nulls={r['null_issues']}")

    print("\nValidation complete.")