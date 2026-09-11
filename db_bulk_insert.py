"""
Shared bulk-insert utility using PostgreSQL's COPY command — much faster
and more resilient than row-by-row INSERT for large tables, and safe to
re-run (uses a staging table + ON CONFLICT DO NOTHING for the merge).

Import this in any generator script:
    from utils.db_bulk_insert import copy_insert_via_staging
"""

import io
import csv
import time
import pandas as pd


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
                row.append("")
            elif isinstance(v, bool):
                row.append("true" if v else "false")
            else:
                row.append(str(v))
        writer.writerow(row)
    buf.seek(0)
    return buf


def copy_insert_via_staging(engine, table, records, columns, conflict_cols, chunk_size=20000, max_retries=5):
    """Bulk-loads records using COPY into a temp staging table, then
    merges into the real table with ON CONFLICT DO NOTHING. Chunked with
    retries so a network drop only costs one chunk, not the whole table.

    records: list of dicts
    columns: list of column names, in the order matching the table
    conflict_cols: list of column names forming the uniqueness constraint
                   to check for ON CONFLICT (usually the primary key)
    """
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