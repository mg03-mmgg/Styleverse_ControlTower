"""
Converts the 174 MB Excel file into compact Parquet files, dropping
columns that are never used as features.

Why Parquet instead of Excel:
  - Loads in seconds instead of minutes (Excel parsing is very slow)
  - Typically 10-20x smaller on disk
  - Preserves data types properly (no re-parsing dates/numbers)

Columns dropped from TRAINING DATA (none are used as features):
  domain, source_dataset, source_category  - provenance only
  sku_id                                    - too high-cardinality, excluded
  days_observed, availability               - same-week observed, excluded
  logprice, logtraffic                      - redundant log transforms

Columns KEPT even though not features (needed by the pipeline):
  week_start     - required for the chronological split
  censored_days  - required to filter censored rows
  units_sold     - the target

SCORING TARGET keeps everything, since its extra columns (on_hand,
in_transit, markdown, cogs, supplier_reliability, position_id,
sku_name, store_name) feed the control tower directly.

Run ONCE with:  python convert_dataset_to_parquet.py
Then train with the Parquet files instead of the xlsx.
"""

import os
import pandas as pd

EXCEL_PATH = "StyleVerse_FINAL_training_data.xlsx"
OUT_DIR = "dataset_parquet"
os.makedirs(OUT_DIR, exist_ok=True)

DROP_FROM_TRAINING = [
    "domain", "source_dataset", "source_category",
    "sku_id",
    "days_observed", "availability",
    "logprice", "logtraffic",
]


def convert_sheet(sheet_name, out_name, drop_cols=None):
    print(f"\nReading '{sheet_name}' from Excel (slow, one time only)...")
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name, engine="openpyxl")
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    if drop_cols:
        present = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=present)
        print(f"  Dropped {len(present)} unused columns: {present}")
        print(f"  Now {len(df.columns)} columns")

    out_path = os.path.join(OUT_DIR, out_name)
    df.to_parquet(out_path, index=False, compression="snappy")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved to {out_path} ({size_mb:.1f} MB)")
    return df


if __name__ == "__main__":
    if not os.path.exists(EXCEL_PATH):
        raise SystemExit(f"File not found: {EXCEL_PATH}")

    excel_size = os.path.getsize(EXCEL_PATH) / (1024 * 1024)
    print(f"Source Excel: {excel_size:.1f} MB")

    convert_sheet("TRAINING DATA", "training_data.parquet", DROP_FROM_TRAINING)
    convert_sheet("SCORING TARGET", "scoring_target.parquet")  # keep all columns

    # small reference sheets, kept whole
    convert_sheet("PRODUCT ATTRIBUTES", "product_attributes.parquet")
    convert_sheet("CATEGORY MAPPING", "category_mapping.parquet")

    total_out = sum(
        os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)
    ) / (1024 * 1024)

    print(f"\n=== Done ===")
    print(f"  Excel:   {excel_size:.1f} MB")
    print(f"  Parquet: {total_out:.1f} MB  ({excel_size/total_out:.1f}x smaller)")
    print(f"\nNow update train_on_real_dataset.py to read from '{OUT_DIR}/' instead")
    print("of the xlsx — loading will drop from minutes to seconds.")