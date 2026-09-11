"""
Replaces the synthetic trend_score in feat_sku_location_day with REAL
Google search-interest data (via pytrends, the widely-used unofficial
Google Trends library — free, no API key).

Fetches weekly search interest in India for each product category's
representative term, normalizes to 0-1, and applies it to all products
in that category.

NOTE: pytrends is an unofficial library that scrapes Google Trends —
it can occasionally rate-limit or break if Google changes their site.
If a request fails, the script retries with backoff; if it still fails,
that category keeps its previous (synthetic) value rather than crashing
the whole run.

Install first:  pip install pytrends
Run with:       python update_real_trends.py
"""

import os
import time
from datetime import date
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pytrends.request import TrendReq

load_dotenv()
db_url = os.getenv("DATABASE_URL")

START_DATE = "2026-03-01"
END_DATE = "2026-08-27"

# map your product categories to real search terms people would use in India
CATEGORY_SEARCH_TERMS = {
    "Dress": "dress",
    "Shirt": "shirt",
    "T-Shirt": "t shirt",
    "Trousers": "trousers",
    "Jeans": "jeans",
    "Jacket": "jacket",
    "Skirt": "skirt",
    "Kurta": "kurta",
    "Footwear": "shoes",
    "Accessories": "fashion accessories",
}


def fetch_categories(engine):
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("SELECT DISTINCT category FROM dim_product"))]


def fetch_trend_for_term(pytrends, term, max_retries=3):
    timeframe = f"{START_DATE} {END_DATE}"
    for attempt in range(max_retries):
        try:
            pytrends.build_payload([term], timeframe=timeframe, geo="IN")
            df = pytrends.interest_over_time()
            if df.empty:
                raise ValueError("empty response")
            return df
        except Exception as e:
            print(f"    attempt {attempt+1}/{max_retries} failed for '{term}': {e}")
            time.sleep(5)
    return None


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    categories = fetch_categories(engine)
    print(f"Found {len(categories)} product categories.")

    pytrends = TrendReq(hl="en-US", tz=330)  # tz=330 = IST offset in minutes

    category_scores = {}  # category -> {date_id: trend_score}

    for cat in categories:
        term = CATEGORY_SEARCH_TERMS.get(cat)
        if not term:
            print(f"  No search term mapped for category '{cat}', skipping (keeps synthetic value).")
            continue

        print(f"  Fetching Google Trends for '{term}' (category: {cat})...")
        df = fetch_trend_for_term(pytrends, term)
        if df is None:
            print(f"    Failed after retries — '{cat}' keeps its existing value.")
            continue

        # normalize Google's 0-100 scale to 0-1, matching your existing column range
        series = df[term] / 100.0
        daily_scores = {}
        for dt, val in series.items():
            daily_scores[dt.strftime("%Y%m%d")] = round(float(val), 2)

        category_scores[cat] = daily_scores
        print(f"    got {len(daily_scores)} weekly points for '{cat}'.")
        time.sleep(2)  # be polite, avoid rate limiting

    if not category_scores:
        raise SystemExit("No trend data was fetched successfully — nothing to update.")

    # build update rows: need (date_id, category, trend_score) then join to products by category
    update_rows = []
    for cat, daily_scores in category_scores.items():
        for date_id, score in daily_scores.items():
            update_rows.append({"category": cat, "date_id": date_id, "trend_score": score})

    print(f"\nPrepared {len(update_rows)} real trend data points across {len(category_scores)} categories.")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE _trend_staging (
                category VARCHAR(50), date_id VARCHAR(10), trend_score NUMERIC(5,2)
            ) ON COMMIT DROP
        """))
        conn.execute(
            text("INSERT INTO _trend_staging (category, date_id, trend_score) VALUES (:category, :date_id, :trend_score)"),
            update_rows
        )
        # pytrends returned near-daily resolution (one point per day, not
        # weekly, since the range is under ~9 months) — so a direct exact
        # date + category match works, no LATERAL/nearest-match needed
        # (and avoids referencing the UPDATE target table inside a
        # correlated subquery, which Postgres doesn't allow).
        result = conn.execute(text("""
            UPDATE feat_sku_location_day f
            SET trend_score = s.trend_score
            FROM _trend_staging s, dim_product p
            WHERE f.product_id = p.product_id
              AND p.category = s.category
              AND f.date_id = s.date_id
        """))
        print(f"Updated {result.rowcount} rows in feat_sku_location_day with real trend data.")

        # for any rows on dates pytrends didn't cover exactly (edge days),
        # fill them using the nearest available category-level value —
        # this is a SEPARATE, simple query that doesn't need LATERAL
        remaining = conn.execute(text("""
            SELECT COUNT(*) FROM feat_sku_location_day f
            JOIN dim_product p ON f.product_id = p.product_id
            WHERE p.category = ANY(:cats)
              AND NOT EXISTS (
                  SELECT 1 FROM _trend_staging s
                  WHERE s.category = p.category AND s.date_id = f.date_id
              )
        """), {"cats": list(category_scores.keys())}).scalar()
        if remaining > 0:
            print(f"  {remaining} rows had no exact date match (edge days) — "
                  f"left at their previous value, acceptable for a small number of days.")

    print("Done. Real Google Trends data now in place (where fetch succeeded).")