"""
Generates the remaining 4 master tables and inserts them into Supabase:
 - dim_supplier   (5 rows)
 - dim_customer   (1000 rows)
 - dim_promotion  (15 rows)
 - dim_calendar   (180 days)

Run with:  python generate_masters_part2.py
"""

import os
import random
from datetime import date, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(11)

# ------------------------------------------------------------------
# dim_supplier
# ------------------------------------------------------------------

def generate_suppliers():
    names = ["Vastra Textiles", "Weavemark Industries", "Indigo Threadworks",
              "Sunrise Garments Co", "Metro Apparel Manufacturing"]
    locations = ["Tiruppur", "Surat", "Ludhiana", "Noida", "Bengaluru"]

    rows = []
    for i, (name, loc) in enumerate(zip(names, locations), start=1):
        rows.append({
            "supplier_id": f"SUP{i:03d}",
            "supplier_name": name,
            "location": loc,
            "normal_lead_days": random.randint(20, 45),
            "fast_lead_days": random.randint(7, 15),
            "minimum_order_qty": random.choice([200, 300, 500, 750]),
            "quality_score": round(random.uniform(3.5, 4.9), 2),
            "on_time_rate": round(random.uniform(0.72, 0.97), 3),
            "source_system": "synthetic_generator",
        })
    return rows


# ------------------------------------------------------------------
# dim_customer
# ------------------------------------------------------------------

def generate_customers(n=1000):
    loyalty_tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    regions = ["North", "South", "East", "West"]
    channels = ["Store", "Online", "App", "Marketplace"]

    rows = []
    for i in range(1, n + 1):
        tier = random.choices(loyalty_tiers, weights=[45, 30, 18, 7])[0]
        created = date(2024, 1, 1) + timedelta(days=random.randint(0, 900))
        rows.append({
            "customer_id": f"CUST{i:05d}",
            "loyalty_tier": tier,
            "home_region": random.choice(regions),
            "preferred_channel": random.choice(channels),
            "consent_flag": random.choices([True, False], weights=[85, 15])[0],
            "created_date": created.isoformat(),
            "source_system": "synthetic_generator",
        })
    return rows


# ------------------------------------------------------------------
# dim_promotion
# ------------------------------------------------------------------

def generate_promotions(n=15):
    promo_types = ["Seasonal Sale", "Flash Sale", "Festival Offer",
                    "Clearance", "Bundle Deal", "Loyalty Exclusive"]
    funding = ["Brand Budget", "Vendor Co-funded", "Marketing Budget", "Clearance Fund"]

    rows = []
    start_base = date(2025, 9, 1)
    for i in range(1, n + 1):
        ptype = random.choice(promo_types)  # pick type ONCE, reuse for name
        start = start_base + timedelta(days=random.randint(0, 300))
        dur = random.choice([3, 5, 7, 10, 14, 21])
        end = start + timedelta(days=dur)
        rows.append({
            "promotion_id": f"PROMO{i:03d}",
            "promotion_name": f"{ptype} {start.strftime('%b %Y')}",
            "promotion_type": ptype,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "discount_percentage": random.choice([10, 15, 20, 25, 30, 40]),
            "funding_source": random.choice(funding),
            "source_system": "synthetic_generator",
        })
    return rows


# ------------------------------------------------------------------
# dim_calendar (180 days)
# ------------------------------------------------------------------

def generate_calendar(start=date(2026, 3, 1), days=180):
    season_by_month = ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer",
                        "Summer", "Summer", "Autumn", "Autumn", "Autumn", "Winter"]

    festival_dates = set()
    for _ in range(6):
        festival_dates.add(start + timedelta(days=random.randint(0, days - 1)))

    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        rows.append({
            "date_id": d.strftime("%Y%m%d"),
            "full_date": d.isoformat(),
            "day": d.strftime("%A"),
            "week": d.isocalendar()[1],
            "month": d.month,
            "quarter": (d.month - 1) // 3 + 1,
            "year": d.year,
            "holiday_flag": d.weekday() >= 5,
            "festival_flag": d in festival_dates,
            "season": season_by_month[d.month - 1],
        })
    return rows


# ------------------------------------------------------------------
# INSERT
# ------------------------------------------------------------------

def insert_rows(engine, table, rows):
    if not rows:
        return
    columns = rows[0].keys()
    col_list = ", ".join(columns)
    val_list = ", ".join(f":{c}" for c in columns)
    stmt = text(f"INSERT INTO {table} ({col_list}) VALUES ({val_list}) ON CONFLICT DO NOTHING")
    with engine.begin() as conn:
        conn.execute(stmt, rows)
    print(f"Inserted {len(rows)} rows into {table}")


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)


    insert_rows(engine, "dim_supplier", generate_suppliers())
    insert_rows(engine, "dim_customer", generate_customers(1000))
    insert_rows(engine, "dim_promotion", generate_promotions(15))
    insert_rows(engine, "dim_calendar", generate_calendar())

    print("Done.")