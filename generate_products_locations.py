"""
Generates dim_product (1000 rows) and dim_location (12 rows) with real
combinatorial variety, and inserts them into the Supabase database using
the DATABASE_URL from your .env file.

Run with:  python generate_products_locations.py
"""

import os
import random
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

random.seed(42)  # remove or change this if you want a different random set each run

# ------------------------------------------------------------------
# 1. GENERATE dim_product
# ------------------------------------------------------------------

brands = ["SpeedStyle", "UrbanEdge", "Maison Luxe", "EcoWeave", "ThreadBasics"]

categories = {
    "Dress": ["Casual Dress", "Party Dress", "Maxi Dress", "Wrap Dress"],
    "Shirt": ["Formal Shirt", "Casual Shirt", "Printed Shirt", "Denim Shirt"],
    "T-Shirt": ["Graphic Tee", "Plain Tee", "Polo", "Oversized Tee"],
    "Trousers": ["Chinos", "Formal Trousers", "Joggers", "Cargo"],
    "Jeans": ["Skinny", "Straight", "Bootcut", "Mom Fit"],
    "Jacket": ["Denim Jacket", "Bomber", "Blazer", "Puffer"],
    "Skirt": ["Mini Skirt", "Midi Skirt", "Pleated Skirt"],
    "Kurta": ["Cotton Kurta", "Printed Kurta", "Embroidered Kurta"],
    "Footwear": ["Sneakers", "Sandals", "Formal Shoes", "Boots"],
    "Accessories": ["Belt", "Scarf", "Bag", "Cap"],
}

sizes_apparel = ["XS", "S", "M", "L", "XL", "XXL"]
sizes_footwear = ["6", "7", "8", "9", "10", "11"]
colours = ["Red", "Black", "White", "Navy", "Beige", "Olive", "Mustard", "Pink",
           "Grey", "Maroon", "Teal", "Lavender", "Rust", "Charcoal", "Cream"]
materials = ["Cotton", "Polyester", "Linen", "Denim", "Rayon", "Wool Blend",
             "Silk Blend", "Leather", "Synthetic", "Viscose"]
seasons = ["Spring", "Summer", "Autumn", "Winter", "All-Season"]
lifecycle_options = ["Active", "Active", "Active", "Clearance", "New Launch", "Discontinued"]

# brand price tiers: (min_mrp, max_mrp, (cost_ratio_min, cost_ratio_max))
brand_tier = {
    "Maison Luxe":  (4500, 25000, (0.35, 0.45)),
    "UrbanEdge":    (1200, 4500,  (0.40, 0.55)),
    "SpeedStyle":   (499,  2499,  (0.35, 0.50)),
    "EcoWeave":     (999,  3999,  (0.42, 0.55)),
    "ThreadBasics": (299,  1499,  (0.45, 0.60)),
}

def generate_products(n=1000):
    seen_combos = set()
    rows = []
    pid_counter = 1001

    while len(rows) < n:
        brand = random.choice(brands)
        category = random.choice(list(categories.keys()))
        subcategory = random.choice(categories[category])
        colour = random.choice(colours)
        material = random.choice(materials)
        season = random.choice(seasons)
        size = random.choice(sizes_footwear if category == "Footwear" else sizes_apparel)

        combo_key = (brand, category, subcategory, colour, material, size)
        if combo_key in seen_combos:
            continue  # avoid exact duplicate attribute combinations
        seen_combos.add(combo_key)

        lo, hi, cost_ratio = brand_tier[brand]
        mrp = round(random.uniform(lo, hi), -1) + random.choice([9, 49, 99])
        cost_pct = random.uniform(*cost_ratio)
        unit_cost = round(mrp * cost_pct, 2)

        style_id = f"ST{random.randint(100, 999)}"
        product_id = f"P{pid_counter}"
        sku_code = f"{brand[:2].upper()}{category[:2].upper()}{pid_counter}{colour[:2].upper()}"
        launch_year = random.choice([2025, 2026])
        launch_month = random.randint(1, 8)
        launch_day = random.randint(1, 28)
        launch_date = f"{launch_year}-{launch_month:02d}-{launch_day:02d}"

        rows.append({
            "product_id": product_id,
            "sku_code": sku_code,
            "brand": brand,
            "category": category,
            "subcategory": subcategory,
            "style_id": style_id,
            "size": size,
            "colour": colour,
            "material": material,
            "season": season,
            "launch_date": launch_date,
            "mrp": mrp,
            "unit_cost": unit_cost,
            "lifecycle_status": random.choice(lifecycle_options),
            "source_system": "synthetic_generator",
        })
        pid_counter += 1

    return rows


# ------------------------------------------------------------------
# 2. GENERATE dim_location
# ------------------------------------------------------------------

cities_regions = [
    ("Mumbai", "West"), ("Delhi", "North"), ("Bengaluru", "South"), ("Chennai", "South"),
    ("Kolkata", "East"), ("Hyderabad", "South"), ("Pune", "West"), ("Ahmedabad", "West"),
    ("Jaipur", "North"), ("Lucknow", "North"), ("Patna", "East"), ("Chandigarh", "North"),
    ("Kochi", "South"), ("Indore", "West"), ("Bhopal", "West"),
]
clusters = ["CL_A", "CL_B", "CL_C", "CL_D"]

def generate_locations():
    rows = []
    loc_counter = 1
    chosen = random.sample(cities_regions, 10)

    for city, region in chosen:
        loc_id = f"ST{loc_counter:03d}"
        rows.append({
            "location_id": loc_id,
            "location_type": "Store",
            "location_name": f"StyleVerse {city}",
            "city": city,
            "region": region,
            "cluster_id": random.choice(clusters),
            "channel": "Offline",
            "capacity": random.randint(2000, 8000),
            "floor_area": round(random.uniform(1500, 6000), 1),
            "source_system": "synthetic_generator",
        })
        loc_counter += 1

    rows.append({
        "location_id": "DC001", "location_type": "Distribution Centre",
        "location_name": "Central DC Gurugram", "city": "Gurugram", "region": "North",
        "cluster_id": None, "channel": "Warehouse", "capacity": 50000,
        "floor_area": 40000.0, "source_system": "synthetic_generator",
    })
    rows.append({
        "location_id": "WEB01", "location_type": "Online",
        "location_name": "India Online", "city": "National", "region": "National",
        "cluster_id": None, "channel": "Online", "capacity": None,
        "floor_area": None, "source_system": "synthetic_generator",
    })

    return rows


# ------------------------------------------------------------------
# 3. INSERT INTO SUPABASE
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

    engine = create_engine(db_url)

    products = generate_products(1000)
    locations = generate_locations()

    insert_rows(engine, "dim_product", products)
    insert_rows(engine, "dim_location", locations)

    print("Done.")