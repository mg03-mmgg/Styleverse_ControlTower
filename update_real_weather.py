"""
Replaces the synthetic weather_score in feat_sku_location_day with REAL
historical weather, fetched from Open-Meteo's free archive API (no API
key required) for your actual store cities and actual date range.

Since your synthetic calendar (Mar 1 - Aug 27, 2026) is already in the
past relative to today, Open-Meteo's historical archive has real
recorded weather for these exact dates.

weather_score is computed from real temperature + precipitation:
  - Closer to 25°C = more "comfortable shopping weather" = higher score
  - More rainfall = lower score
This keeps the same 0-1 scale as before, but now derived from real data.

Run with:  python update_real_weather.py
"""

import os
import time
import requests
from datetime import date
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

START_DATE = "2026-03-01"
END_DATE = "2026-08-27"

# real coordinates for the candidate store cities (matches generate_masters_part2.py)
CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Patna": (25.5941, 85.1376),
    "Chandigarh": (30.7333, 76.7794),
    "Kochi": (9.9312, 76.2673),
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Gurugram": (28.4595, 77.0266),  # for DC001
}


def fetch_real_locations(engine):
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT location_id, city FROM dim_location WHERE city IS NOT NULL AND city != 'National'"
        ))
        return result.fetchall()


def fetch_weather_for_city(city, lat, lon, max_retries=3):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Asia/Kolkata",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            dates = data["daily"]["time"]
            temps = data["daily"]["temperature_2m_mean"]
            precip = data["daily"]["precipitation_sum"]
            return list(zip(dates, temps, precip))
        except Exception as e:
            print(f"    attempt {attempt+1}/{max_retries} failed for {city}: {e}")
            time.sleep(3)
    raise RuntimeError(f"Could not fetch weather for {city} after {max_retries} attempts")


def temp_precip_to_score(temp_c, precip_mm):
    if temp_c is None:
        temp_c = 28.0
    if precip_mm is None:
        precip_mm = 0.0
    temp_comfort = max(0.0, 1 - abs(temp_c - 25) / 25)
    dryness = max(0.0, 1 - min(1.0, precip_mm / 20))
    score = round(0.6 * temp_comfort + 0.4 * dryness, 2)
    return max(0.05, min(1.0, score))  # keep within same bounds as the old synthetic version


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    locations = fetch_real_locations(engine)
    print(f"Found {len(locations)} locations with real cities.")

    # fetch weather once per unique city (not per store — saves API calls)
    unique_cities = sorted(set(city for _, city in locations))
    print(f"Fetching real weather for {len(unique_cities)} unique cities: {unique_cities}")

    city_weather = {}
    for city in unique_cities:
        if city not in CITY_COORDS:
            print(f"  WARNING: no coordinates for '{city}', skipping.")
            continue
        lat, lon = CITY_COORDS[city]
        print(f"  Fetching {city}...")
        daily = fetch_weather_for_city(city, lat, lon)
        city_weather[city] = {
            d.replace("-", ""): temp_precip_to_score(t, p) for d, t, p in daily
        }
        print(f"    got {len(daily)} days of real weather.")
        time.sleep(1)  # be polite to the free API

    # build update rows: (date_id, location_id, weather_score)
    update_rows = []
    for location_id, city in locations:
        if city not in city_weather:
            continue
        for date_id, score in city_weather[city].items():
            update_rows.append({"date_id": date_id, "location_id": location_id, "weather_score": score})

    print(f"\nPrepared {len(update_rows)} real weather updates.")

    # bulk UPDATE via a temp staging table (fast, single join-based update)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE _weather_staging (
                date_id VARCHAR(10), location_id VARCHAR(20), weather_score NUMERIC(5,2)
            ) ON COMMIT DROP
        """))
        conn.execute(
            text("INSERT INTO _weather_staging (date_id, location_id, weather_score) VALUES (:date_id, :location_id, :weather_score)"),
            update_rows
        )
        result = conn.execute(text("""
            UPDATE feat_sku_location_day f
            SET weather_score = s.weather_score
            FROM _weather_staging s
            WHERE f.date_id = s.date_id AND f.location_id = s.location_id
        """))
        print(f"Updated {result.rowcount} rows in feat_sku_location_day with real weather.")

    print("Done. Real weather data now in place.")