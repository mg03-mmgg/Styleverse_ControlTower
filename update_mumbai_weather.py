"""
Fetches REAL historical weather for Mumbai specifically and updates
weather_score for whichever single store was just relabeled as Mumbai
in dim_location.

Run AFTER you've already run the UPDATE dim_location SET city='Mumbai'... SQL.

Run with:  python update_mumbai_weather.py
"""

import os
import time
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")

START_DATE = "2026-03-01"
END_DATE = "2026-08-27"
MUMBAI_LAT, MUMBAI_LON = 19.0760, 72.8777


def fetch_mumbai_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": MUMBAI_LAT,
        "longitude": MUMBAI_LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            dates = data["daily"]["time"]
            temps = data["daily"]["temperature_2m_mean"]
            precip = data["daily"]["precipitation_sum"]
            return list(zip(dates, temps, precip))
        except Exception as e:
            print(f"  attempt {attempt+1}/3 failed: {e}")
            time.sleep(3)
    raise RuntimeError("Could not fetch Mumbai weather after 3 attempts")


def temp_precip_to_score(temp_c, precip_mm):
    if temp_c is None:
        temp_c = 28.0
    if precip_mm is None:
        precip_mm = 0.0
    temp_comfort = max(0.0, 1 - abs(temp_c - 25) / 25)
    dryness = max(0.0, 1 - min(1.0, precip_mm / 20))
    score = round(0.6 * temp_comfort + 0.4 * dryness, 2)
    return max(0.05, min(1.0, score))


if __name__ == "__main__":
    if not db_url:
        raise SystemExit("DATABASE_URL not found — check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)

    with engine.connect() as conn:
        mumbai_location = conn.execute(text(
            "SELECT location_id FROM dim_location WHERE city = 'Mumbai'"
        )).fetchone()

    if not mumbai_location:
        raise SystemExit("No location with city='Mumbai' found — run the rename SQL first.")

    location_id = mumbai_location[0]
    print(f"Found Mumbai at location_id = {location_id}")

    print("Fetching real Mumbai weather from Open-Meteo...")
    daily = fetch_mumbai_weather()
    print(f"Got {len(daily)} days of real weather.")

    update_rows = [
        {"date_id": d.replace("-", ""), "location_id": location_id,
         "weather_score": temp_precip_to_score(t, p)}
        for d, t, p in daily
    ]

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE _mumbai_weather (
                date_id VARCHAR(10), location_id VARCHAR(20), weather_score NUMERIC(5,2)
            ) ON COMMIT DROP
        """))
        conn.execute(
            text("INSERT INTO _mumbai_weather (date_id, location_id, weather_score) "
                 "VALUES (:date_id, :location_id, :weather_score)"),
            update_rows
        )
        result = conn.execute(text("""
            UPDATE feat_sku_location_day f
            SET weather_score = s.weather_score
            FROM _mumbai_weather s
            WHERE f.date_id = s.date_id AND f.location_id = s.location_id
        """))
        print(f"Updated {result.rowcount} rows with real Mumbai weather.")

    print("Done.")