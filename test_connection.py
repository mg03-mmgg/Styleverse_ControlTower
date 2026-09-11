import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

db_url = os.getenv("DATABASE_URL")

if not db_url:
    print("ERROR: DATABASE_URL not found. Check your .env file.")
else:
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"))
            tables = [row[0] for row in result]
            print(f"Connected successfully. Found {len(tables)} tables:")
            for t in tables:
                print(" -", t)
    except Exception as e:
        print("Connection FAILED:")
        print(e)
