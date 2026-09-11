# StyleVerse Control Tower — AI Prototype

An AI-led retail decision engine prototype for StyleVerse Global, built for the
PwC Challenge 8th Edition case study. Simulates demand forecasting, inventory
allocation, and markdown recommendations on synthetic data across a 27-table
schema, with a planner dashboard for accepting/rejecting AI recommendations.

## Stack
- **Database:** Postgres (hosted on Supabase, shared across the team)
- **Data generation & pipeline:** Python (pandas, SQLAlchemy)
- **ML models:** scikit-learn, XGBoost, LightGBM
- **API (upcoming):** FastAPI
- **Frontend (upcoming):** slider-driven what-if dashboard

## Project structure
```
sql/                 CREATE TABLE schema (27 tables)
data_generation/      Scripts that generate synthetic dummy data and load it into Postgres
utils/                 Shared helpers, connection tests
models/                ML training scripts + saved model files (fills in during model phase)
decision_engine/       Rule-based / optimization logic for recommendations
api/                    Backend API that the frontend slider calls
frontend/               The dashboard/website
```

## Setup (for each team member)

1. Clone the repo.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in the real `DATABASE_URL`
   (get this from whoever set up the Supabase project — never commit `.env`).
5. Test the connection:
   ```
   python utils/test_connection.py
   ```
   You should see "Connected successfully" and a list of 27 tables.

## Running the data generators
Run these **in order** from the project root (later scripts depend on earlier ones):
```
python data_generation/generate_products_locations.py
python data_generation/generate_masters_part2.py
python data_generation/generate_bridge_product_supplier.py
python data_generation/generate_fact_sales_line.py
python data_generation/generate_fact_inventory_snapshot.py
python data_generation/generate_fact_return_line.py
python data_generation/generate_po_and_capacity.py
python data_generation/generate_remaining_operational.py
python data_generation/generate_feat_sku_location_day.py
python data_generation/generate_remaining_features.py
```

## Schema
Full 27-table schema lives in `sql/schema.sql`. Run once against a fresh
Supabase/Postgres instance to create all tables before running any generator.

## Status
- [x] Schema created (27 tables)
- [x] 7 master tables populated
- [x] 10 operational tables populated
- [x] 4 feature/label tables populated
- [ ] ML models trained (`ai_model_registry`, `ai_demand_forecast`)
- [ ] Decision engine (`ai_recommendation`)
- [ ] Planner decision + execution simulation
- [ ] API
- [ ] Frontend dashboard with what-if sliders
