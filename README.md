# Jungle Dashboard

Simple Streamlit app that connects to Supabase to display monthly net sales per studio.

## Setup

1. Install deps: `pip install -r requirements.txt`
2. Create `.streamlit/secrets.toml` with:
   ```toml
   [general]
   SUPABASE_DB_URL="postgresql://..."
   ```
3. Run locally: `streamlit run app.py`
4. Studio daily dashboard: `streamlit run app_studio_daily.py`

### Database expectations

- Legacy dashboard (`app.py`): Supabase table `public."Jun"` with columns `studio`, `date`, `netsales`, and `weekday`.
- Studio daily dashboard (`app_studio_daily.py`): Supabase table `public.studio_daily_metrics` with columns `studio`, `date`, and `net_sales` (weekday is derived from the date in the app).

## Loading New Studio Metrics

1. Drop the source CSV in `data/JFWTest.csv` (or pass a custom path with `--csv`).
2. Ensure `SUPABASE_DB_URL` is set to the Supabase/Postgres connection string.
3. Run `python scripts/import_jfw_metrics.py --dry-run` to verify parsing.
4. Load the data with `python scripts/import_jfw_metrics.py` (adds/updates rows in `public.studio_daily_metrics`).

## Deploy

Push to `main` (repo linked to Streamlit Cloud) with `.streamlit/config.toml` + secrets configured in the Streamlit dashboard.
