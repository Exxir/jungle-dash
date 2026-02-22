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

### Database expectations

- Supabase table `public."Jun"` with columns `studio`, `date`, `netsales`, and `weekday` (TEXT)

## Deploy

Push to `main` (repo linked to Streamlit Cloud) with `.streamlit/config.toml` + secrets configured in the Streamlit dashboard.
