import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

st.set_page_config(layout="wide")
st.title("Jungle Dashboard")

engine = create_engine(
    st.secrets["SUPABASE_DB_URL"],
    connect_args={"sslmode": "require"}
)

@st.cache_data(ttl=60)
def load_data():
    query = text("""
        SELECT "studio", "date", "netsales"
        FROM public."Jun"
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

df = load_data()

# --- Studio Selector ---
studios = sorted(df["studio"].unique())
selected_studio = st.selectbox("Select studio", studios)

studio_df = df[df["studio"] == selected_studio].copy()
studio_df["date"] = pd.to_datetime(studio_df["date"])

min_date = studio_df["date"].min().date()
max_date = studio_df["date"].max().date()

selected_date = st.date_input(
    "Select date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

selected_ts = pd.Timestamp(selected_date)
filtered_df = studio_df[studio_df["date"] <= selected_ts]

if filtered_df.empty:
    st.warning("No data available for the selected date.")
    st.stop()

# --- MTD Calculation ---
latest_date = filtered_df["date"].max()
month_start = latest_date.replace(day=1)

mtd_sales = filtered_df[
    (filtered_df["date"] >= month_start) &
    (filtered_df["date"] <= latest_date)
]["netsales"].sum()

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label="MTD net sales",
        value=f"${mtd_sales:,.0f}"
    )

with col2:
    st.subheader("Last 14 Days")
    last_14 = filtered_df.sort_values("date", ascending=False).head(14)
    st.dataframe(last_14)
