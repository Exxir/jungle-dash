from datetime import date, timedelta
from typing import Tuple, cast

import pandas as pd
import streamlit as st
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

default_start = max(max_date - timedelta(days=13), min_date)

selected_range = st.date_input(
    "Select date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Choose start/end dates to sum net sales"
)

range_tuple: Tuple[date, date]

if isinstance(selected_range, tuple):
    if len(selected_range) >= 2:
        range_tuple = (selected_range[0], selected_range[1])
    elif len(selected_range) == 1:
        range_tuple = (selected_range[0], selected_range[0])
    else:
        range_tuple = (max_date, max_date)
else:
    range_tuple = (selected_range, selected_range)

start_date, end_date = range_tuple

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

if start_ts > end_ts:
    start_ts, end_ts = end_ts, start_ts

filtered_slice = studio_df[
    (studio_df["date"] >= start_ts) &
    (studio_df["date"] <= end_ts)
].copy()
filtered_df = cast(pd.DataFrame, filtered_slice)

if filtered_df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

# --- Range Calculation ---
range_sales = filtered_df["netsales"].sum()

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label="Net sales (selected range)",
        value=f"${range_sales:,.0f}"
    )

with col2:
    st.subheader("Selected Range Details")
    range_view = filtered_df.sort_values("date", ascending=False)
    st.dataframe(range_view)
