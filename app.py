from datetime import date, timedelta
from typing import Sequence, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text


def ensure_date(value, fallback):
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return fallback


def normalize_range(selection, fallback: Tuple[date, date]) -> Tuple[date, date]:
    start, end = fallback
    seq: Sequence[object] = ()

    if isinstance(selection, (list, tuple)):
        seq = selection
    elif selection is not None:
        return (ensure_date(selection, start), ensure_date(selection, end))

    if len(seq) >= 2:
        start = ensure_date(seq[0], start)
        end = ensure_date(seq[1], end)
    elif len(seq) == 1:
        start = end = ensure_date(seq[0], start)

    if start > end:
        start, end = end, start

    return (start, end)


def clamp_date(value: date, lower: date, upper: date) -> date:
    return max(min(value, upper), lower)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    if "date" in view.columns:
        view["date"] = view["date"].dt.strftime("%Y-%m-%d")
    if "weekday" in view.columns:
        view["weekday"] = view["weekday"].fillna("").str[:3]
    return view

st.set_page_config(layout="wide")
st.title("Jungle Dashboard")

engine = create_engine(
    st.secrets["SUPABASE_DB_URL"],
    connect_args={"sslmode": "require"}
)

@st.cache_data(ttl=60)
def load_data():
    query = text("""
        SELECT "studio", "date", "netsales", "weekday"
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

range_input_col, comparison_input_col = st.columns(2)

with range_input_col:
    selected_range = st.date_input(
        "Select date range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Choose start/end dates to sum net sales"
    )

range_tuple = normalize_range(selected_range, (default_start, max_date))
start_date = clamp_date(range_tuple[0], min_date, max_date)
end_date = clamp_date(range_tuple[1], min_date, max_date)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

if start_ts > end_ts:
    start_ts, end_ts = end_ts, start_ts

filtered_selection = studio_df[
    (studio_df["date"] >= start_ts) &
    (studio_df["date"] <= end_ts)
]
filtered_df = pd.DataFrame(filtered_selection).copy()

if filtered_df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

range_sales = filtered_df["netsales"].sum()

range_length_days = max((end_date - start_date).days, 0)
comparison_end_default = clamp_date(start_date - timedelta(days=1), min_date, max_date)
comparison_start_default = clamp_date(
    comparison_end_default - timedelta(days=range_length_days),
    min_date,
    comparison_end_default
)

with comparison_input_col:
    comparison_selection = st.date_input(
        "Comparison date range",
        value=(comparison_start_default, comparison_end_default),
        min_value=min_date,
        max_value=max_date,
        help="Pick another range to compare against"
    )

comparison_tuple = normalize_range(
    comparison_selection,
    (comparison_start_default, comparison_end_default)
)
comp_start_date = clamp_date(comparison_tuple[0], min_date, max_date)
comp_end_date = clamp_date(comparison_tuple[1], min_date, max_date)

comp_start_ts = pd.Timestamp(comp_start_date)
comp_end_ts = pd.Timestamp(comp_end_date)

comparison_selection_df = studio_df[
    (studio_df["date"] >= comp_start_ts) &
    (studio_df["date"] <= comp_end_ts)
]
comparison_df = pd.DataFrame(comparison_selection_df).copy()
comparison_sales = comparison_df["netsales"].sum() if not comparison_df.empty else 0.0
comparison_delta = None if comparison_df.empty else range_sales - comparison_sales

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.metric(
        label="Net sales (selected range)",
        value=f"${range_sales:,.0f}"
    )

with col2:
    st.metric(
        label="Comparison net sales",
        value=f"${comparison_sales:,.0f}",
        delta=None if comparison_delta is None else comparison_delta
    )

st.subheader("Selected Range Details")
range_view = filtered_df.sort_values("date", ascending=False)
st.dataframe(format_table(range_view))

st.subheader("Comparison Range Details")
if comparison_df.empty:
    st.info("No data available for the comparison range.")
else:
    comparison_view = comparison_df.sort_values("date", ascending=False)
    st.dataframe(format_table(comparison_view))
