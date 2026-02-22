from datetime import date, timedelta
from typing import Sequence, Tuple, cast

import altair as alt
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
        view["date"] = view["date"].dt.strftime("%m-%d-%y")
    if "weekday" in view.columns:
        view["weekday"] = view["weekday"].fillna("").str[:3]
    if "netsales" in view.columns:
        view["netsales"] = view["netsales"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    return view


def build_chart_data(df: pd.DataFrame, series_label: str, range_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "netsales": pd.Series(dtype=float),
            "series": pd.Series(dtype="string"),
            "weekday": pd.Series(dtype="string"),
            "range_label": pd.Series(dtype="string"),
        })
    grouped = (
        df.groupby("date")["netsales"].sum().reset_index()
    )
    grouped["series"] = series_label
    grouped["weekday"] = grouped["date"].dt.strftime("%a")
    grouped["range_label"] = range_label
    return grouped

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
        "Current",
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
comparison_last_year = (
    clamp_date(start_date - timedelta(days=365), min_date, max_date),
    clamp_date(end_date - timedelta(days=365), min_date, max_date)
)

with comparison_input_col:
    comparison_selection = st.date_input(
        "Comparison",
        value=comparison_last_year,
        min_value=min_date,
        max_value=max_date,
        help="Pick another range to compare against"
    )

comparison_tuple = normalize_range(
    comparison_selection,
    comparison_last_year
)
comp_start_date = clamp_date(comparison_tuple[0], min_date, max_date)
comp_end_date = clamp_date(comparison_tuple[1], min_date, max_date)

comp_start_ts = pd.Timestamp(comp_start_date)
comp_end_ts = pd.Timestamp(comp_end_date)

comparison_selection_df = studio_df[
    (studio_df["date"] >= comp_start_ts) &
    (studio_df["date"] <= comp_end_ts)
]
comparison_df: pd.DataFrame = pd.DataFrame(comparison_selection_df).copy()
comparison_sales = comparison_df["netsales"].sum() if not comparison_df.empty else 0.0
comparison_delta_pct = None
if comparison_df.empty or comparison_sales == 0:
    comparison_delta_pct = None
else:
    diff_pct = ((range_sales - comparison_sales) / comparison_sales) * 100
    comparison_delta_pct = f"{diff_pct:+.1f}%"

forecast_df: pd.DataFrame = pd.DataFrame(columns=studio_df.columns)
forecast_days = 7
if not comparison_df.empty:
    yoy_multiplier = range_sales / comparison_sales if comparison_sales else 1.0
    shifted = comparison_df.copy()
    if not isinstance(shifted, pd.DataFrame):
        shifted = pd.DataFrame(shifted)
    shifted["date"] = pd.to_datetime(shifted["date"]) + pd.Timedelta(days=365)
    cutoff = pd.Timestamp(end_date)
    shifted_future = shifted[shifted["date"] > cutoff]
    if shifted_future.empty:
        shifted_future = shifted.sort_values(by="date")
        if not shifted_future.empty:
            future_dates = pd.date_range(
                start=cutoff + pd.Timedelta(days=1),
                periods=len(shifted_future),
                freq="D"
            )
            shifted_future = shifted_future.assign(date=future_dates)
    shifted_future = shifted_future.sort_values(by="date").head(forecast_days)  # type: ignore[arg-type]
    if not shifted_future.empty:
        shifted_future["netsales"] = shifted_future["netsales"] * yoy_multiplier
        shifted_future["weekday"] = shifted_future["date"].dt.strftime("%a")
        forecast_df = shifted_future
forecast_df = cast(pd.DataFrame, forecast_df)
forecast_label = ""
forecast_series_label = ""
if not forecast_df.empty:
    forecast_start = forecast_df["date"].min().date()
    forecast_end = forecast_df["date"].max().date()
    forecast_label = f"{forecast_start:%m-%d-%y} – {forecast_end:%m-%d-%y}"
    forecast_series_label = f"Forecast {forecast_label}"

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
        delta=comparison_delta_pct
    )

tab_tables, tab_chart = st.tabs(["Tables", "Line Chart"])

with tab_tables:
    st.subheader("Selected Range Details")
    range_view = filtered_df.sort_values("date", ascending=False)
    st.dataframe(format_table(range_view))

    st.subheader("Comparison Range Details")
    if comparison_df.empty:
        st.info("No data available for the comparison range.")
    else:
        comparison_view = comparison_df.sort_values("date", ascending=False)
        st.dataframe(format_table(comparison_view))

    st.subheader("Forecast (next 7 days)")
    if forecast_df.empty:
        st.info("No forecast available yet.")
    else:
        forecast_view = forecast_df.sort_values("date")
        st.dataframe(format_table(forecast_view))

with tab_chart:
    selected_label = f"{start_date:%m-%d-%y} – {end_date:%m-%d-%y}"
    comparison_label = f"{comp_start_date:%m-%d-%y} – {comp_end_date:%m-%d-%y}"

    selected_series_label = f"Current {selected_label}"
    comparison_series_label = f"Comparison {comparison_label}"

    selected_chart_df = build_chart_data(filtered_df, selected_series_label, selected_label)
    comparison_chart_df = build_chart_data(comparison_df, comparison_series_label, comparison_label)
    chart_frames = [selected_chart_df, comparison_chart_df]
    legend_order = [selected_series_label, comparison_series_label]

    if not forecast_df.empty:
        forecast_chart_df = build_chart_data(forecast_df, forecast_series_label, forecast_label)
        chart_frames.append(forecast_chart_df)
        legend_order.append(forecast_series_label)

    chart_df = pd.concat(chart_frames, ignore_index=True)

    if chart_df.empty:
        st.info("Not enough data to render the chart.")
    else:
        weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "weekday:N",
                    title="Weekday",
                    sort=weekday_order
                ),
                y=alt.Y("netsales:Q", title="Net sales"),
                color=alt.Color(
                    "series:N",
                    title="",
                    scale=alt.Scale(domain=legend_order),
                    legend=alt.Legend(
                        orient="bottom",
                        direction="horizontal",
                        labelLimit=0,
                        symbolType="circle"
                    )
                ),
                tooltip=[
                    alt.Tooltip("series:N", title="Range"),
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("weekday:N", title="Weekday"),
                    alt.Tooltip("netsales:Q", title="Net sales", format="$.0f"),
                    alt.Tooltip("range_label:N", title="Date range")
                ]
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
