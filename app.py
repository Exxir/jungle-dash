from datetime import date, timedelta
from typing import Optional, Sequence, Tuple, cast

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
    if "comparison_pct" in view.columns:
        view["comparison_pct"] = view["comparison_pct"].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else ""
        )
    if "comparison_date" in view.columns:
        view["comparison_date"] = pd.to_datetime(view["comparison_date"]).dt.strftime("%m-%d-%y")
    return view


def add_comparison_metrics(current_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    result = current_df.copy()
    result["comparison_date"] = pd.NaT
    result["comparison_pct"] = pd.NA

    if comparison_df.empty:
        return result

    comp_series = comparison_df.groupby("date")["netsales"].sum().sort_index()
    comp_index = pd.DatetimeIndex(comp_series.index)

    if len(comp_index) == 0:
        return result

    for idx, row in result.iterrows():
        raw_date = row["date"]
        if raw_date is None:
            continue
        target_ts = pd.to_datetime(raw_date)
        candidate = target_ts - pd.DateOffset(years=1)
        matched_ts = closest_timestamp(comp_index, candidate)
        comp_value = float(comp_series.loc[matched_ts]) if matched_ts in comp_series.index else None
        if comp_value is None or comp_value == 0:
            pct = pd.NA
        else:
            pct = ((row["netsales"] - comp_value) / comp_value) * 100
        result.at[idx, "comparison_date"] = matched_ts
        result.at[idx, "comparison_pct"] = pct

    return result


def prepare_display_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    if "netsales" in table.columns and "comparison_pct" in table.columns:
        pct_column = table.pop("comparison_pct")
        insert_at = list(table.columns).index("netsales") + 1 if "netsales" in table.columns else len(table.columns)
        table.insert(insert_at, "Δ% vs comparison", pct_column)
    return table


def render_table(table: pd.DataFrame, tooltip_value_col: Optional[str] = None, tooltip_source_col: Optional[str] = None):
    display_table = table.copy()
    styled = None
    if (
        tooltip_value_col and tooltip_source_col and
        tooltip_value_col in display_table.columns and
        tooltip_source_col in display_table.columns
    ):
        tooltip_series = display_table[tooltip_source_col]
        display_table = display_table.drop(columns=[tooltip_source_col])
        tooltip_frame = pd.DataFrame("", index=display_table.index, columns=display_table.columns)
        tooltip_frame[tooltip_value_col] = tooltip_series.apply(
            lambda src: f"Comparison date: {src}" if pd.notna(src) and src else ""
        )
        styled = display_table.style.set_tooltips(tooltip_frame)

    st.dataframe(styled if styled is not None else display_table, use_container_width=True)


def closest_timestamp(index: pd.DatetimeIndex, candidate: pd.Timestamp) -> pd.Timestamp:
    if len(index) == 0:
        return candidate
    first_ts = pd.Timestamp(index[0])  # type: ignore[index]
    last_ts = pd.Timestamp(index[-1])  # type: ignore[index]
    if candidate <= first_ts:
        return first_ts
    if candidate >= last_ts:
        return last_ts
    pos = index.get_indexer([candidate], method="nearest")[0]
    return pd.Timestamp(index[pos])  # type: ignore[index]


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
yoy_multiplier = 1.0
comparison_delta_pct = None
if comparison_df.empty or comparison_sales == 0:
    comparison_delta_pct = None
else:
    diff_pct = ((range_sales - comparison_sales) / comparison_sales) * 100
    comparison_delta_pct = f"{diff_pct:+.1f}%"
    yoy_multiplier = range_sales / comparison_sales if comparison_sales else 1.0

history_series = studio_df.groupby("date")["netsales"].sum().sort_index()
history_index: pd.DatetimeIndex = pd.DatetimeIndex(history_series.index)
weekday_index_map = {}
if len(history_index) > 0:
    history_weekday_series = pd.Series(history_index, index=history_index).dt.weekday
    for weekday in range(7):
        mask = history_weekday_series == weekday
        if mask.any():
            weekday_index_map[weekday] = history_weekday_series.index[mask]

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

tab_current, tab_chart, tab_forecast = st.tabs(["Current", "Line Chart", "Forecast"])

with tab_current:
    st.subheader("Selected Range Details")
    range_view = filtered_df.sort_values("date", ascending=False)
    range_view = add_comparison_metrics(range_view, comparison_df)
    formatted_range = format_table(range_view)
    range_display = prepare_display_table(formatted_range)
    render_table(range_display, "Δ% vs comparison", "comparison_date")

    st.subheader("Comparison Range Details")
    if comparison_df.empty:
        st.info("No data available for the comparison range.")
    else:
        comparison_view = comparison_df.sort_values("date", ascending=False)
        render_table(format_table(comparison_view))

with tab_chart:
    selected_label = f"{start_date:%m-%d-%y} – {end_date:%m-%d-%y}"
    comparison_label = f"{comp_start_date:%m-%d-%y} – {comp_end_date:%m-%d-%y}"

    selected_series_label = f"Current {selected_label}"
    comparison_series_label = f"Comparison {comparison_label}"

    selected_chart_df = build_chart_data(filtered_df, selected_series_label, selected_label)
    comparison_chart_df = build_chart_data(comparison_df, comparison_series_label, comparison_label)
    chart_frames = [selected_chart_df, comparison_chart_df]
    legend_order = [selected_series_label, comparison_series_label]

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

with tab_forecast:
    if history_series.empty:
        st.info("Not enough historical data to project future sales.")
    else:
        future_min = end_date + timedelta(days=1)
        future_max = max_date + timedelta(days=365)
        if future_min > future_max:
            st.info("Extend your dataset to enable future projections.")
        else:
            default_end = min(future_max, future_min + timedelta(days=13))
            forecast_range_input = st.date_input(
                "Forecast range",
                value=(future_min, default_end),
                min_value=future_min,
                max_value=future_max,
                help="Select future dates to view projected net sales"
            )

            normalized_forecast_range = normalize_range(
                forecast_range_input,
                (future_min, default_end)
            )
            forecast_start = clamp_date(normalized_forecast_range[0], future_min, future_max)
            forecast_end = clamp_date(normalized_forecast_range[1], future_min, future_max)

            forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq="D")
            forecast_rows = []

            for ts in forecast_dates:
                target_ts = cast(pd.Timestamp, pd.Timestamp(ts))
                candidate = target_ts - pd.DateOffset(years=1)
                target_weekday = int(target_ts.dayofweek)
                weekday_history = weekday_index_map.get(target_weekday)

                if weekday_history is not None and len(weekday_history) > 0:
                    source_timestamp = closest_timestamp(weekday_history, candidate)
                else:
                    source_timestamp = closest_timestamp(history_index, candidate)

                base_value = float(history_series.loc[source_timestamp])
                projected = base_value * yoy_multiplier

                forecast_rows.append(
                    {
                        "date": target_ts,
                        "weekday": target_ts.strftime("%a"),
                        "netsales": projected,
                        "studio": selected_studio,
                        "source_date": source_timestamp.date()
                    }
                )

            forecast_view = pd.DataFrame(forecast_rows)
            if forecast_view.empty:
                st.info("No forecast data for the selected range.")
            else:
                render_table(format_table(forecast_view))
