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
        SELECT "Studio", "Date", "NetSales"
        FROM public."Jun"
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

df = load_data()

# --- Studio Selector ---
studios = sorted(df["Studio"].unique())
selected_studio = st.selectbox("Select Studio", studios)

studio_df = df[df["Studio"] == selected_studio].copy()
studio_df["Date"] = pd.to_datetime(studio_df["Date"])

# --- MTD Calculation ---
latest_date = studio_df["Date"].max()
month_start = latest_date.replace(day=1)

mtd_sales = studio_df[
    (studio_df["Date"] >= month_start) &
    (studio_df["Date"] <= latest_date)
]["NetSales"].sum()

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label="MTD Net Sales",
        value=f"${mtd_sales:,.0f}"
    )

with col2:
    st.subheader("Last 14 Days")
    last_14 = studio_df.sort_values("Date", ascending=False).head(14)
    st.dataframe(last_14)
