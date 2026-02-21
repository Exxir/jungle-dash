import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

st.title("Jungle Dashboard")

engine = create_engine(st.secrets["SUPABASE_DB_URL"])

@st.cache_data(ttl=60)
def load_count():
    with engine.connect() as conn:
        result = conn.execute(text('select count(*) from public."Jun"'))
        return result.scalar()

count = load_count()

st.write("Rows in Jun table:", count)
