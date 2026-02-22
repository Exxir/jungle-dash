#!/usr/bin/env python3
"""Utility to clean and load studio metrics CSV data into Supabase."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import Column, Date, Integer, MetaData, Numeric, String, Table, create_engine
from sqlalchemy.dialects.postgresql import insert


DEFAULT_TABLE_NAME = "studio_daily_metrics"
DATE_COLUMNS = ["week", "date"]
INT_COLUMNS = [
    "id",
    "mt_visits",
    "cp_visits",
    "total_visits",
    "first_time",
    "capacity",
    "classes",
    "slots",
]
FLOAT_COLUMNS = [
    "est_visits",
    "occ_pct",
    "mt_sales",
    "cp_sales",
    "net_sales",
    "est_sales",
]
TEXT_COLUMNS = ["studio", "month", "day"]


def normalize_columns(columns: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for col in columns:
        name = col.strip().lower().replace("%", "pct")
        name = re.sub(r"[^0-9a-z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        if name == "":
            name = "column"
        normalized.append(name)
    return normalized


def parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        cleaned = re.sub(r"[^0-9.\-]", "", stripped)
        if cleaned in {"", "-", "."}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return None


def clean_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = normalize_columns(df.columns)

    column_map = {
        "id": "id",
        "studio": "studio",
        "week": "week",
        "month": "month",
        "day": "day",
        "date": "date",
        "mtvisits": "mt_visits",
        "cpvisits": "cp_visits",
        "totalvisits": "total_visits",
        "estvisits": "est_visits",
        "firsttime": "first_time",
        "capacity": "capacity",
        "classes": "classes",
        "slots": "slots",
        "occpct": "occ_pct",
        "mtsales": "mt_sales",
        "cpsales": "cp_sales",
        "netsales": "net_sales",
        "estsales": "est_sales",
    }

    df = df.rename(columns=column_map)

    missing = set(column_map.values()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="%m/%d/%y")
        df[col] = df[col].dt.date

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": None})

    for col in FLOAT_COLUMNS + INT_COLUMNS:
        df[col] = df[col].apply(parse_numeric)

    for col in INT_COLUMNS:
        df[col] = df[col].apply(lambda x: int(x) if x is not None else None)

    cleaned = df[[
        "id",
        "studio",
        "week",
        "month",
        "day",
        "date",
        "mt_visits",
        "cp_visits",
        "total_visits",
        "est_visits",
        "first_time",
        "capacity",
        "classes",
        "slots",
        "occ_pct",
        "mt_sales",
        "cp_sales",
        "net_sales",
        "est_sales",
    ]].dropna(subset=["id", "studio", "date"])

    return cleaned


def build_table(metadata: MetaData, table_name: str) -> Table:
    return Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True),
        Column("studio", String, nullable=False),
        Column("week", Date, nullable=True),
        Column("month", String, nullable=True),
        Column("day", String, nullable=True),
        Column("date", Date, nullable=False),
        Column("mt_visits", Integer),
        Column("cp_visits", Integer),
        Column("total_visits", Integer),
        Column("est_visits", Numeric(18, 2)),
        Column("first_time", Integer),
        Column("capacity", Integer),
        Column("classes", Integer),
        Column("slots", Integer),
        Column("occ_pct", Numeric(8, 4)),
        Column("mt_sales", Numeric(18, 2)),
        Column("cp_sales", Numeric(18, 2)),
        Column("net_sales", Numeric(18, 2)),
        Column("est_sales", Numeric(18, 2)),
        schema="public",
    )


def upsert_rows(table: Table, records: List[Dict[str, Any]], engine_url: str) -> int:
    engine = create_engine(engine_url, connect_args={"sslmode": "require"})
    metadata = table.metadata
    metadata.bind = engine

    with engine.begin() as conn:
        metadata.create_all(conn, tables=[table])
        if not records:
            return 0

        insert_stmt = insert(table).values(records)
        update_cols = {
            col.name: getattr(insert_stmt.excluded, col.name)
            for col in table.columns
            if col.name != "id"
        }
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[table.c.id],
            set_=update_cols,
        )
        conn.execute(upsert_stmt)

    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load cleaned studio metrics CSV into Supabase")
    parser.add_argument(
        "--csv",
        default="data/JFWTest.csv",
        help="Path to the CSV file (default: data/JFWTest.csv)",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_NAME,
        help=f"Destination table name (default: {DEFAULT_TABLE_NAME})",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("SUPABASE_DB_URL"),
        help="Connection string. Defaults to SUPABASE_DB_URL env var.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip database writes and just report cleaned row count",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    cleaned_df = clean_frame(df)

    metadata = MetaData()
    table = build_table(metadata, args.table)

    if args.dry_run:
        print(f"Prepared {len(cleaned_df)} rows for table {table.fullname}")
        return

    if not args.database_url:
        raise SystemExit("Set --database-url or SUPABASE_DB_URL before running")

    row_count = upsert_rows(table, cleaned_df.to_dict("records"), args.database_url)
    print(f"Loaded {row_count} rows into {table.fullname}")


if __name__ == "__main__":
    main()
