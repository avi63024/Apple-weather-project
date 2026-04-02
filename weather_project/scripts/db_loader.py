from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from csv_cleaner import clean_csv
from pdf_parser import parse_pdfs

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "weather.db"


def create_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_table(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS weather_data")
    connection.execute(
        """
        CREATE TABLE weather_data (
            id INTEGER PRIMARY KEY,
            date TEXT,
            max_temp REAL,
            min_temp REAL,
            avg_temp REAL,
            precipitation REAL,
            departure REAL,
            normal_max_temp REAL,
            normal_min_temp REAL,
            normal_avg_temp REAL,
            record_high_temp REAL,
            record_low_temp REAL,
            record_high_year INTEGER,
            record_low_year INTEGER,
            qc_flag INTEGER,
            source_file TEXT,
            source TEXT,
            UNIQUE(date, source)
        )
        """
    )
    connection.commit()


def load_dataframe(connection: sqlite3.Connection, dataframe: pd.DataFrame) -> int:
    if dataframe.empty:
        return 0

    columns = [
        "date",
        "max_temp",
        "min_temp",
        "avg_temp",
        "precipitation",
        "departure",
        "normal_max_temp",
        "normal_min_temp",
        "normal_avg_temp",
        "record_high_temp",
        "record_low_temp",
        "record_high_year",
        "record_low_year",
        "qc_flag",
        "source_file",
        "source",
    ]

    def normalize_sql_value(value: object) -> object:
        if pd.isna(value):
            return None
        return value.item() if hasattr(value, "item") else value

    rows = (
        tuple(normalize_sql_value(value) for value in row)
        for row in dataframe[columns].itertuples(index=False, name=None)
    )
    before = connection.total_changes
    connection.executemany(
        """
        INSERT OR IGNORE INTO weather_data (
            date, max_temp, min_temp, avg_temp, precipitation, departure,
            normal_max_temp, normal_min_temp, normal_avg_temp,
            record_high_temp, record_low_temp, record_high_year, record_low_year,
            qc_flag, source_file, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()
    return connection.total_changes - before


def load_weather_data() -> None:
    pdf_df = parse_pdfs()
    csv_df = clean_csv()

    with create_connection() as connection:
        create_table(connection)
        inserted_pdf = load_dataframe(connection, pdf_df)
        inserted_csv = load_dataframe(connection, csv_df)
        total_rows = connection.execute("SELECT COUNT(*) FROM weather_data").fetchone()[0]

    print(f"Inserted {inserted_pdf} PDF rows.")
    print(f"Inserted {inserted_csv} CSV rows.")
    print(f"Database ready at: {DB_PATH}")
    print(f"Total rows in weather_data: {total_rows}")


if __name__ == "__main__":
    load_weather_data()
