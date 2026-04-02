from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "raw" / "weather.csv"
OUTPUT_COLUMNS = [
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


def normalize_value(value: object) -> object:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "M", "—", "-", "ERROR"}:
        return None
    return text


def convert_block(block: list[list[str]], width: int) -> pd.DataFrame:
    rows = []
    for row in block:
        segment = row[:width]
        if not segment or not str(segment[0]).strip():
            continue
        rows.append(segment)

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    columns = ["date", "max_temp", "min_temp", "avg_temp", "departure", "hdd", "cdd", "precipitation"]
    if width == 9:
        columns.append("snow_depth")
    frame = pd.DataFrame(rows, columns=columns)
    frame["raw_date"] = frame["date"]

    for column in frame.columns:
        if column != "raw_date":
            frame[column] = frame[column].map(normalize_value)

    frame = frame[~frame["raw_date"].astype(str).str.contains(r"^(?:Sum|Average|Normal|Above)", case=False, regex=True)]

    frame = frame[frame["date"].notna() | frame["date"].astype(str).str.strip().ne("")]
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%y", errors="coerce")
    frame = frame[frame["date"].notna()]

    for column in ["max_temp", "min_temp", "avg_temp", "precipitation"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["departure"] = pd.to_numeric(frame["departure"], errors="coerce")
    frame["normal_max_temp"] = pd.NA
    frame["normal_min_temp"] = pd.NA
    frame["normal_avg_temp"] = pd.NA
    frame["record_high_temp"] = pd.NA
    frame["record_low_temp"] = pd.NA
    frame["record_high_year"] = pd.NA
    frame["record_low_year"] = pd.NA
    frame["qc_flag"] = pd.NA
    frame["source_file"] = "weather.csv"
    frame["source"] = "csv"
    return frame[OUTPUT_COLUMNS]


def clean_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"CSV file not found: {CSV_PATH}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw = pd.read_csv(CSV_PATH, header=None).fillna("")
    rows = raw.values.tolist()
    frames: list[pd.DataFrame] = []
    i = 0

    while i < len(rows):
        row = rows[i]
        if "Date" in row[:8]:
            block = []
            j = i + 1
            while j < len(rows) and not ("Date" in rows[j][:8] or str(rows[j][0]).endswith("-Mar")):
                block.append(rows[j])
                j += 1

            left_rows = [r[:8] for r in block if any(str(cell).strip() for cell in r[:8])]
            if left_rows:
                width = 9 if any(len(r) > 8 and str(r[8]).strip() for r in block) else 8
                if width == 9:
                    left_rows = [r[:9] for r in block if any(str(cell).strip() for cell in r[:9])]
                frames.append(convert_block(left_rows, width))

            right_rows = [r[8:16] for r in block if len(r) >= 16 and any(str(cell).strip() for cell in r[8:16])]
            if right_rows:
                frames.append(convert_block(right_rows, 8))

            i = j
            continue

        i += 1

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    cleaned = pd.concat(frames, ignore_index=True)
    cleaned = cleaned.drop_duplicates(subset=["date", "source"]).sort_values("date").reset_index(drop=True)
    return cleaned


if __name__ == "__main__":
    csv_df = clean_csv()
    print(csv_df.to_string(index=False) if not csv_df.empty else "No CSV records cleaned.")
