from __future__ import annotations

import re
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
PDF_DIR = DATA_DIR / "pdfs"


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def normalize_numeric(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    cleaned = value.strip().replace(",", "")
    cleaned = re.sub(r"[^\d.\-M]", "", cleaned)

    if not cleaned or cleaned.upper() == "M":
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_first_number_after_label(text: str, labels: list[str]) -> Optional[float]:
    for label in labels:
        pattern = rf"{label}\s*(?:\([^)]+\))?\s*[:\-]?\s*(?:Daily Obs\s*)?(?:Obs\s*)?(M|[-+]?\d+(?:\.\d+)?)\s*(?:°?F|in)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return normalize_numeric(match.group(1))
    return None


def extract_stacked_metric(text: str, labels: list[str]) -> Optional[float]:
    for label in labels:
        pattern = rf"{label}\s*(?:\([^)]+\))?\s*(?:Observed|Obs|Value|Daily Obs)?\s*(M|[-+]?\d+(?:\.\d+)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return normalize_numeric(match.group(1))
    return None


def daily_section(text: str) -> str:
    start_markers = ["WEATHER DATA", "WEATHER REPORT", "MARCH "]
    end_markers = ["MONTH-TO-DATE", "Month and Year-to-Date Data", "MTD", "YTD", "RECORD EXTREMES"]

    start = 0
    for marker in start_markers:
        index = text.find(marker)
        if index != -1:
            start = index
            break

    section = text[start:]
    end_positions = [section.find(marker) for marker in end_markers if section.find(marker) != -1]
    if end_positions:
        section = section[: min(end_positions)]
    return section


def extract_precipitation(text: str) -> Optional[float]:
    section = daily_section(text)
    if re.search(r"No precipitation was recorded", section, re.IGNORECASE):
        return 0.0
    for pattern in [
        r"Precipitation\s*(?:\([^)]+\))?\s*[:\-]?\s*(M|[-+]?\d+(?:\.\d+)?)\s*(?:in|inches)?",
        r"Precip\s*(?:\([^)]+\))?\s*[:\-]?\s*(M|[-+]?\d+(?:\.\d+)?)\s*(?:in|inches)?",
        r"No precipitation was recorded\s*\(?([0-9.]+)?",
    ]:
        match = re.search(pattern, section, re.IGNORECASE)
        if match:
            value = match.group(1) if match.groups() else "0"
            return normalize_numeric(value or "0")
    return None


def extract_qc_flag(text: str) -> Optional[int]:
    match = re.search(r"QC FLAG:\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_metric_table_values(text: str, labels: list[str]) -> dict[str, Optional[float]]:
    for label in labels:
        pattern = (
            rf"{label}\s*(?:\([^)]+\))?\s*"
            rf"(M|[-+]?\d+(?:\.\d+)?)\s*(?:°?F|in)?\s*"
            rf"(M|[-+]?\d+(?:\.\d+)?)?\s*(?:°?F|in)?\s*"
            rf"(?:\s*(M|[-+]?\d+(?:\.\d+)?)\s*(?:°?F|in)?\s*\((\d{{4}})\))?"
            rf"(?:\s*(M|[-+]?\d+(?:\.\d+)?)\s*(?:°?F|in)?\s*\((\d{{4}})\))?"
        )
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                "observed": normalize_numeric(match.group(1)),
                "normal": normalize_numeric(match.group(2)),
                "record_high": normalize_numeric(match.group(3)),
                "record_high_year": normalize_numeric(match.group(4)),
                "record_low": normalize_numeric(match.group(5)),
                "record_low_year": normalize_numeric(match.group(6)),
            }
    return {
        "observed": None,
        "normal": None,
        "record_high": None,
        "record_high_year": None,
        "record_low": None,
        "record_low_year": None,
    }


def extract_narrative_normal_high_low(text: str) -> tuple[Optional[float], Optional[float]]:
    patterns = [
        r"normal expectations .*?high of (\d+(?:\.\d+)?)°?F and low of (\d+(?:\.\d+)?)°?F",
        r"Normal conditions .*?high of (\d+(?:\.\d+)?)°?F and low of (\d+(?:\.\d+)?)°?F",
        r"Normal Values .*?Max (\d+(?:\.\d+)?)°?F,\s*Min (\d+(?:\.\d+)?)°?F",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_numeric(match.group(1)), normalize_numeric(match.group(2))
    return None, None


def extract_narrative_normal_avg(text: str) -> Optional[float]:
    patterns = [
        r"normal average of (\d+(?:\.\d+)?)°?F",
        r"Normal Values .*?Avg (\d+(?:\.\d+)?)°?F",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_numeric(match.group(1))
    return None


def extract_named_record_values(text: str) -> tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    high_patterns = [
        r"Record High\s*(\d+(?:\.\d+)?)\s*°?F?\s*\((\d{4})\)",
        r"record high of\s*(\d+(?:\.\d+)?)°?F.*?(?:set in|from)\s*(\d{4})",
    ]
    low_patterns = [
        r"Record Low\s*(\d+(?:\.\d+)?)\s*°?F?\s*\((\d{4})\)",
        r"record low of\s*(\d+(?:\.\d+)?)°?F.*?(?:set in|from)\s*(\d{4})",
    ]

    record_high = record_high_year = record_low = record_low_year = None
    for pattern in high_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            record_high = normalize_numeric(match.group(1))
            record_high_year = int(match.group(2))
            break
    for pattern in low_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            record_low = normalize_numeric(match.group(1))
            record_low_year = int(match.group(2))
            break

    return record_high, record_low, record_high_year, record_low_year


def extract_narrative_record_high_low(text: str) -> tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    patterns = [
        r"historical highs of (\d+(?:\.\d+)?)°?F \((\d{4})\) and lows of (\d+(?:\.\d+)?)°?F \((\d{4})\)",
        r"Record data for this date:\s*High of (\d+(?:\.\d+)?)°?F \((\d{4})\),\s*Low of (\d+(?:\.\d+)?)°?F \((\d{4})\)",
        r"RECORD EXTREMES FOR MARCH \d+.*?Max Temp\s*(\d+(?:\.\d+)?)°?F \((\d{4})\).*?Min Temp\s*(\d+(?:\.\d+)?)°?F \((\d{4})\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return (
                normalize_numeric(match.group(1)),
                normalize_numeric(match.group(3)),
                int(match.group(2)),
                int(match.group(4)),
            )
    return None, None, None, None


def extract_daily_metrics(text: str) -> dict[str, Optional[float]]:
    section = daily_section(text)
    max_values = extract_metric_table_values(section, ["Max Temperature", "Max Temp", "Max"])
    min_values = extract_metric_table_values(section, ["Min Temperature", "Min Temp", "Min"])
    avg_values = extract_metric_table_values(section, ["Avg Temperature", "Avg Temp", "Avg"])

    normal_high_fallback, normal_low_fallback = extract_narrative_normal_high_low(text)
    normal_avg_fallback = extract_narrative_normal_avg(text)
    record_high_fallback, record_low_fallback, record_high_year_fallback, record_low_year_fallback = extract_narrative_record_high_low(text)
    named_record_high, named_record_low, named_record_high_year, named_record_low_year = extract_named_record_values(text)

    return {
        "max_temp": max_values["observed"] or extract_first_number_after_label(section, ["Max Temperature", "Max Temp", "Max"]) or extract_stacked_metric(section, ["Max Temperature", "Max Temp", "Max"]),
        "min_temp": min_values["observed"] or extract_first_number_after_label(section, ["Min Temperature", "Min Temp", "Min"]) or extract_stacked_metric(section, ["Min Temperature", "Min Temp", "Min"]),
        "avg_temp": avg_values["observed"] or extract_first_number_after_label(section, ["Avg Temperature", "Avg Temp", "Avg"]) or extract_stacked_metric(section, ["Avg Temperature", "Avg Temp", "Avg"]),
        "normal_max_temp": max_values["normal"] or normal_high_fallback,
        "normal_min_temp": min_values["normal"] or normal_low_fallback,
        "normal_avg_temp": avg_values["normal"] or normal_avg_fallback,
        "record_high_temp": max_values["record_high"] or named_record_high or record_high_fallback,
        "record_low_temp": min_values["record_low"] or named_record_low or record_low_fallback,
        "record_high_year": max_values["record_high_year"] or named_record_high_year or record_high_year_fallback,
        "record_low_year": min_values["record_low_year"] or named_record_low_year or record_low_year_fallback,
    }


def extract_date(text: str) -> Optional[str]:
    patterns = [
        r"REF:\s+\w+-(\d{4}-\d{2}-\d{2})",
        r"WEATHER DATA\s+[—-]\s+(\d{1,2}/\d{1,2}/\d{2})",
        r"WEATHER DATA\s+[—-]\s+(\d{1,2}-[A-Z]{3}-\d{4})",
        r"WEATHER REPORT\s+MARCH\s+(\d{1,2}),\s+(\d{4})",
        r"MARCH\s+(\d{1,2}),\s+(\d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        groups = match.groups()
        if len(groups) == 1:
            raw = groups[0]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
                return raw
            if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", raw):
                date = pd.to_datetime(raw, format="%m/%d/%y", errors="coerce")
                return None if pd.isna(date) else date.strftime("%Y-%m-%d")
            if re.fullmatch(r"\d{1,2}-[A-Z]{3}-\d{4}", raw):
                date = pd.to_datetime(raw, format="%d-%b-%Y", errors="coerce")
                return None if pd.isna(date) else date.strftime("%Y-%m-%d")
        if len(groups) == 2:
            date = pd.to_datetime(f"March {groups[0]} {groups[1]}", errors="coerce")
            return None if pd.isna(date) else date.strftime("%Y-%m-%d")

    return None


def parse_pdf(pdf_path: Path) -> Optional[dict]:
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as exc:
        print(f"Skipping broken PDF {pdf_path.name}: {exc}")
        return None

    if not text.strip():
        print(f"Skipping empty PDF {pdf_path.name}")
        return None

    metrics = extract_daily_metrics(text)
    data = {
        "date": extract_date(text),
        "max_temp": metrics["max_temp"],
        "min_temp": metrics["min_temp"],
        "avg_temp": metrics["avg_temp"],
        "precipitation": extract_precipitation(text),
        "departure": None,
        "normal_max_temp": metrics["normal_max_temp"],
        "normal_min_temp": metrics["normal_min_temp"],
        "normal_avg_temp": metrics["normal_avg_temp"],
        "record_high_temp": metrics["record_high_temp"],
        "record_low_temp": metrics["record_low_temp"],
        "record_high_year": metrics["record_high_year"],
        "record_low_year": metrics["record_low_year"],
        "qc_flag": extract_qc_flag(text),
        "source_file": pdf_path.name,
        "source": "pdf",
    }

    if not data["date"]:
        print(f"Skipping PDF with missing date {pdf_path.name}")
        return None

    return data


def parse_pdfs() -> pd.DataFrame:
    pdf_files = sorted(Path(path) for path in glob(str(PDF_DIR / "*.pdf")))
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
    if not PDF_DIR.exists():
        print(f"PDF directory not found: {PDF_DIR}")
        return pd.DataFrame(columns=columns)
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return pd.DataFrame(columns=columns)

    rows = []
    for pdf_file in pdf_files:
        parsed = parse_pdf(pdf_file)
        if parsed:
            rows.append(parsed)

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for column in [
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
        ]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.sort_values("date").drop_duplicates(subset=["date", "source"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    pdf_df = parse_pdfs()
    print(pdf_df.to_string(index=False) if not pdf_df.empty else "No PDF records parsed.")
