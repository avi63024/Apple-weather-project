from __future__ import annotations

import re
import sys

import pandas as pd

from analysis import build_analysis_views, load_weather_data, identify_record_events


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def mentions_recent_window(text: str) -> bool:
    return has_any(
        text,
        [
            "last 10",
            "last ten",
            "recent 10",
            "recent ten",
            "recent days",
            "last few days",
            "past 10 days",
            "past ten days",
        ],
    )


def is_hottest_question(text: str) -> bool:
    return has_any(text, ["hottest", "warmest", "highest temperature", "highest temp", "max temperature", "max temp"]) and (
        mentions_recent_window(text) or "day" in text or "date" in text
    )


def is_biggest_swing_question(text: str) -> bool:
    return (
        has_any(text, ["biggest swing", "largest swing", "temperature swing", "high-low swing", "difference between the high and low"])
        or (has_any(text, ["swing", "range", "difference", "spread"]) and has_any(text, ["high", "low"]))
        or ("diurnal" in text and "range" in text)
    )


def is_warmer_than_average_question(text: str) -> bool:
    return has_any(
        text,
        [
            "warmer than average",
            "days above average",
            "above average",
            "above normal",
            "warmer than normal",
            "warmer than expected",
            "how many warm days",
        ],
    )


def is_record_question(text: str) -> bool:
    return has_any(
        text,
        [
            "record",
            "record high",
            "record low",
            "all-time high",
            "all-time low",
            "close to breaking",
            "close to a record",
            "near a record",
            "broke any records",
            "set any records",
        ],
    )


def is_trend_question(text: str) -> bool:
    return has_any(
        text,
        [
            "trend",
            "trends",
            "past 3 months",
            "past three months",
            "three months",
            "3 months",
            "long term",
            "history",
            "historical trend",
        ],
    )


def supported_topics_text() -> str:
    return (
        "I can answer questions grounded in the weather database about specific dates, temperatures, precipitation, "
        "the hottest recent day, largest high-low swing, warmer-than-average days, record highs/lows, coldest day, and 3-month trends. "
        "Try asking in natural language, for example: 'What was the weather on 2026-01-04?' or 'Summarize the 3-month trend.'"
    )


def extract_date_from_query(query: str) -> pd.Timestamp | None:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2}(?:,\s*\d{4})?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            continue
        raw = match.group(0)
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.notna(parsed):
            if re.search(r"\d{4}", raw) is None:
                parsed = parsed.replace(year=2026)
            return parsed.normalize()
    return None


def find_day_row(merged_history: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series | None:
    day_rows = merged_history[merged_history["date"] == target_date].copy()
    if day_rows.empty:
        return None
    return day_rows.sort_values("date").iloc[0]


def format_number(value: object, suffix: str = "") -> str:
    if pd.isna(value):
        return "not available"
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}{suffix}"
    return f"{value}{suffix}"


def answer_specific_date_question(query: str, merged_history: pd.DataFrame) -> str | None:
    target_date = extract_date_from_query(query)
    if target_date is None:
        return None

    row = find_day_row(merged_history, target_date)
    date_text = target_date.strftime("%Y-%m-%d")
    if row is None:
        return f"I could not find a weather record for {date_text} in the database."

    high = format_number(row.get("max_temp"), "F")
    low = format_number(row.get("min_temp"), "F")
    avg = format_number(row.get("avg_temp"), "F")
    precipitation = format_number(row.get("precipitation"), " in")
    normal_avg = format_number(row.get("normal_avg_temp"), "F")
    swing = format_number(row.get("temp_swing"), "F")

    source_parts: list[str] = []
    if pd.notna(row.get("pdf_source_file")):
        source_parts.append(f"PDF ({row['pdf_source_file']})")
    if pd.notna(row.get("source_file")):
        source_parts.append(f"CSV ({row['source_file']})")
    source_text = ", ".join(source_parts) if source_parts else "database history"

    response = (
        f"For {date_text}, the high was {high}, the low was {low}, the average temperature was {avg}, "
        f"and precipitation was {precipitation}. The temperature swing was {swing}. "
        f"Normal average temperature was {normal_avg}. Source used: {source_text}."
    )

    if has_any(query, ["record", "normal high", "normal low", "record high", "record low"]):
        extras: list[str] = []
        if pd.notna(row.get("normal_max_temp")):
            extras.append(f"normal high {float(row['normal_max_temp']):.1f}F")
        if pd.notna(row.get("normal_min_temp")):
            extras.append(f"normal low {float(row['normal_min_temp']):.1f}F")
        if pd.notna(row.get("record_high_temp")):
            extras.append(f"record high {float(row['record_high_temp']):.1f}F")
        if pd.notna(row.get("record_low_temp")):
            extras.append(f"record low {float(row['record_low_temp']):.1f}F")
        if extras:
            response += " Additional context: " + ", ".join(extras) + "."

    return response


def run_query(query: str) -> str:
    normalized = normalize_query(query)
    weather_df = load_weather_data()
    last_10_days, merged_history = build_analysis_views(weather_df)

    specific_date_answer = answer_specific_date_question(normalized, merged_history)
    if specific_date_answer:
        return specific_date_answer

    if is_hottest_question(normalized):
        hottest_day = last_10_days.dropna(subset=["max_temp"]).sort_values(["max_temp", "date"], ascending=[False, True]).iloc[0]
        return (
            f"The hottest day in the last 10 PDF reports was {hottest_day['date'].strftime('%Y-%m-%d')}. "
            f"The observed high was {hottest_day['max_temp']:.1f}F, which was {hottest_day['high_vs_normal']:.1f}F above the normal high."
        )

    if is_biggest_swing_question(normalized):
        biggest_swing = last_10_days.dropna(subset=["temp_swing"]).sort_values(["temp_swing", "date"], ascending=[False, True]).iloc[0]
        return (
            f"The biggest swing was on {biggest_swing['date'].strftime('%Y-%m-%d')}: "
            f"{biggest_swing['max_temp']:.1f}F high, {biggest_swing['min_temp']:.1f}F low, "
            f"for a {biggest_swing['temp_swing']:.1f}F swing."
        )

    if is_warmer_than_average_question(normalized):
        warmer_than_average = last_10_days[
            last_10_days["avg_temp"].notna()
            & last_10_days["effective_normal_avg_temp"].notna()
            & (last_10_days["avg_temp"] > last_10_days["effective_normal_avg_temp"])
        ]
        comparable_days = last_10_days[
            last_10_days["avg_temp"].notna() & last_10_days["effective_normal_avg_temp"].notna()
        ]
        return f"{len(warmer_than_average)} of {len(comparable_days)} comparable PDF days were warmer than the daily normal average."

    if is_record_question(normalized):
        findings = identify_record_events(last_10_days)
        if not findings:
            return "None of the last 10 PDF days were within 2F of an all-time record high or low."
        return "\n".join(findings)

    if is_trend_question(normalized):
        trend = merged_history.dropna(subset=["date", "avg_temp"]).copy()
        trend["month"] = trend["date"].dt.to_period("M").astype(str)
        monthly = trend.groupby("month", as_index=False).agg(
            avg_temp=("avg_temp", "mean"),
            max_temp=("max_temp", "max"),
            precipitation=("precipitation", "sum"),
        )
        lines = ["Past 3-month trend summary:"]
        for row in monthly.itertuples(index=False):
            lines.append(
                f"- {row.month}: avg temp {row.avg_temp:.1f}F, warmest high {row.max_temp:.1f}F, precipitation {row.precipitation:.2f} in"
            )
        return "\n".join(lines)

    if "coldest" in normalized:
        coldest = merged_history.dropna(subset=["min_temp"]).sort_values(["min_temp", "date"], ascending=[True, True]).iloc[0]
        return f"The coldest day in the 3-month history was {coldest['date'].strftime('%Y-%m-%d')} with a low of {coldest['min_temp']:.1f}F."

    if has_any(normalized, ["summary", "summarize", "overview", "key findings", "main findings"]):
        findings = identify_record_events(last_10_days)
        warmer_than_average = last_10_days[
            last_10_days["avg_temp"].notna()
            & last_10_days["effective_normal_avg_temp"].notna()
            & (last_10_days["avg_temp"] > last_10_days["effective_normal_avg_temp"])
        ]
        hottest_day = last_10_days.dropna(subset=["max_temp"]).sort_values(["max_temp", "date"], ascending=[False, True]).iloc[0]
        biggest_swing = last_10_days.dropna(subset=["temp_swing"]).sort_values(["temp_swing", "date"], ascending=[False, True]).iloc[0]
        return (
            f"Summary: hottest recent day was {hottest_day['date'].strftime('%Y-%m-%d')} at {hottest_day['max_temp']:.1f}F; "
            f"largest swing was {biggest_swing['temp_swing']:.1f}F on {biggest_swing['date'].strftime('%Y-%m-%d')}; "
            f"{len(warmer_than_average)} recent days were warmer than average; "
            f"record-related findings: {len(findings)}."
        )

    return supported_topics_text()


def demo_questions() -> list[str]:
    return [
        "What was the hottest day in the last 10 days?",
        "Which day had the biggest swing between high and low temperature in the last 10 days?",
        "How many of the last 10 days were warmer than average?",
        "Did any of the last 10 days set or come close to breaking a record?",
        "What are the trends in the past 3 months?",
    ]


def print_demo() -> None:
    for question in demo_questions():
        print(f"Q: {question}")
        print(f"A: {run_query(question)}")
        print()


def chat() -> None:
    print("Weather chatbot ready. Type 'exit' to quit, or '--demo' to print five grounded example Q&A pairs.")
    while True:
        user_input = input("Ask a question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        try:
            print(run_query(user_input))
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print_demo()
    else:
        chat()
