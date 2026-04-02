from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "weather.db"
NOTEBOOK_DIR = BASE_DIR / "notebook"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "weather_project_mplconfig"))

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_weather_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Run db_loader.py first.")

    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query("SELECT * FROM weather_data ORDER BY date, source", connection)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_analysis_views(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pdf = df[df["source"] == "pdf"].copy()
    csv = df[df["source"] == "csv"].copy()

    csv["normal_avg_temp_from_departure"] = csv["avg_temp"] - csv["departure"]

    pdf_last_10 = pdf.sort_values("date").tail(10).copy()
    pdf_last_10 = pdf_last_10.merge(
        csv[["date", "normal_avg_temp_from_departure", "avg_temp", "departure"]].rename(
            columns={
                "avg_temp": "csv_avg_temp",
                "departure": "csv_departure",
            }
        ),
        on="date",
        how="left",
    )
    pdf_last_10["effective_normal_avg_temp"] = pdf_last_10["normal_avg_temp"].combine_first(pdf_last_10["normal_avg_temp_from_departure"])
    pdf_last_10["high_vs_normal"] = pdf_last_10["max_temp"] - pdf_last_10["normal_max_temp"]
    pdf_last_10["avg_vs_normal"] = pdf_last_10["avg_temp"] - pdf_last_10["effective_normal_avg_temp"]
    pdf_last_10["temp_swing"] = pdf_last_10["max_temp"] - pdf_last_10["min_temp"]

    merged_history = csv.copy()
    pdf_for_merge = pdf[
        [
            "date",
            "max_temp",
            "min_temp",
            "avg_temp",
            "precipitation",
            "normal_max_temp",
            "normal_min_temp",
            "normal_avg_temp",
            "record_high_temp",
            "record_low_temp",
            "record_high_year",
            "record_low_year",
            "qc_flag",
            "source_file",
        ]
    ].rename(
        columns={
            "max_temp": "pdf_max_temp",
            "min_temp": "pdf_min_temp",
            "avg_temp": "pdf_avg_temp",
            "precipitation": "pdf_precipitation",
            "normal_max_temp": "pdf_normal_max_temp",
            "normal_min_temp": "pdf_normal_min_temp",
            "normal_avg_temp": "pdf_normal_avg_temp",
            "record_high_temp": "pdf_record_high_temp",
            "record_low_temp": "pdf_record_low_temp",
            "record_high_year": "pdf_record_high_year",
            "record_low_year": "pdf_record_low_year",
            "qc_flag": "pdf_qc_flag",
            "source_file": "pdf_source_file",
        }
    )

    merged_history = merged_history.merge(pdf_for_merge, on="date", how="outer")
    for metric in ["max_temp", "min_temp", "avg_temp", "precipitation"]:
        merged_history[metric] = merged_history[f"pdf_{metric}"].combine_first(merged_history[metric])

    merged_history["normal_avg_temp"] = merged_history["pdf_normal_avg_temp"].combine_first(
        merged_history["avg_temp"] - merged_history["departure"]
    )
    merged_history["normal_max_temp"] = merged_history["pdf_normal_max_temp"].combine_first(merged_history["normal_max_temp"])
    merged_history["normal_min_temp"] = merged_history["pdf_normal_min_temp"].combine_first(merged_history["normal_min_temp"])
    merged_history["record_high_temp"] = merged_history["pdf_record_high_temp"].combine_first(merged_history["record_high_temp"])
    merged_history["record_low_temp"] = merged_history["pdf_record_low_temp"].combine_first(merged_history["record_low_temp"])
    merged_history["record_high_year"] = merged_history["pdf_record_high_year"].combine_first(merged_history["record_high_year"])
    merged_history["record_low_year"] = merged_history["pdf_record_low_year"].combine_first(merged_history["record_low_year"])
    merged_history["temp_swing"] = merged_history["max_temp"] - merged_history["min_temp"]
    merged_history = merged_history.sort_values("date").reset_index(drop=True)

    return pdf_last_10, merged_history


def identify_record_events(last_10_days: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    for row in last_10_days.itertuples(index=False):
        date_text = row.date.strftime("%Y-%m-%d")

        if pd.notna(row.max_temp) and pd.notna(row.record_high_temp):
            gap = row.record_high_temp - row.max_temp
            if gap <= 0:
                findings.append(
                    f"{date_text}: max temp {row.max_temp:.1f}F tied or broke the record high of {row.record_high_temp:.1f}F ({int(row.record_high_year) if pd.notna(row.record_high_year) else 'year unknown'})."
                )
            elif gap <= 2:
                findings.append(
                    f"{date_text}: max temp {row.max_temp:.1f}F came within {gap:.1f}F of the record high {row.record_high_temp:.1f}F."
                )

        if pd.notna(row.min_temp) and pd.notna(row.record_low_temp):
            gap = row.min_temp - row.record_low_temp
            if gap <= 0:
                findings.append(
                    f"{date_text}: min temp {row.min_temp:.1f}F tied or broke the record low of {row.record_low_temp:.1f}F ({int(row.record_low_year) if pd.notna(row.record_low_year) else 'year unknown'})."
                )
            elif gap <= 2:
                findings.append(
                    f"{date_text}: min temp {row.min_temp:.1f}F came within {gap:.1f}F of the record low {row.record_low_temp:.1f}F."
                )

    return findings


def save_chart(fig: plt.Figure, filename: str) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NOTEBOOK_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def create_charts(last_10_days: pd.DataFrame, merged_history: pd.DataFrame) -> None:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    chart_data = last_10_days.dropna(subset=["date"]).copy()
    ax1.plot(chart_data["date"], chart_data["max_temp"], marker="o", label="Observed Max (PDF)")
    ax1.plot(chart_data["date"], chart_data["normal_max_temp"], marker="o", linestyle="--", label="Normal Max (PDF)")
    ax1.set_title("Last 10 Days: Observed vs Normal High Temperature")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (F)")
    ax1.legend()
    save_chart(fig1, "last_10_days_observed_vs_normal_high.png")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(chart_data["date"], chart_data["temp_swing"])
    ax2.set_title("Last 10 Days: Daily Temperature Swing")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("High-Low Swing (F)")
    save_chart(fig2, "last_10_days_temperature_swing.png")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    trend = merged_history.dropna(subset=["date", "avg_temp"]).copy()
    trend["rolling_7d_avg"] = trend["avg_temp"].rolling(window=7, min_periods=1).mean()
    ax3.plot(trend["date"], trend["avg_temp"], alpha=0.35, label="Daily Avg Temp")
    ax3.plot(trend["date"], trend["rolling_7d_avg"], linewidth=2.5, label="7-Day Rolling Avg")
    ax3.set_title("Past 3 Months: Average Temperature Trend")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Temperature (F)")
    ax3.legend()
    save_chart(fig3, "three_month_temperature_trend.png")

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    monthly = trend.assign(month=trend["date"].dt.to_period("M").dt.to_timestamp())
    monthly = monthly.groupby("month", as_index=False)["precipitation"].sum()
    ax4.bar(monthly["month"].dt.strftime("%Y-%m"), monthly["precipitation"].fillna(0))
    ax4.set_title("Past 3 Months: Monthly Precipitation Totals")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Precipitation (in)")
    save_chart(fig4, "three_month_monthly_precipitation.png")


def summarize_results(last_10_days: pd.DataFrame, merged_history: pd.DataFrame) -> list[str]:
    hottest_day = last_10_days.dropna(subset=["max_temp"]).sort_values(["max_temp", "date"], ascending=[False, True]).iloc[0]
    biggest_swing = last_10_days.dropna(subset=["temp_swing"]).sort_values(["temp_swing", "date"], ascending=[False, True]).iloc[0]

    warmer_than_average = last_10_days[
        last_10_days["avg_temp"].notna()
        & last_10_days["effective_normal_avg_temp"].notna()
        & (last_10_days["avg_temp"] > last_10_days["effective_normal_avg_temp"])
    ]
    comparable_days = last_10_days[
        last_10_days["avg_temp"].notna() & last_10_days["effective_normal_avg_temp"].notna()
    ]

    trend = merged_history.dropna(subset=["date", "avg_temp"]).copy()
    monthly_summary = (
        trend.assign(month=trend["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(avg_temp=("avg_temp", "mean"), max_temp=("max_temp", "max"), precipitation=("precipitation", "sum"))
    )

    lines = [
        "Assignment Questions",
        f"1. Hottest day in the last 10 PDF days: {hottest_day['date'].strftime('%Y-%m-%d')} at {hottest_day['max_temp']:.1f}F, which was {hottest_day['high_vs_normal']:.1f}F above the daily normal high.",
        f"2. Biggest high-low swing in the last 10 PDF days: {biggest_swing['date'].strftime('%Y-%m-%d')} with a swing of {biggest_swing['temp_swing']:.1f}F.",
        f"3. Warmer-than-average days in the last 10 PDF days: {len(warmer_than_average)} of {len(comparable_days)} comparable days.",
    ]

    record_findings = identify_record_events(last_10_days)
    if record_findings:
        lines.append("4. Record proximity findings:")
        lines.extend(f"   - {finding}" for finding in record_findings)
    else:
        lines.append("4. No last-10-day observations were within 2F of an all-time record high or low.")

    lines.append("5. Past 3-month trend summary:")
    for row in monthly_summary.itertuples(index=False):
        lines.append(
            f"   - {row.month}: average temp {row.avg_temp:.1f}F, warmest observed high {row.max_temp:.1f}F, total precipitation {row.precipitation:.2f} in."
        )

    return lines


def run_analysis() -> dict[str, pd.DataFrame]:
    weather_df = load_weather_data()
    last_10_days, merged_history = build_analysis_views(weather_df)
    create_charts(last_10_days, merged_history)

    for line in summarize_results(last_10_days, merged_history):
        print(line)

    return {
        "raw": weather_df,
        "last_10_days": last_10_days,
        "merged_history": merged_history,
    }


if __name__ == "__main__":
    run_analysis()
