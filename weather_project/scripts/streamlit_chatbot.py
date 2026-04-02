from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analysis import build_analysis_views, identify_record_events, load_weather_data, summarize_results
from chatbot import extract_date_from_query, run_query

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


def load_context() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    weather_df = load_weather_data()
    last_10_days, merged_history = build_analysis_views(weather_df)
    summary_lines = summarize_results(last_10_days, merged_history)
    return last_10_days, merged_history, summary_lines


def dataframe_preview(frame: pd.DataFrame, columns: list[str], limit: int) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return "No rows available."
    return frame[available].head(limit).to_csv(index=False)


def build_grounding_context(last_10_days: pd.DataFrame, merged_history: pd.DataFrame, summary_lines: list[str], question: str) -> str:
    monthly = (
        merged_history.dropna(subset=["date", "avg_temp"])
        .assign(month=lambda df: df["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(avg_temp=("avg_temp", "mean"), max_temp=("max_temp", "max"), precipitation=("precipitation", "sum"))
    )
    target_date = extract_date_from_query(question)
    matched_rows = pd.DataFrame()
    if target_date is not None:
        matched_rows = merged_history[merged_history["date"] == target_date].copy()

    sections = [
        "Project scope: Downtown LA weather analysis using parsed PDF reports and cleaned CSV history.",
        "Use only the data below. If the data is insufficient, say that clearly.",
        "",
        "Assignment summary:",
        "\n".join(summary_lines),
        "",
        "Last 10 PDF days preview:",
        dataframe_preview(
            last_10_days.sort_values("date"),
            [
                "date",
                "max_temp",
                "min_temp",
                "avg_temp",
                "normal_max_temp",
                "effective_normal_avg_temp",
                "temp_swing",
                "record_high_temp",
                "record_low_temp",
            ],
            10,
        ),
        "",
        "Three-month monthly summary:",
        dataframe_preview(monthly, ["month", "avg_temp", "max_temp", "precipitation"], 12),
        "",
        "Full merged history table:",
        dataframe_preview(
            merged_history.sort_values("date"),
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
                "temp_swing",
                "source_file",
                "pdf_source_file",
            ],
            len(merged_history),
        ),
    ]

    if not matched_rows.empty:
        sections.extend(
            [
                "",
                "Exact rows matching the date in the question:",
                dataframe_preview(
                    matched_rows,
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
                        "temp_swing",
                        "source_file",
                        "pdf_source_file",
                    ],
                    len(matched_rows),
                ),
            ]
        )

    record_events = identify_record_events(last_10_days)
    if record_events:
        sections.extend(["", "Record-related findings:", "\n".join(record_events)])

    return "\n".join(sections)


def answer_with_openai(api_key: str, question: str, context: str, model: str) -> str:
    if OpenAI is None:
        return "OpenAI package is not installed, so the app is using local answers only."

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a grounded weather data assistant. Answer only from the provided context. "
                            "Do not invent facts. Keep answers concise, specific, and mention uncertainty when data is missing."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Weather database context:\n{context}"},
                    {"type": "input_text", "text": f"Question: {question}"},
                ],
            },
        ],
    )
    return (response.output_text or "").strip() or "OpenAI returned an empty response."


def answer_question(question: str, use_openai: bool, api_key: str, model: str, context: str) -> tuple[str, str]:
    local_answer = run_query(question)
    if not use_openai:
        return local_answer, "Local database logic"

    if not api_key.strip():
        return local_answer, "Local database logic (no API key provided)"

    try:
        openai_answer = answer_with_openai(api_key.strip(), question, context, model)
        return openai_answer, f"OpenAI + grounded database context ({model})"
    except Exception as exc:  # pragma: no cover
        fallback = f"{local_answer}\n\nOpenAI fallback note: {exc}"
        return fallback, "Local database logic (OpenAI request failed)"


def render_sidebar() -> tuple[bool, str, str]:
    st.sidebar.header("Chat Settings")
    use_openai = st.sidebar.toggle("Use OpenAI if API key is provided", value=False)
    api_key = st.sidebar.text_input("Paste OpenAI API key", type="password")
    model = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")

    st.sidebar.markdown(
        "Without an API key, the app still answers using the existing database-backed local logic."
    )
    return use_openai, api_key, model


def render_data_overview(last_10_days: pd.DataFrame, merged_history: pd.DataFrame, summary_lines: list[str]) -> None:
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("PDF Days", int(len(last_10_days)))
    col2.metric("History Rows", int(len(merged_history)))
    col3.metric("Record Alerts", int(len(identify_record_events(last_10_days))))

    with st.expander("Current assignment summary", expanded=False):
        st.text("\n".join(summary_lines))

    with st.expander("Last 10 PDF days", expanded=False):
        st.dataframe(last_10_days.sort_values("date"), use_container_width=True)

    with st.expander("Merged 3-month history", expanded=False):
        st.dataframe(merged_history.sort_values("date", ascending=False), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Weather Data Chatbot", page_icon=":partly_sunny:", layout="wide")
    st.title("Downtown LA Weather Chatbot")
    st.caption("Ask questions grounded in the SQLite weather database. OpenAI is optional.")

    use_openai, api_key, model = render_sidebar()

    try:
        last_10_days, merged_history, summary_lines = load_context()
    except Exception as exc:
        st.error(f"Could not load weather data: {exc}")
        st.info("Run `python scripts/db_loader.py` first if the database has not been created yet.")
        return

    context = build_grounding_context(last_10_days, merged_history, summary_lines, question="")
    render_data_overview(last_10_days, merged_history, summary_lines)

    st.subheader("Ask a Question")
    suggestions = [
        "What was the hottest day in the last 10 days?",
        "Which day had the biggest swing between high and low temperature?",
        "How many of the last 10 days were warmer than average?",
        "Did any of the last 10 days set or come close to breaking a record?",
        "What are the trends in the past 3 months?",
    ]

    selected = st.selectbox("Sample questions", [""] + suggestions)
    default_question = selected if selected else "What was the hottest day in the last 10 days?"
    question = st.text_area("Your question", value=default_question, height=100)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("Get Answer", type="primary"):
        context = build_grounding_context(last_10_days, merged_history, summary_lines, question)
        answer, source_label = answer_question(question, use_openai, api_key, model, context)
        st.session_state.messages.append(
            {
                "question": question,
                "answer": answer,
                "source": source_label,
            }
        )

    if st.session_state.messages:
        st.subheader("Conversation")
        for item in reversed(st.session_state.messages):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.caption(f"Answer source: {item['source']}")
            st.divider()


if __name__ == "__main__":
    main()
