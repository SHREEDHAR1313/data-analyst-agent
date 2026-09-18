from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent_gemini import DataAnalystAgent

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="centered")
st.title("📊 AI Data Analyst Agent")
st.caption(
    "Upload a CSV and ask questions in plain English. Gemini selects deterministic Python analysis tools, "
    "and the app shows the tool trace and generated charts."
)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = st.text_input(
        "Google AI API key",
        type="password",
        help="Create an API key in Google AI Studio and paste it here. The key is used only for this session.",
    )

if not api_key:
    st.info("Add your Google AI API key above or set GOOGLE_API_KEY in .env to start.")
    st.stop()

uploaded = st.file_uploader("Upload a CSV", type=["csv"])
use_sample = st.checkbox("Use the included sample sales dataset", value=uploaded is None)

try:
    if use_sample:
        sample_path = BASE_DIR / "sample_data" / "sales_data.csv"
        df = pd.read_csv(sample_path)
        source_name = "sample_data/sales_data.csv"
    elif uploaded is not None:
        df = pd.read_csv(uploaded)
        source_name = uploaded.name
    else:
        st.stop()
except pd.errors.EmptyDataError:
    st.error("The CSV is empty. Please upload a CSV containing data.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("The CSV contains no usable data.")
    st.stop()

# Stable dataset key so Streamlit reruns do not wipe the conversation.
hash_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
dataset_key = hashlib.sha256(source_name.encode() + hash_bytes).hexdigest()

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

if st.session_state.get("dataset_key") != dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.agent = DataAnalystAgent(df, api_key=api_key)
    st.session_state.history = []

st.divider()

example_questions = [
    "Give me an overview of this dataset and flag any data quality issues.",
    "What's driving revenue the most?",
    "Are there any outliers I should know about?",
    "Show me revenue by region as a chart.",
]

cols = st.columns(2)
clicked = None
for i, question_text in enumerate(example_questions):
    if cols[i % 2].button(question_text, use_container_width=True):
        clicked = question_text

question = st.chat_input("Ask a question about your data...") or clicked

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for chart_path in entry.get("charts", []):
            if Path(chart_path).exists():
                st.image(chart_path)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                result = st.session_state.agent.ask(question, verbose=False)
                st.markdown(result["answer"])
                for chart_path in result["charts"]:
                    if Path(chart_path).exists():
                        st.image(chart_path)

                with st.expander("See what the agent actually did (tool calls)"):
                    if result["tool_trace"]:
                        for step in result["tool_trace"]:
                            st.code(f"{step['tool']}({step['input']})", language="python")
                            st.json(step["result"])
                    else:
                        st.caption("No Python analysis tool was called for this question.")

                st.session_state.history.append(
                    {"role": "assistant", "content": result["answer"], "charts": result["charts"]}
                )
            except Exception as exc:
                st.error(f"Something went wrong: {type(exc).__name__}: {exc}")
