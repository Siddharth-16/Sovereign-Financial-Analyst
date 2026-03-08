import logging
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.agent import ask_agent  

logging.getLogger("tornado.websocket").setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="Sovereign Financial Analyst",
    page_icon="📊",
    layout="centered",
)

DISPLAY_NAMES = {
    "nvidia": "Nvidia",
    "apple": "Apple",
    "tesla": "Tesla",
    "microsoft": "Microsoft",
    "amazon": "Amazon",
    "alphabet": "Alphabet",
    "meta": "Meta",
    "amd": "AMD",
    "broadcom": "Broadcom",
    "caterpillar": "Caterpillar",
    "boeing": "Boeing",
    "general_electric": "General Electric",
    "jpmorgan_chase": "JPMorgan Chase",
    "goldman_sachs": "Goldman Sachs",
    "visa": "Visa",
    "johnson_and_johnson": "Johnson & Johnson",
    "eli_lilly": "Eli Lilly",
    "pfizer": "Pfizer",
    "exxonmobil": "ExxonMobil",
    "walmart": "Walmart",
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "sc" not in st.session_state:
    st.session_state.sc = 0
if "current_company" not in st.session_state:
    st.session_state.current_company = None

SUGGESTIONS = [
    "What are the 3 main risks in Nvidia's 10-K?",
    "How is NVDA stock performing today?",
    "What is Nvidia's revenue trend from the filing?",
    "Compare Nvidia and AMD risk factors.",
]


def process_prompt(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            reply, active_company = ask_agent(
                prompt,
                st.session_state.current_company,
            )
        st.markdown(reply)

    st.session_state.current_company = active_company
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.sc += 1

with st.sidebar:
    st.title("📊 Sovereign FA")
    st.caption("Privacy-first · Local LLM · Agentic RAG")
    st.divider()

    st.subheader("System Status")
    st.success("Ollama · llama3.1 · Running")
    st.info("ChromaDB · Local · Ready")
    st.divider()

    st.subheader("Indexed Filings")
    st.markdown("**20 companies · FY2023–FY2025**")
    if st.session_state.current_company:
        pretty = DISPLAY_NAMES.get(st.session_state.current_company, st.session_state.current_company)
        st.caption(f"Active context: **{pretty}**")
    st.divider()

    st.subheader("Upload Filing (Future Feature)")
    uploaded = st.file_uploader("Upload a 10-K PDF", type=["pdf"])

    if uploaded:
        st.info(
            f"""
    PDF upload detected: **{uploaded.name}**

    Dynamic ingestion is not enabled in this demo version.

    To add new filings, place them in the `data/raw/` directory and run the ingestion pipeline.
    """
        )

    st.divider()

    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.current_company = None
        st.session_state.sc += 1
        st.rerun()

    st.caption("All data stays local · Zero telemetry")

st.title("Sovereign Financial Analyst")
st.caption("Ask anything about indexed 10-K filings and live stock data. Everything runs locally.")
st.divider()

with st.expander("💡 Suggested queries", expanded=len(st.session_state.messages) == 0):
    col1, col2 = st.columns(2)
    for i, text in enumerate(SUGGESTIONS):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(text, key=f"s{i}_{st.session_state.sc}", use_container_width=True):
                st.session_state.pending = text
                st.session_state.sc += 1
                st.rerun()

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None
    process_prompt(prompt)

if prompt := st.chat_input("Ask about a 10-K filing or stock performance..."):
    process_prompt(prompt)