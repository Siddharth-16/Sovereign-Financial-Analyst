#ui.py
import logging
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.agent import ask_agent_stream

logging.getLogger("tornado.websocket").setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="Sovereign Financial Analyst",
    page_icon="📊",
    layout="centered",
)

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
    "Summarize Nvidia's key business segments.",
]


def process_prompt(prompt: str):
    st.session_state.sc += 1 
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    current_company = st.session_state.current_company
    final_company = current_company
    collected_chunks: list[str] = []

    with st.chat_message("assistant"):
        def response_stream():
            nonlocal final_company

            for event in ask_agent_stream(prompt, current_company):
                if event["type"] == "token":
                    collected_chunks.append(event["content"])
                    yield event["content"]
                elif event["type"] == "done":
                    final_company = event["company"]

        rendered = st.write_stream(response_stream)

    reply = rendered if isinstance(rendered, str) else "".join(collected_chunks)
    st.session_state.current_company = final_company
    st.session_state.messages.append({"role": "assistant", "content": reply})


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
        st.caption(f"Active company context: {st.session_state.current_company}")
    st.divider()

    st.subheader("Add Filing")
    uploaded = st.file_uploader("Upload a 10-K PDF", type=["pdf"])
    if uploaded:
        st.warning(f"Ingestion coming soon.\n\n**{uploaded.name}** received.")
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

if not st.session_state.messages:
    st.markdown("**Suggested queries**")
    col1, col2 = st.columns(2)
    for i, text in enumerate(SUGGESTIONS):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(text, key=f"s{i}_{st.session_state.sc}", use_container_width=True):
                st.session_state.pending = text
                st.session_state.sc += 1  # must increment BEFORE rerun
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