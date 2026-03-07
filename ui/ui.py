import logging
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.agent import ask_agent

logging.getLogger("tornado.websocket").setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="Sovereign Financial Analyst",
    page_icon="📊",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "sc" not in st.session_state:
    st.session_state.sc = 0

SUGGESTIONS = [
    "What are the 3 main risks in Nvidia's 10-K?",
    "How is NVDA stock performing today?",
    "What is Nvidia's revenue trend from the filing?",
    "Summarize Nvidia's key business segments.",
]


# ── Single prompt handler — used by both chips and chat input ─────────────────
def process_prompt(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            reply = ask_agent(prompt)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Sovereign FA")
    st.caption("Privacy-first · Local LLM · Agentic RAG")
    st.divider()

    st.subheader("System Status")
    st.success("Ollama · llama3.1 · Running")
    st.info("ChromaDB · Local · Ready")
    st.divider()

    st.subheader("Indexed Filings")
    st.markdown("**NVDA** — Nvidia Corporation")
    st.code("10-K · FY2024", language=None)
    st.divider()

    st.subheader("Add Filing")
    uploaded = st.file_uploader("Upload a 10-K PDF", type=["pdf"])
    if uploaded:
        st.warning(f"Ingestion coming soon.\n\n**{uploaded.name}** received.")
    st.divider()

    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.sc += 1
        st.rerun()

    st.caption("All data stays local · Zero telemetry")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Sovereign Financial Analyst")
st.caption("Ask anything about indexed 10-K filings and live stock data. Everything runs locally.")
st.divider()

# Suggestions only shown when chat is empty
if not st.session_state.messages:
    st.markdown("**Suggested queries**")
    col1, col2 = st.columns(2)
    for i, text in enumerate(SUGGESTIONS):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(text, key=f"s{i}_{st.session_state.sc}", use_container_width=True):
                st.session_state.pending = text
                st.session_state.sc += 1
                st.rerun()
    st.divider()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle suggestion chip click
if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None
    process_prompt(prompt)

# Handle typed input
if prompt := st.chat_input("Ask about a 10-K filing or stock performance..."):
    process_prompt(prompt)