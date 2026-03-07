import streamlit as st
import sys
import os
import logging

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

SUGGESTIONS = [
    "What are the 3 main risks in Nvidia's 10-K?",
    "How is NVDA stock performing today?",
    "What is Nvidia's revenue trend from the filing?",
    "Summarize Nvidia's key business segments.",
]


def call_agent(prompt: str) -> str:
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.agent import run_agent_ui
        return run_agent_ui(prompt)
    except ImportError as e:
        return f"Could not load agent: {e}\n\nMake sure Ollama is running and your venv is active."
    except Exception as e:
        return f"Agent error: {e}"


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
        st.session_state.sc += 1
        st.rerun()

    st.caption("All data stays local · Zero telemetry")


st.title("Sovereign Financial Analyst")
st.caption("Ask anything about indexed 10-K filings and live stock data. Everything runs locally.")
st.divider()

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            reply = call_agent(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

if prompt := st.chat_input("Ask about a 10-K filing or stock performance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            reply = call_agent(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})