#agent.py
from typing import Any, Iterator

from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.tools import get_stock_performance, query_financial_reports
from app.config import (
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
    SUPPORTED_COMPANIES,
    FINANCIAL_KEYWORDS,
    TICKER_TO_COMPANY,
)

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
tools = [get_stock_performance, query_financial_reports]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SystemMessage(content=SYSTEM_PROMPT),
)

NORMALIZED_TICKER_TO_COMPANY = {k.lower(): v for k, v in TICKER_TO_COMPANY.items()}

COMPANY_ALIASES = {
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
    "general electric": "General Electric",
    "jpmorgan chase": "JPMorgan Chase",
    "goldman sachs": "Goldman Sachs",
    "visa": "Visa",
    "johnson & johnson": "Johnson & Johnson",
    "eli lilly": "Eli Lilly",
    "pfizer": "Pfizer",
    "exxonmobil": "ExxonMobil",
    "walmart": "Walmart",
}


def extract_company_from_text(user_input: str) -> str | None:
    lowered = user_input.lower()

    for ticker, company in NORMALIZED_TICKER_TO_COMPANY.items():
        if ticker in lowered:
            return company

    for alias, company in COMPANY_ALIASES.items():
        if alias in lowered:
            return company

    return None


def detect_mismatch(user_input: str) -> bool:
    lowered = user_input.lower()
    explicit_company = extract_company_from_text(user_input)

    found_tickers = [ticker for ticker in NORMALIZED_TICKER_TO_COMPANY if ticker in lowered]
    for ticker in found_tickers:
        mapped_company = NORMALIZED_TICKER_TO_COMPANY[ticker]
        if explicit_company and mapped_company.lower() != explicit_company.lower():
            return True
    return False


def validate_query(
    user_input: str,
    conversation_company: str | None = None,
) -> tuple[str | None, str | None]:
    lowered = user_input.lower()
    explicit_company = extract_company_from_text(user_input)
    active_company = explicit_company or conversation_company
    has_financial_intent = any(k in lowered for k in FINANCIAL_KEYWORDS)

    if detect_mismatch(user_input):
        return (
            "⚠ **Inconsistency detected**: the company name and stock ticker in your query refer to different companies. Please clarify which company you want data for.",
            active_company,
        )

    if has_financial_intent and not active_company:
        return "Please specify a company.", None

    if not has_financial_intent and not active_company:
        return (
            "This system is designed for company-specific financial analysis. "
            "Ask about a company’s 10-K risks, revenue trends, business segments, or stock performance.",
            None,
        )

    return None, active_company


def build_effective_input(
    user_input: str,
    conversation_company: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Returns:
      (error_message, effective_input, active_company)
    """
    error, active_company = validate_query(user_input, conversation_company)
    if error:
        return error, None, active_company

    effective_input = user_input
    if active_company and not extract_company_from_text(user_input):
        effective_input = f"For {active_company}, {user_input}"

    return None, effective_input, active_company


def _chunk_to_text(chunk: Any) -> str:
    """
    Extract text safely from streamed LangGraph/LangChain message chunks.
    """
    content = getattr(chunk, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


def extract_final_response(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return "No response generated. Please try rephrasing your question."


def ask_agent_stream(
    user_input: str,
    conversation_company: str | None = None,
) -> Iterator[dict]:
    error, effective_input, active_company = build_effective_input(user_input, conversation_company)
    if error:
        yield {"type": "token", "content": error}
        yield {"type": "done", "company": active_company}
        return

    emitted_any = False

    try:
        for chunk, metadata in agent.stream(
            {"messages": [HumanMessage(content=effective_input)]},
            stream_mode="messages",
        ):
            if (
                isinstance(chunk, AIMessage)
                and not getattr(chunk, "tool_calls", None)
                and chunk.content
            ):
                text = _chunk_to_text(chunk)
                if text:
                    emitted_any = True
                    yield {"type": "token", "content": text}

    except Exception as e:
        print(f"Streaming failed, falling back: {e}")
        reply, _ = ask_agent(user_input, conversation_company)
        yield {"type": "token", "content": reply}
        yield {"type": "done", "company": active_company}
        return

    if not emitted_any:
        reply, _ = ask_agent(user_input, conversation_company)
        yield {"type": "token", "content": reply}

    yield {"type": "done", "company": active_company}