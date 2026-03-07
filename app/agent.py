from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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


def detect_mismatch(user_input: str) -> bool:
    """Return True if user mentions Company A but ticker for Company B."""
    lowered = user_input.lower()
    found_companies = [c for c in ["nvidia", "apple", "tesla"] if c in lowered]
    found_tickers = [t for t in TICKER_TO_COMPANY if t in lowered]

    for ticker in found_tickers:
        mapped = TICKER_TO_COMPANY[ticker]
        if found_companies and mapped not in found_companies:
            return True
    return False


def validate_query(user_input: str) -> str | None:
    """
    Returns an error string if the query should be blocked,
    or None if it's safe to pass to the agent.
    """
    lowered = user_input.lower()
    has_company = any(c in lowered for c in SUPPORTED_COMPANIES)
    has_financial_intent = any(k in lowered for k in FINANCIAL_KEYWORDS)

    if has_financial_intent and not has_company:
        return "Please specify a company. Currently indexed: **Nvidia (NVDA)**."

    if not has_financial_intent and not has_company:
        return (
            "I can only answer financial questions about indexed companies. "
            "Try asking about Nvidia's 10-K risks or NVDA stock performance."
        )

    if detect_mismatch(user_input):
        return (
            "⚠ **Inconsistency detected**: the company name and stock ticker in your "
            "query refer to different companies. Please clarify which company you want data for."
        )

    return None


def extract_final_response(result: dict) -> str:
    """Pull the last meaningful AI message from the agent result."""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return "No response generated. Please try rephrasing your question."


def ask_agent(user_input: str) -> str:
    """Single entrypoint for both CLI and UI."""
    error = validate_query(user_input)
    if error:
        return error

    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    return extract_final_response(result)


def run_agent(user_input: str):
    """CLI runner — prints the response."""
    print(ask_agent(user_input))


if __name__ == "__main__":
    test_cases = [
        "For Nvidia, what are the 3 main risks in the 10-K and how is NVDA stock performing today?",
        "What are the 3 main risks in the 10-K and how is the stock performing today?",
        "For Apple, what are the main risks in the 10-K and how is AAPL performing today?",
        "For Nvidia, what is the revenue trend and how is the stock doing?",
        "For Tesla, summarize the 10-K risks and stock performance.",
        "How is NVDA performing today?",
        "Compare Nvidia and Apple risk factors.",
        "For Nvidia, what are the top 3 risks and how is AAPL performing today?",
        "For Nvidia, what are the top risks and how is the stock performing?",
    ]

    for i, query in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"TEST CASE {i}: {query}")
        print("=" * 80)
        run_agent(query)