from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

from app.tools import get_stock_performance, query_financial_reports

llm = ChatOllama(model="llama3.1", temperature=0)
tools = [get_stock_performance, query_financial_reports]

system_prompt = (
    "You are Sovereign Financial Analyst, a privacy-first financial analysis assistant. "
    "Your job is to answer questions about company filings and stock performance using the available tools. "
    "Always keep the company consistent across the entire response and across all tool calls. "
    "Never switch to a different company unless the user explicitly asks you to. "
    "If the user does not specify a company, do not guess; clearly say that the company is not specified. "
    "Use query_financial_reports for questions about 10-K filings, risks, revenue, R&D, net income, business segments, or other filing-based information. "
    "Use get_stock_performance only when you know the correct ticker or when the user explicitly provides it. "
    "If the company name is given but the ticker is not, infer the ticker only when it is obvious and unambiguous, such as Nvidia -> NVDA or Apple -> AAPL. "
    "Base filing-related answers only on retrieved report content. "
    "Do not invent facts that are not present in the tool results. "
    "If tool results are incomplete or ambiguous, say so clearly. "
    "When answering, provide a concise, professional summary that separates filing insights from stock market insights. "
    "If the query_financial_reports tool returns that no data was found, explicitly tell the user that this company's filings are not in the database. Do NOT attempt to answer from general knowledge. Your answers must be grounded entirely in tool results. "
    "CRITICAL RULE: If the user asks about Company A's filings but Company B's stock ticker, you must flag this as an inconsistency and ask for clarification. Never silently substitute one company's stock data for another."
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

SUPPORTED_COMPANIES = {"nvidia", "nvda", "apple", "aapl", "tesla", "tsla"}
INDEXED_COMPANIES = {"nvidia", "nvda"}
FINANCIAL_KEYWORDS = {"10-k", "risk", "revenue", "filing", "stock", "performing", "earnings", "income", "trend"}
TICKER_TO_COMPANY = {"nvda": "nvidia", "aapl": "apple", "tsla": "tesla"}

def detect_mismatch(user_input: str) -> bool:
    lowered = user_input.lower()
    found_companies = [c for c in ["nvidia", "apple", "tesla"] if c in lowered]
    found_tickers = [t for t, c in TICKER_TO_COMPANY.items() if t in lowered]
    for ticker in found_tickers:
        mapped = TICKER_TO_COMPANY[ticker]
        if found_companies and mapped not in found_companies:
            return True
    return False

def run_agent_ui(user_input: str) -> str:
    lowered = user_input.lower()

    has_company = any(c in lowered for c in SUPPORTED_COMPANIES)
    has_financial_intent = any(k in lowered for k in FINANCIAL_KEYWORDS)

    if has_financial_intent and not has_company:
        return "Please specify a company. Currently indexed: **Nvidia (NVDA)**."

    if not has_financial_intent and not has_company:
        return "I can only answer financial questions about indexed companies. Try asking about Nvidia's 10-K risks or NVDA stock performance."

    if detect_mismatch(user_input):
        return (
            "⚠ **Inconsistency detected**: the company name and stock ticker in your query "
            "refer to different companies. Please clarify which company you want data for."
        )

    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content

    return "No response generated. Please try rephrasing your question."

def run_agent(user_input: str):
    lowered = user_input.lower()

    has_company = any(c in lowered for c in SUPPORTED_COMPANIES)
    has_financial_intent = any(k in lowered for k in FINANCIAL_KEYWORDS)

    if has_financial_intent and not has_company:
        print("Please specify a company. Currently indexed: Nvidia (NVDA).")
        return

    if not has_financial_intent and not has_company:
        print("I can only answer financial questions about indexed companies.")
        return

    if detect_mismatch(user_input):
        print("Inconsistency detected: the company name and stock ticker refer to different companies. Please clarify.")
        return

    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(result["messages"][-1].content)


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
        "For Nvidia, what are the top risks and how is the stock performing?"
    ]

    for i, query in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"TEST CASE {i}: {query}")
        print("=" * 80)
        run_agent(query)