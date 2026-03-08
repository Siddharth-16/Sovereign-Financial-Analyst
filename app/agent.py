from typing import Optional
from langchain_ollama import ChatOllama
import re

from app.tools import (
    get_stock_performance,
    query_financial_reports,
    TICKER_TO_COMPANY,
    SLUG_TO_DISPLAY,
    SLUG_TO_TICKER,
)

from app.config import OLLAMA_MODEL, FINANCIAL_KEYWORDS

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

NORMALIZED_TICKER_MAP = {k.lower(): v for k, v in TICKER_TO_COMPANY.items()}

COMPANY_ALIASES = {
    "nvidia": "nvidia",
    "apple": "apple",
    "tesla": "tesla",
    "microsoft": "microsoft",
    "amazon": "amazon",
    "alphabet": "alphabet",
    "google": "alphabet",
    "meta": "meta",
    "amd": "amd",
    "broadcom": "broadcom",
    "caterpillar": "caterpillar",
    "boeing": "boeing",
    "general electric": "general_electric",
    "jpmorgan": "jpmorgan_chase",
    "jpmorgan chase": "jpmorgan_chase",
    "goldman sachs": "goldman_sachs",
    "visa": "visa",
    "johnson & johnson": "johnson_and_johnson",
    "eli lilly": "eli_lilly",
    "pfizer": "pfizer",
    "exxonmobil": "exxonmobil",
    "walmart": "walmart",
}

def find_tickers(text: str) -> list[str]:
    lowered = text.lower()
    found = []

    for ticker in NORMALIZED_TICKER_MAP:
        pattern = rf"\b{re.escape(ticker)}\b"
        if re.search(pattern, lowered):
            found.append(ticker)

    return found

def extract_company(text: str) -> Optional[str]:
    lowered = text.lower()

    for ticker in find_tickers(text):
        return NORMALIZED_TICKER_MAP[ticker]

    for alias, company_slug in COMPANY_ALIASES.items():
        if alias in lowered:
            return company_slug

    return None


def extract_explicit_company_name(text: str) -> Optional[str]:
    lowered = text.lower()

    for alias, company_slug in COMPANY_ALIASES.items():
        if alias in lowered:
            return company_slug

    return None


def detect_mismatch(text: str) -> bool:
    explicit_company = extract_explicit_company_name(text)

    if explicit_company is None:
        return False

    found_tickers = find_tickers(text)

    for ticker in found_tickers:
        mapped_company = NORMALIZED_TICKER_MAP[ticker]
        if mapped_company != explicit_company:
            return True

    return False


def validate(
    user_input: str,
    conversation_company: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    lowered = user_input.lower()
    explicit = extract_company(user_input)
    active = explicit or conversation_company
    has_intent = any(k in lowered for k in FINANCIAL_KEYWORDS)

    if detect_mismatch(user_input):
        return (
            "⚠ **Inconsistency**: the company name and ticker refer to different companies. Please clarify.",
            active,
        )

    if has_intent and not active:
        return "Please specify a company to analyze.", None

    if not has_intent and not active:
        return (
            "I can only answer questions about company 10-K filings and stock performance.",
            None,
        )

    return None, active


def infer_needs(user_input: str) -> tuple[bool, bool]:
    lowered = user_input.lower()

    filing_keywords = {
        "10-k", "risk", "risks", "revenue", "r&d", "net income",
        "business", "segment", "segments", "filing", "strategy",
        "risk factors", "revenue trend", "summarize", "compare"
    }

    stock_keywords = {
        "stock", "price", "performing", "performance", "trading",
        "market", "high", "low", "volume"
    }

    needs_filings = any(k in lowered for k in filing_keywords)
    needs_stock = any(k in lowered for k in stock_keywords)

    return needs_filings, needs_stock

def infer_section(user_input: str) -> Optional[str]:
    lowered = user_input.lower()

    risk_keywords = {
        "risk", "risks", "risk factors", "regulatory risk",
        "supply chain risk", "export risk", "geopolitical risk"
    }

    mdna_keywords = {
        "revenue", "revenue trend", "growth", "margin", "gross margin",
        "operating income", "profit", "net income", "cash flow",
        "drivers", "financial performance", "results of operations"
    }

    business_keywords = {
        "business", "business model", "segments", "segment",
        "products", "customers", "competition", "strategy",
        "strategic priorities", "market position"
    }

    financial_keywords = {
        "balance sheet", "income statement", "financial statements",
        "financials", "assets", "liabilities"
    }

    if any(k in lowered for k in risk_keywords):
        return "risk_factors"

    if any(k in lowered for k in mdna_keywords):
        return "mdna"

    if any(k in lowered for k in business_keywords):
        return "business"

    if any(k in lowered for k in financial_keywords):
        return "financial_statements"

    return None


def build_answer(
    user_input: str,
    company_slug: str,
    filing_context: Optional[str] = None,
    stock_context: Optional[dict] = None,
) -> str:
    company_name = SLUG_TO_DISPLAY.get(company_slug, company_slug)

    system_prompt = f"""
You are Sovereign Financial Analyst, a precise financial research assistant.

Rules:
- Answer only the user's question.
- Do not mention tools, function calls, or internal system behavior.
- Do not output JSON.
- Keep the answer concise and professional.
- If the user asks for 3 risks, provide exactly 3.
- If filing data is unavailable, state that directly.
- If stock data is unavailable, state that directly.
- Do not invent facts beyond the provided context.
- Ignore irrelevant or conflicting details from unrelated companies.
- For business segment questions, prefer formal reportable segments named in the filing.

Company: {company_name}
"""

    user_prompt = f"""
User question:
{user_input}

Filing context:
{filing_context if filing_context else "No filing data available."}

Stock context:
{stock_context if stock_context else "No stock data available."}

Write the final answer for the user.
"""

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content if hasattr(response, "content") else str(response)


def ask_agent(
    user_input: str,
    conversation_company: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    error, active_company = validate(user_input, conversation_company)
    if error:
        return error, active_company

    effective_input = user_input
    if active_company and not extract_company(user_input):
        company_name = SLUG_TO_DISPLAY.get(active_company, active_company)
        effective_input = f"For {company_name}, {user_input}"

    needs_filings, needs_stock = infer_needs(effective_input)

    filing_context = None
    stock_context = None

    if needs_filings:
        section = infer_section(effective_input)
        filing_context = query_financial_reports(
            query=effective_input,
            company=active_company,
            fiscal_year=None,
            section=section,
        )

    if needs_stock:
        ticker = SLUG_TO_TICKER.get(active_company)
        if ticker:
            stock_context = get_stock_performance(ticker)

    reply = build_answer(
        user_input=effective_input,
        company_slug=active_company,
        filing_context=filing_context,
        stock_context=stock_context,
    )

    return reply, active_company