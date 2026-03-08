from typing import Optional
import re

from langchain_ollama import ChatOllama

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


def find_company_aliases(text: str) -> list[str]:
    lowered = text.lower()
    found = []

    for alias, company_slug in COMPANY_ALIASES.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, lowered) and company_slug not in found:
            found.append(company_slug)

    return found


def extract_company(text: str) -> Optional[str]:
    for ticker in find_tickers(text):
        return NORMALIZED_TICKER_MAP[ticker]

    aliases = find_company_aliases(text)
    if aliases:
        return aliases[0]

    return None


def extract_explicit_company_name(text: str) -> Optional[str]:
    aliases = find_company_aliases(text)
    if aliases:
        return aliases[0]
    return None


def extract_companies(text: str) -> list[str]:
    companies: list[str] = []

    for company_slug in find_company_aliases(text):
        if company_slug not in companies:
            companies.append(company_slug)

    for ticker in find_tickers(text):
        company_slug = NORMALIZED_TICKER_MAP[ticker]
        if company_slug not in companies:
            companies.append(company_slug)

    return companies


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


def is_compare_query(text: str) -> bool:
    lowered = text.lower()
    compare_markers = {"compare", "vs", "versus", "difference", "different"}
    return any(marker in lowered for marker in compare_markers)


def validate(
    user_input: str,
    conversation_company: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    lowered = user_input.lower()
    companies = extract_companies(user_input)
    explicit = companies[0] if companies else extract_company(user_input)
    active = explicit or conversation_company
    has_intent = any(k in lowered for k in FINANCIAL_KEYWORDS)

    # allow true comparison queries with >= 2 companies
    if not (is_compare_query(user_input) and len(companies) >= 2):
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
    filing_citations: Optional[list[str]] = None,
    stock_context: Optional[dict] = None,
    stock_citation: Optional[str] = None,
) -> str:
    company_name = SLUG_TO_DISPLAY.get(company_slug, company_slug)

    filing_citations_text = "\n".join(f"• {c}" for c in (filing_citations or []))
    stock_citation_text = stock_citation if stock_citation else "None"

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
- For revenue trend questions, prefer total company revenue and overall year-over-year trend.
- Do not mix in segment growth or segment commentary unless the user explicitly asks about segments.
- If a value is not clearly available in the provided context, say it is not available instead of guessing.
- End the answer with a short "Sources:" section when source information is available.
- In the Sources section, use only the provided citation labels. Do not invent new ones.
- Put each source on its own new line as a bullet.
- Never place sources inline in a sentence.
- If the question is only about stock performance, do not mention filing availability unless the filing is directly relevant.
- If the question is only about filings, do not mention stock data.

Company: {company_name}
"""

    user_prompt = f"""
User question:
{user_input}

Filing context:
{filing_context if filing_context else "No filing data available."}

Filing citations:
{filing_citations_text if filing_citations_text else "None"}

Stock context:
{stock_context if stock_context else "No stock data available."}

Stock citation:
{stock_citation_text}

Write the final answer for the user.
"""
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content if hasattr(response, "content") else str(response)

def build_comparison_answer(
    user_input: str,
    company_a: str,
    company_b: str,
    context_a: Optional[str],
    context_b: Optional[str],
    citations_a: list[str],
    citations_b: list[str],
) -> str:
    company_a_name = SLUG_TO_DISPLAY.get(company_a, company_a)
    company_b_name = SLUG_TO_DISPLAY.get(company_b, company_b)

    citations_a_text = "\n".join(f"• {c}" for c in citations_a)
    citations_b_text = "\n".join(f"• {c}" for c in citations_b)

    system_prompt = """
You are Sovereign Financial Analyst, a precise financial research assistant.

Rules:
- Answer only the user's question.
- Compare only the two companies provided.
- Be structured, concise, and professional.
- Focus on the most material similarities and differences only.
- Give 2–3 similarities and 2–3 differences at most.
- Keep the full comparison under 8 sentences total, excluding the Sources section.
- Do not invent facts beyond the provided context.
- Do not add extra commentary or conclusions unless directly supported by the context.
- End the answer with a short "Sources:" section when source information is available.
- In the Sources section, use only the provided citation labels.
- Put each source on its own new line as a bullet.
- Never place sources inline in a sentence.

Preferred format:
Similarities:
- ...
- ...

Differences:
- ...
- ...
"""

    user_prompt = f"""
User question:
{user_input}

Company A: {company_a_name}
Context A:
{context_a if context_a else "No filing data available."}

Citations A:
{citations_a_text if citations_a_text else "None"}

Company B: {company_b_name}
Context B:
{context_b if context_b else "No filing data available."}

Citations B:
{citations_b_text if citations_b_text else "None"}

Write a comparison answer for the user.
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

    companies = extract_companies(user_input)

    # comparison mode
    if is_compare_query(user_input) and len(companies) >= 2:
        company_a, company_b = companies[0], companies[1]
        section = infer_section(user_input)

        result_a = query_financial_reports(
            query=user_input,
            company=company_a,
            fiscal_year=None,
            section=section,
        )
        result_b = query_financial_reports(
            query=user_input,
            company=company_b,
            fiscal_year=None,
            section=section,
        )

        reply = build_comparison_answer(
            user_input=user_input,
            company_a=company_a,
            company_b=company_b,
            context_a=result_a.get("content"),
            context_b=result_b.get("content"),
            citations_a=result_a.get("citations", []),
            citations_b=result_b.get("citations", []),
        )

        return reply, None

    effective_input = user_input
    if active_company and not extract_company(user_input):
        company_name = SLUG_TO_DISPLAY.get(active_company, active_company)
        effective_input = f"For {company_name}, {user_input}"

    needs_filings, needs_stock = infer_needs(effective_input)

    filing_context = None
    filing_citations = []
    stock_context = None
    stock_citation = None

    if needs_filings:
        section = infer_section(effective_input)
        filing_result = query_financial_reports(
            query=effective_input,
            company=active_company,
            fiscal_year=None,
            section=section,
        )
        filing_context = filing_result.get("content")
        filing_citations = filing_result.get("citations", [])

    if needs_stock:
        ticker = SLUG_TO_TICKER.get(active_company)
        if ticker:
            stock_result = get_stock_performance(ticker)
            stock_context = stock_result.get("data")
            stock_citation = stock_result.get("citation")

    reply = build_answer(
        user_input=effective_input,
        company_slug=active_company,
        filing_context=filing_context,
        filing_citations=filing_citations,
        stock_context=stock_context,
        stock_citation=stock_citation,
    )

    return reply, active_company