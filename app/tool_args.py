from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_QUERY_STOPWORDS = {
    "a", "an", "and", "about", "are", "as", "at", "by", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "say", "says", "the", "their",
    "this", "to", "was", "were", "what", "which", "with",
}
_COMPANY_NOISE = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "plc"}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def _query_has_content(question: str, company: str) -> bool:
    question_tokens = _tokens(question) - _QUERY_STOPWORDS
    company_tokens = _tokens(company) - _COMPANY_NOISE
    return bool(question_tokens - company_tokens)


def _sanitize_fiscal_year(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return None
        match = re.fullmatch(r"(?:fy\s*)?(20\d{2})", stripped)
        if match:
            return int(match.group(1))
    # A malformed optional filter should not make the entire filing search fail.
    return None



_VALID_SECTIONS = {"business", "risk_factors", "mdna", "financial_statements"}


def _infer_section_from_question(question: str) -> str | None:
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    if not q:
        return None

    # Explicit section language wins.
    if any(term in q for term in ("md&a", "management's discussion", "management’s discussion")):
        return "mdna"
    if "risk factor" in q or re.search(r"\brisk(?:s)?\b", q):
        # Avoid stealing an explicitly financial question that merely mentions risk.
        if not any(term in q for term in ("balance sheet", "financial statement", "cash flow statement")):
            return "risk_factors"
    if any(term in q for term in (
        "balance sheet", "financial statement", "financial statements",
        "statement of cash flows", "cash flow statement", "income statement",
        "total assets", "total liabilities", "total debt",
    )):
        return "financial_statements"

    # High-confidence business intents. This repairs cases such as
    # "main reportable business segments" being routed to financial notes.
    if any(term in q for term in (
        "reportable business segment", "business segments", "business segment",
        "business model", "core business", "product lines", "store formats",
        "therapeutic areas", "products and services", "business strategy",
    )):
        return "business"

    return None


def sanitize_tool_args(name: str, args: dict | None, *, user_input: str) -> dict:
    """Normalize recoverable LLM tool-call mistakes before invocation.

    The raw model arguments should still be retained separately for auditability.
    """
    cleaned = dict(args or {})
    if name != "search_filing":
        return cleaned

    cleaned["fiscal_year"] = _sanitize_fiscal_year(cleaned.get("fiscal_year"))

    section = cleaned.get("section")
    if isinstance(section, str) and not section.strip():
        cleaned["section"] = None

    inferred_section = _infer_section_from_question(user_input)
    if inferred_section is None:
        inferred_section = _infer_section_from_question(str(cleaned.get("question") or ""))
    if inferred_section in _VALID_SECTIONS:
        cleaned["section"] = inferred_section

    company = str(cleaned.get("company") or "").strip()
    question = str(cleaned.get("question") or "").strip()

    # Example caught by this rule: question="What does Microsoft" with
    # company="Microsoft". After removing boilerplate/company tokens there is
    # no retrieval intent left, so use the original user question instead.
    if not question or not _query_has_content(question, company):
        cleaned["question"] = user_input.strip()
    else:
        cleaned["question"] = question

    if company:
        cleaned["company"] = company

    return cleaned
