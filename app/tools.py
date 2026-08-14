from typing import Optional
import logging
import re
import yfinance as yf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_PATH, EMBED_MODEL
from app.companies import (
    COMPANY_NAME_MAP,
    TICKER_TO_SLUG as TICKER_TO_COMPANY,
    SLUG_TO_DISPLAY,
    SLUG_TO_TICKER,  # noqa: F401 -- re-exported, consumed by app.agent
    SECTION_NAME_MAP,
    SECTION_DISPLAY_MAP,
)
from app.exceptions import VectorStoreUnavailableError
from app.retrieval import hybrid_retrieve

logger = logging.getLogger("sovereign_fa.tools")

INVALID_TICKERS = {"STOCK TICKER", "TICKER", "COMPANY", ""}

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        try:
            _vectorstore = Chroma(
                persist_directory=CHROMA_PATH, embedding_function=get_embeddings()
            )
        except Exception as exc:
            logger.error("vector_store_init_failed", extra={"error": str(exc)})
            raise VectorStoreUnavailableError(
                f"Could not initialize Chroma at '{CHROMA_PATH}': {exc}"
            ) from exc
    return _vectorstore


def reset_vectorstore_cache() -> None:
    """Test-only helper: clears the cached singletons so a fresh (possibly
    mocked) vector store / embeddings model gets built on next use."""
    global _embeddings, _vectorstore
    _embeddings = None
    _vectorstore = None


def normalize_section(section: Optional[str]) -> Optional[str]:
    if not section:
        return None
    lowered = section.strip().lower()
    return SECTION_NAME_MAP.get(lowered, lowered)


def clean_filing_text(text: str) -> str:
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\bTable of Contents\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Item\s+1A\.\s*Risk Factors", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "plc",
}


def _canonical_company_name(name: str) -> str:
    """Normalize punctuation and common legal suffixes for company matching."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", name.lower())
    words = [word for word in normalized.split() if word not in _LEGAL_SUFFIXES]
    return " ".join(words)


_COMPANY_CANONICAL_MAP = {
    _canonical_company_name(name): slug
    for name, slug in COMPANY_NAME_MAP.items()
}


def normalize_company(company: Optional[str]) -> Optional[str]:
    if not company:
        return None

    company = company.strip()

    if company.upper() in TICKER_TO_COMPANY:
        return TICKER_TO_COMPANY[company.upper()]

    lowered = company.lower()
    if lowered in COMPANY_NAME_MAP:
        return COMPANY_NAME_MAP[lowered]

    canonical = _canonical_company_name(company)
    if canonical in _COMPANY_CANONICAL_MAP:
        return _COMPANY_CANONICAL_MAP[canonical]

    return lowered


def format_filing_citation(company_name: str, fiscal_year: int | str, section: str) -> str:
    section_display = SECTION_DISPLAY_MAP.get(section, section.replace("_", " ").title())
    return f"{company_name} 10-K FY{fiscal_year} – {section_display}"


def format_stock_citation(ticker: str) -> str:
    return f"{ticker} market data – latest 5d window"


def _expand_retrieval_query(query: str) -> str:
    """Add filing terminology that often appears in tables/10-K text.

    The expansion is intentionally conservative: it does not add facts, only
    common synonymous labels for the concept already present in the question.
    """
    lowered = query.lower()
    additions: list[str] = []

    if "cash flow from operations" in lowered or "operating cash flow" in lowered:
        additions.extend([
            "net cash provided by operating activities",
            "net cash from operating activities",
        ])

    if "debt" in lowered:
        additions.extend([
            "total debt",
            "long-term debt",
            "current portion of long-term debt",
            "short-term debt",
            "borrowings",
        ])

    if "capital expenditure" in lowered or "capital expenditures" in lowered or "capex" in lowered:
        additions.extend([
            "capital expenditures",
            "capital and exploration expenditures",
            "additions to property plant and equipment",
            "property plant and equipment additions",
        ])

    if "total revenue" in lowered:
        additions.extend([
            "total revenue",
            "net revenue",
            "total revenues",
            "consolidated revenue",
        ])

    if "revenue" in lowered and any(term in lowered for term in ("growth", "driver", "trend")):
        additions.extend([
            "revenue increased",
            "revenue decreased",
            "revenue growth",
            "driven by",
        ])

    if "gross margin" in lowered:
        additions.extend([
            "gross margin increased",
            "gross margin decreased",
            "cost of revenue",
        ])

    if "reportable segment" in lowered or "business segment" in lowered:
        additions.extend([
            "reportable segments",
            "operating segments",
            "business segments",
        ])

    if not additions:
        return query

    # Preserve order while removing duplicates.
    unique_additions = list(dict.fromkeys(additions))
    return f"{query} {' '.join(unique_additions)}"


def _merge_documents(primary: list, secondary: list, limit: int) -> list:
    """Interleave two ranked result lists while removing duplicate chunks."""
    merged = []
    seen: set[str] = set()

    max_len = max(len(primary), len(secondary))
    for i in range(max_len):
        for docs in (primary, secondary):
            if i >= len(docs):
                continue
            doc = docs[i]
            key = getattr(doc, "page_content", "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= limit:
                return merged

    return merged


def _search_with_query_expansion(vectorstore, query: str, k: int, filter_dict: dict) -> list:
    primary = vectorstore.similarity_search(query, k=k, filter=filter_dict)

    expanded_query = _expand_retrieval_query(query)
    if expanded_query == query:
        return primary

    expanded = vectorstore.similarity_search(
        expanded_query,
        k=k,
        filter=filter_dict,
    )
    return _merge_documents(primary, expanded, limit=k)


def get_stock_performance(ticker: str) -> dict:
    """
    Fetch latest 5-day price/volume data for a ticker.

    Failure modes are handled gracefully (returned as {"error": ...} rather
    than raised) so callers like app.agent.ask_agent and the LLM-driven
    agentic tool loop can synthesize an honest "stock data unavailable"
    sentence instead of crashing the whole request over a flaky market-data
    provider. This covers yfinance network errors, timeouts, and rate
    limiting, none of which were previously caught.
    """
    ticker = ticker.strip().upper()

    if ticker in INVALID_TICKERS:
        return {
            "error": f"Invalid ticker '{ticker}'.",
            "citation": None,
        }

    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception as exc:
        logger.warning(
            "stock_fetch_failed",
            extra={"ticker": ticker, "error": str(exc)},
        )
        return {
            "error": (
                f"Could not fetch stock data for '{ticker}' right now "
                "(the market data provider timed out or rate-limited the request)."
            ),
            "citation": None,
        }

    hist = hist.dropna(subset=["Close"])

    if hist.empty:
        return {
            "error": "No valid stock data available.",
            "citation": None,
        }

    latest = hist.iloc[-1]
    return {
        "data": {
            "ticker": ticker,
            "latest_price": round(float(latest["Close"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "volume": int(latest["Volume"]),
        },
        "citation": format_stock_citation(ticker),
    }


def query_financial_reports(
    query: str,
    company: str,
    fiscal_year: Optional[int] = None,
    section: Optional[str] = None,
    k: int = 8,
) -> dict:
    """
    Search local 10-K filings for a specific company.
    Optionally filter by fiscal year and section.
    """
    company_slug = normalize_company(company)
    section_slug = normalize_section(section)

    conditions = [{"company_slug": company_slug}]

    if fiscal_year is not None:
        conditions.append({"fiscal_year": fiscal_year})

    if section_slug is not None:
        conditions.append({"section": section_slug})

    filter_dict = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    try:
        vectorstore = get_vectorstore()
        docs = hybrid_retrieve(
            vectorstore,
            query=query,
            k=k,
            filter_dict=filter_dict,
            prefer_latest_year=(fiscal_year is None),
            financial_statement_mode=(section_slug == "financial_statements"),
        )

        # A section classifier can be correct while a section-filtered semantic
        # search still returns nothing because the indexed chunk metadata is
        # incomplete. Retry at company level before declaring the evidence
        # unavailable. Query expansion is also used on this fallback.
        if not docs and section_slug is not None:
            logger.info(
                "section_retrieval_fallback",
                extra={
                    "company_slug": company_slug,
                    "section_slug": section_slug,
                },
            )
            docs = hybrid_retrieve(
                vectorstore,
                query=query,
                k=k,
                filter_dict={"company_slug": company_slug},
                prefer_latest_year=(fiscal_year is None),
                financial_statement_mode=(section_slug == "financial_statements"),
            )
    except VectorStoreUnavailableError:
        raise
    except Exception as exc:
        logger.error(
            "similarity_search_failed",
            extra={"company_slug": company_slug, "error": str(exc)},
        )
        raise VectorStoreUnavailableError(str(exc)) from exc

    display = SLUG_TO_DISPLAY.get(company_slug, company_slug)

    if not docs:
        if fiscal_year is not None and section_slug is not None:
            return {
                "content": f"{display} FY{fiscal_year} {section_slug} content is not indexed in the database.",
                "citations": [],
            }
        if fiscal_year is not None:
            return {
                "content": f"{display} FY{fiscal_year} 10-K is not indexed in the database.",
                "citations": [],
            }
        if section_slug is not None:
            return {
                "content": f"{display} {section_slug} content is not indexed in the database.",
                "citations": [],
            }

        return {
            "content": f"{display} 10-K filings are not indexed in the database.",
            "citations": [],
        }

    cleaned_chunks = []
    citations = []

    for doc in docs:
        cleaned_chunks.append(clean_filing_text(doc.page_content))

        md = doc.metadata
        fy = md.get("fiscal_year", "unknown")
        sec = md.get("section", "full_filing")
        company_name = md.get("company", display)

        citation = format_filing_citation(company_name, fy, sec)
        if citation not in citations:
            citations.append(citation)

    return {
        "content": "\n\n".join(cleaned_chunks),
        "citations": citations,
    }