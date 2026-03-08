from typing import Optional
import re

import yfinance as yf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import CHROMA_PATH, EMBED_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

INVALID_TICKERS = {"STOCK TICKER", "TICKER", "COMPANY", ""}

COMPANY_NAME_MAP = {
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

TICKER_TO_COMPANY = {
    "NVDA": "nvidia",
    "AAPL": "apple",
    "TSLA": "tesla",
    "MSFT": "microsoft",
    "AMZN": "amazon",
    "GOOG": "alphabet",
    "GOOGL": "alphabet",
    "META": "meta",
    "AMD": "amd",
    "AVGO": "broadcom",
    "CAT": "caterpillar",
    "BA": "boeing",
    "GE": "general_electric",
    "JPM": "jpmorgan_chase",
    "GS": "goldman_sachs",
    "V": "visa",
    "JNJ": "johnson_and_johnson",
    "LLY": "eli_lilly",
    "PFE": "pfizer",
    "XOM": "exxonmobil",
    "WMT": "walmart",
}

SLUG_TO_DISPLAY = {
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
    "general_electric": "General Electric",
    "jpmorgan_chase": "JPMorgan Chase",
    "goldman_sachs": "Goldman Sachs",
    "visa": "Visa",
    "johnson_and_johnson": "Johnson & Johnson",
    "eli_lilly": "Eli Lilly",
    "pfizer": "Pfizer",
    "exxonmobil": "ExxonMobil",
    "walmart": "Walmart",
}

SLUG_TO_TICKER = {
    "nvidia": "NVDA",
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOG",
    "meta": "META",
    "amd": "AMD",
    "broadcom": "AVGO",
    "caterpillar": "CAT",
    "boeing": "BA",
    "general_electric": "GE",
    "jpmorgan_chase": "JPM",
    "goldman_sachs": "GS",
    "visa": "V",
    "johnson_and_johnson": "JNJ",
    "eli_lilly": "LLY",
    "pfizer": "PFE",
    "exxonmobil": "XOM",
    "walmart": "WMT",
}

SECTION_NAME_MAP = {
    "business": "business",
    "risk_factors": "risk_factors",
    "risks": "risk_factors",
    "risk": "risk_factors",
    "mdna": "mdna",
    "mda": "mdna",
    "management_discussion": "mdna",
    "financial_statements": "financial_statements",
    "financials": "financial_statements",
}

SECTION_DISPLAY_MAP = {
    "business": "Business",
    "risk_factors": "Risk Factors",
    "mdna": "MD&A",
    "financial_statements": "Financial Statements",
    "full_filing": "Full Filing",
}


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


def normalize_company(company: Optional[str]) -> Optional[str]:
    if not company:
        return None

    company = company.strip()

    if company.upper() in TICKER_TO_COMPANY:
        return TICKER_TO_COMPANY[company.upper()]

    lowered = company.lower()
    if lowered in COMPANY_NAME_MAP:
        return COMPANY_NAME_MAP[lowered]

    return lowered


def format_filing_citation(company_name: str, fiscal_year: int | str, section: str) -> str:
    section_display = SECTION_DISPLAY_MAP.get(section, section.replace("_", " ").title())
    return f"{company_name} 10-K FY{fiscal_year} – {section_display}"


def format_stock_citation(ticker: str) -> str:
    return f"{ticker} market data – latest 5d window"


def get_stock_performance(ticker: str) -> dict:
    ticker = ticker.strip().upper()

    if ticker in INVALID_TICKERS:
        return {
            "error": f"Invalid ticker '{ticker}'.",
            "citation": None,
        }

    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        return {
            "error": f"No stock data found for '{ticker}'.",
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
    docs = db.similarity_search(query, k=4, filter=filter_dict)

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
    