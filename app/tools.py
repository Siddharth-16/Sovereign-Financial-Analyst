#tools.py
import yfinance as yf
from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Optional

from app.config import CHROMA_PATH, EMBED_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)

INVALID_TICKERS = {"STOCK TICKER", "TICKER", "COMPANY", ""}

COMPANY_NAME_MAP = {
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

TICKER_TO_COMPANY = {
    "NVDA": "Nvidia",
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOG": "Alphabet",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "CAT": "Caterpillar",
    "BA": "Boeing",
    "GE": "General Electric",
    "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "LLY": "Eli Lilly",
    "PFE": "Pfizer",
    "XOM": "ExxonMobil",
    "WMT": "Walmart",
}


def normalize_company(company: str | None) -> str | None:
    if not company:
        return None

    company = company.strip()

    # ticker input
    upper = company.upper()
    if upper in TICKER_TO_COMPANY:
        return TICKER_TO_COMPANY[upper]

    # company-name input
    lowered = company.lower()
    if lowered in COMPANY_NAME_MAP:
        return COMPANY_NAME_MAP[lowered]

    return company


@tool
def get_stock_performance(ticker: str):
    """Get recent stock performance for a real company ticker like AAPL, NVDA, or TSLA."""
    ticker = ticker.strip().upper()

    if ticker in INVALID_TICKERS:
        return "Error: invalid ticker. Use a real ticker symbol like AAPL, NVDA, or TSLA."

    hist = yf.Ticker(ticker).history(period="5d")

    if hist.empty:
        return f"Error: no stock data found for ticker '{ticker}'."

    latest = hist.iloc[-1]

    return {
        "ticker": ticker,
        "latest_price": round(float(latest["Close"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
    }

@tool
def query_financial_reports(
    query: str,
    company: str,
    fiscal_year: Optional[int] = None,
):
    """
    Search private 10-K filings for a specific company.
    
    Only pass fiscal_year if the user explicitly mentions a specific year.
    If no year is mentioned, leave fiscal_year as None to search all available filings.
    """
    normalized_company = normalize_company(company)

    # Single filter — no $and needed
    if fiscal_year is None:
        filter_dict = {"company": {"$eq": normalized_company}}
    else:
        # Multiple filters MUST use $and in ChromaDB
        filter_dict = {
            "$and": [
                {"company": {"$eq": normalized_company}},
                {"fiscal_year": {"$eq": fiscal_year}},
            ]
        }

    docs = db.similarity_search(query, k=6, filter=filter_dict)

    if not docs:
        if fiscal_year:
            return f"{normalized_company} FY{fiscal_year} 10-K is not indexed."
        return f"{normalized_company} 10-K filings are not indexed."

    return "\n\n".join(doc.page_content for doc in docs)