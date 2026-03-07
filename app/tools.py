from typing import Optional
import yfinance as yf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re
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


def get_stock_performance(ticker: str) -> dict:
    ticker = ticker.strip().upper()

    if ticker in INVALID_TICKERS:
        return {"error": f"Invalid ticker '{ticker}'."}

    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        return {"error": f"No stock data found for '{ticker}'."}

    latest = hist.iloc[-1]
    return {
        "ticker": ticker,
        "latest_price": round(float(latest["Close"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
    }


def query_financial_reports(
    query: str,
    company: str,
    fiscal_year: Optional[int] = None,
) -> str:
    """
    Search local 10-K filings for a specific company.
    Optionally filter by fiscal year if the user explicitly requests one.
    """
    company_slug = normalize_company(company)

    if fiscal_year is None:
        filter_dict = {"company_slug": company_slug}
    else:
        filter_dict = {
            "$and": [
                {"company_slug": company_slug},
                {"fiscal_year": fiscal_year},
            ]
        }

    docs = db.similarity_search(query, k=4, filter=filter_dict)

    if not docs:
        display = SLUG_TO_DISPLAY.get(company_slug, company_slug)
        if fiscal_year is not None:
            return f"{display} FY{fiscal_year} 10-K is not indexed in the database."
        return f"{display} 10-K filings are not indexed in the database."

    return "\n\n".join(clean_filing_text(doc.page_content) for doc in docs)
    