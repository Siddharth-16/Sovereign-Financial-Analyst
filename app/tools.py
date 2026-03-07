import yfinance as yf
from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_PATH, EMBED_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)

INVALID_TICKERS = {"STOCK TICKER", "TICKER", "COMPANY", ""}


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
def query_financial_reports(query: str, company: str = "Nvidia"):
    """Search private financial reports for a specific company."""
    docs = db.similarity_search(query, k=3, filter={"company": company})

    if not docs:
        return f"No relevant {company} 10-K chunks found. The filing may not be in the database."

    return "\n\n".join(doc.page_content for doc in docs)