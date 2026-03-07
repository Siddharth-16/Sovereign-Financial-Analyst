import yfinance as yf
from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load once at startup
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)

@tool
def get_stock_performance(ticker: str):
    """Get recent stock performance for a real company ticker like AAPL, NVDA, or TSLA."""
    ticker = ticker.strip().upper()

    if ticker in {"STOCK TICKER", "TICKER", "COMPANY", ""}:
        return "Error: invalid ticker. Use a real ticker symbol like AAPL, NVDA, or TSLA."

    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")

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
    """Searches private financial reports. Pass the company name explicitly."""
    docs = db.similarity_search(
        query,
        k=3,
        filter={"company": company}
    )
    if not docs:
        return f"No relevant {company} 10-K chunks found. The filing may not be in the database."
    return "\n\n".join(d.page_content for d in docs)