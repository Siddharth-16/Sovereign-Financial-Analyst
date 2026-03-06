import yfinance as yf
from langchain_community.tools import tool

@tool
def get_stock_performance(ticker: str):
    """Refers to real-time stock data to get the current price and change."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    return hist.tail(5).to_dict()

@tool
def query_financial_reports(query: str):
    """Searches the local vector database for specific details in 10-K filings."""

    return "Searching local database for: " + query