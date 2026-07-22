"""
Domain-level exceptions for Sovereign Financial Analyst.

These exist so the API layer (api/main.py) can map a specific failure mode
to the right HTTP status code and a clean, honest error message instead of
leaking a raw stack trace or a generic 500 to the caller. Each one maps to
a failure mode called out in the Phase 2 roadmap:

  - OllamaUnavailableError    -> local LLM daemon isn't running / unreachable
  - VectorStoreUnavailableError -> Chroma / embeddings failed to load or query
  - StockDataUnavailableError -> yfinance timeout, rate limit, or bad ticker
"""


class SovereignError(Exception):
    """Base class for all domain-level errors in this app."""


class OllamaUnavailableError(SovereignError):
    """Raised when the local Ollama LLM cannot be reached or errors out
    after retries. Typically means `ollama serve` isn't running, the model
    hasn't been pulled, or the host is unreachable."""


class VectorStoreUnavailableError(SovereignError):
    """Raised when the Chroma vector store / embeddings model fails to
    initialize or a similarity search call errors out."""


class StockDataUnavailableError(SovereignError):
    """Raised when yfinance fails outright (network error, timeout). Note:
    an empty-but-successful response (e.g. bad ticker) is NOT an error --
    it's handled as a normal graceful-degradation case in app.tools."""
