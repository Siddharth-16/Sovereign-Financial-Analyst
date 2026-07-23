from __future__ import annotations
import pytest
from app import tools


# --------------------------------------------------------------------- normalize_company


def test_normalize_company_by_ticker():
    assert tools.normalize_company("NVDA") == "nvidia"


def test_normalize_company_by_ticker_lowercase_input():
    assert tools.normalize_company("nvda") == "nvidia"


def test_normalize_company_by_display_name():
    assert tools.normalize_company("Nvidia") == "nvidia"


def test_normalize_company_by_alias():
    assert tools.normalize_company("google") == "alphabet"


def test_normalize_company_unknown_passthrough():
    # Unknown input is lowercased and passed through rather than raising --
    # query_financial_reports then reports "not indexed" instead of erroring.
    assert tools.normalize_company("Unknown Corp") == "unknown corp"


def test_normalize_company_none():
    assert tools.normalize_company(None) is None


def test_normalize_company_empty_string():
    assert tools.normalize_company("") is None


# --------------------------------------------------------------------- normalize_section


def test_normalize_section_canonical():
    assert tools.normalize_section("risk_factors") == "risk_factors"


def test_normalize_section_alias():
    assert tools.normalize_section("risks") == "risk_factors"
    assert tools.normalize_section("mda") == "mdna"
    assert tools.normalize_section("financials") == "financial_statements"


def test_normalize_section_none():
    assert tools.normalize_section(None) is None


# --------------------------------------------------------------------- clean_filing_text


def test_clean_filing_text_strips_urls():
    dirty = "See https://example.com/filing for details"
    assert "https://" not in tools.clean_filing_text(dirty)


def test_clean_filing_text_strips_toc_boilerplate():
    dirty = "Table of Contents\nItem 1A. Risk Factors\nReal content here"
    cleaned = tools.clean_filing_text(dirty)
    assert "Table of Contents" not in cleaned
    assert "Real content here" in cleaned


def test_clean_filing_text_collapses_whitespace():
    dirty = "Too    many\n\n\nspaces"
    assert tools.clean_filing_text(dirty) == "Too many spaces"


# --------------------------------------------------------------------- citations


def test_format_filing_citation():
    citation = tools.format_filing_citation("Nvidia", 2024, "risk_factors")
    assert citation == "Nvidia 10-K FY2024 – Risk Factors"


def test_format_filing_citation_unknown_section_title_cased():
    # Not present in SECTION_DISPLAY_MAP -- exercises the .title() fallback.
    citation = tools.format_filing_citation("Nvidia", 2024, "made_up_section")
    assert citation == "Nvidia 10-K FY2024 – Made Up Section"


def test_format_stock_citation():
    # format_stock_citation itself doesn't uppercase -- get_stock_performance
    # already normalizes the ticker to uppercase before calling it.
    assert tools.format_stock_citation("NVDA") == "NVDA market data – latest 5d window"


# --------------------------------------------------------------------- get_stock_performance


def test_get_stock_performance_invalid_ticker():
    result = tools.get_stock_performance("")
    assert "error" in result
    assert result["citation"] is None


def test_get_stock_performance_yfinance_raises_returns_graceful_error(monkeypatch):
    """Phase 2 fix: a yfinance timeout/rate-limit exception must NOT
    propagate and crash the request -- it should degrade to the same
    {"error": ...} shape the function already uses for other failure modes."""

    class _ExplodingTicker:
        def __init__(self, *_a, **_kw):
            pass

        def history(self, *_a, **_kw):
            raise ConnectionError("simulated yfinance timeout")

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", _ExplodingTicker)

    result = tools.get_stock_performance("NVDA")
    assert "error" in result
    assert "timed out" in result["error"] or "rate-limited" in result["error"]
    assert result["citation"] is None


def test_get_stock_performance_empty_history_returns_graceful_error(monkeypatch):
    class _EmptyFrame:
        empty = True

        def dropna(self, *_a, **_kw):
            return self

    class _EmptyTicker:
        def __init__(self, *_a, **_kw):
            pass

        def history(self, *_a, **_kw):
            return _EmptyFrame()

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", _EmptyTicker)

    result = tools.get_stock_performance("NVDA")
    assert result["error"] == "No valid stock data available."


# --------------------------------------------------------------------- lazy vector store


def test_get_vectorstore_wraps_init_failure(monkeypatch):
    """Phase 2 fix: vector store init used to happen at import time, so a
    bad CHROMA_PATH would crash the whole process. Now it's lazy and
    failures surface as a typed, catchable exception."""
    tools.reset_vectorstore_cache()

    class _ExplodingChroma:
        def __init__(self, *_a, **_kw):
            raise RuntimeError("simulated corrupt chroma_db directory")

    monkeypatch.setattr(tools, "Chroma", _ExplodingChroma)

    with pytest.raises(tools.VectorStoreUnavailableError):
        tools.get_vectorstore()

    tools.reset_vectorstore_cache()
