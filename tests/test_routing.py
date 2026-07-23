"""
Coverage for the brittle keyword/regex logic in app.agent: entity/ticker
extraction and section inference. This is exactly the logic flagged in the
roadmap as most likely to silently break, since it's string matching, not
a model making a judgment call.
"""

from __future__ import annotations

from app.agent import (
    detect_mismatch,
    extract_companies,
    extract_company,
    find_company_aliases,
    find_tickers,
    infer_needs,
    infer_section,
    is_compare_query,
)


# --------------------------------------------------------------------- tickers


def test_find_tickers_basic():
    assert find_tickers("How is NVDA doing today?") == ["nvda"]


def test_find_tickers_word_boundary_no_false_positive():
    # "V" is Visa's ticker -- must not match inside "revenue", "everest", etc.
    assert "v" not in find_tickers("What was the revenue trend?")


def test_find_tickers_multiple():
    tickers = find_tickers("Compare AMD and NVDA risk factors")
    assert set(tickers) == {"amd", "nvda"}


def test_find_tickers_case_insensitive():
    assert find_tickers("nvda stock price") == ["nvda"]


# --------------------------------------------------------------------- aliases


def test_find_company_aliases_display_name():
    assert find_company_aliases("What are Nvidia's main risks?") == ["nvidia"]


def test_find_company_aliases_known_alias():
    # "google" is a registered alias for the "alphabet" slug
    assert find_company_aliases("How is google performing?") == ["alphabet"]


def test_find_company_aliases_ampersand_alias():
    assert find_company_aliases("J&J risk factors") == ["johnson_and_johnson"]


def test_find_company_aliases_none_found():
    assert find_company_aliases("What's the weather like?") == []


# --------------------------------------------------------------------- extract_company


def test_extract_company_prefers_ticker():
    assert extract_company("NVDA risk factors") == "nvidia"


def test_extract_company_falls_back_to_alias():
    assert extract_company("Nvidia risk factors") == "nvidia"


def test_extract_company_none_when_absent():
    assert extract_company("What are the risks in general?") is None


def test_extract_companies_dedupes_and_preserves_alias_first():
    # Nvidia named explicitly AND its ticker both appear -- should not
    # produce two separate entries for the same slug.
    companies = extract_companies("Compare Nvidia (NVDA) and AMD")
    assert companies.count("nvidia") == 1
    assert "amd" in companies


# --------------------------------------------------------------------- mismatch detection


def test_detect_mismatch_true_when_name_and_ticker_disagree():
    # Explicit company name "Nvidia" but ticker for AMD -- inconsistent
    assert detect_mismatch("Nvidia stock AMD ticker") is True


def test_detect_mismatch_false_when_consistent():
    assert detect_mismatch("Nvidia stock NVDA ticker") is False


def test_detect_mismatch_false_when_no_explicit_name():
    assert detect_mismatch("NVDA stock price") is False


# --------------------------------------------------------------------- compare queries


def test_is_compare_query_true_variants():
    assert is_compare_query("Compare Nvidia and AMD")
    assert is_compare_query("Nvidia vs AMD")
    assert is_compare_query("What's the difference between Nvidia and AMD?")


def test_is_compare_query_false():
    assert not is_compare_query("What are Nvidia's risks?")


# --------------------------------------------------------------------- infer_needs


def test_infer_needs_filing_only():
    needs_filings, needs_stock = infer_needs("What are Nvidia's risk factors?")
    assert needs_filings is True
    assert needs_stock is False


def test_infer_needs_stock_only():
    needs_filings, needs_stock = infer_needs("How is NVDA stock performing today?")
    assert needs_filings is False
    assert needs_stock is True


def test_infer_needs_both():
    needs_filings, needs_stock = infer_needs("Nvidia revenue trend and stock price")
    assert needs_filings is True
    assert needs_stock is True


# --------------------------------------------------------------------- infer_section


def test_infer_section_risk_factors_strong_signal():
    assert infer_section("What are the risk factors for Nvidia?") == "risk_factors"


def test_infer_section_mdna_strong_signal():
    assert infer_section("Summarize the MD&A section") == "mdna"


def test_infer_section_business_strong_signal():
    assert infer_section("What are Nvidia's business segments?") == "business"


def test_infer_section_financial_statements_strong_signal():
    assert infer_section("Show me the balance sheet") == "financial_statements"


def test_infer_section_weak_signal_revenue_falls_to_mdna():
    # "revenue trend" alone (no strong-signal phrase) should still route
    # to mdna via the weak-keyword fallback layer.
    assert infer_section("What's the revenue trend?") == "mdna"


def test_infer_section_none_when_no_signal():
    assert infer_section("Tell me a joke") is None


def test_infer_section_risk_precedence_over_business_keywords():
    # "risk" should win even if a business-ish word like "strategy" also
    # appears, since risk_keywords are checked first in the weak-signal
    # fallback chain.
    assert infer_section("What regulatory risk affects their strategy?") == "risk_factors"
