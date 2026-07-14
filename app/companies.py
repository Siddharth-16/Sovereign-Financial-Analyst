from __future__ import annotations

''' slug -> {display name, primary ticker, extra name aliases a user might type}
The slug is what's stored in Chroma metadata (company_slug) and used as the
folder name under data/raw/.'''

COMPANIES: dict[str, dict] = {
    "nvidia":              {"display": "Nvidia",           "ticker": "NVDA",  "aliases": [], "extra_tickers": []},
    "apple":                {"display": "Apple",            "ticker": "AAPL",  "aliases": [], "extra_tickers": []},
    "tesla":                {"display": "Tesla",            "ticker": "TSLA",  "aliases": [], "extra_tickers": []},
    "microsoft":            {"display": "Microsoft",        "ticker": "MSFT",  "aliases": [], "extra_tickers": []},
    "amazon":               {"display": "Amazon",           "ticker": "AMZN",  "aliases": [], "extra_tickers": []},
    "alphabet":             {"display": "Alphabet",         "ticker": "GOOG",  "aliases": ["google", "googl"], "extra_tickers": ["GOOGL"]},
    "meta":                 {"display": "Meta",             "ticker": "META",  "aliases": [], "extra_tickers": []},
    "amd":                  {"display": "AMD",              "ticker": "AMD",   "aliases": [], "extra_tickers": []},
    "broadcom":             {"display": "Broadcom",         "ticker": "AVGO",  "aliases": [], "extra_tickers": []},
    "caterpillar":          {"display": "Caterpillar",      "ticker": "CAT",   "aliases": [], "extra_tickers": []},
    "boeing":               {"display": "Boeing",           "ticker": "BA",    "aliases": [], "extra_tickers": []},
    "general_electric":     {"display": "General Electric", "ticker": "GE",    "aliases": [], "extra_tickers": []},
    "jpmorgan_chase":       {"display": "JPMorgan Chase",   "ticker": "JPM",   "aliases": ["jpmorgan"], "extra_tickers": []},
    "goldman_sachs":        {"display": "Goldman Sachs",    "ticker": "GS",    "aliases": [], "extra_tickers": []},
    "visa":                 {"display": "Visa",             "ticker": "V",     "aliases": [], "extra_tickers": []},
    "johnson_and_johnson":  {"display": "Johnson & Johnson", "ticker": "JNJ",  "aliases": ["johnson and johnson", "j&j"], "extra_tickers": []},
    "eli_lilly":            {"display": "Eli Lilly",        "ticker": "LLY",   "aliases": [], "extra_tickers": []},
    "pfizer":               {"display": "Pfizer",           "ticker": "PFE",   "aliases": [], "extra_tickers": []},
    "exxonmobil":           {"display": "ExxonMobil",       "ticker": "XOM",   "aliases": ["exxon mobil", "exxon"], "extra_tickers": []},
    "walmart":              {"display": "Walmart",          "ticker": "WMT",   "aliases": [], "extra_tickers": []},
}

TARGET_FISCAL_YEARS: set[int] = {2023, 2024, 2025}

# slug -> display name, e.g. "jpmorgan_chase" -> "JPMorgan Chase"
SLUG_TO_DISPLAY: dict[str, str] = {
    slug: meta["display"] for slug, meta in COMPANIES.items()
}

# slug -> primary ticker, e.g. "nvidia" -> "NVDA"
SLUG_TO_TICKER: dict[str, str] = {
    slug: meta["ticker"] for slug, meta in COMPANIES.items()
}

# UPPERCASE ticker -> slug, e.g. "NVDA" -> "nvidia", "GOOGL" -> "alphabet"
TICKER_TO_SLUG: dict[str, str] = {}
for _slug, _meta in COMPANIES.items():
    TICKER_TO_SLUG[_meta["ticker"].upper()] = _slug
    for _extra_ticker in _meta.get("extra_tickers", []):
        TICKER_TO_SLUG[_extra_ticker.upper()] = _slug
del _slug, _meta, _extra_ticker

# lowercase name/alias -> slug, e.g. "jpmorgan chase" -> "jpmorgan_chase",
# "google" -> "alphabet". Built from the display name plus any extra aliases.
COMPANY_NAME_MAP: dict[str, str] = {}
for _slug, _meta in COMPANIES.items():
    _display_lower = _meta["display"].lower()
    COMPANY_NAME_MAP[_display_lower] = _slug
    COMPANY_NAME_MAP[_slug.replace("_", " ")] = _slug
    for _alias in _meta["aliases"]:
        COMPANY_NAME_MAP[_alias.lower()] = _slug
del _slug, _meta, _display_lower, _alias  

# The set of companies actually supported/indexed by the system.
INDEXED_COMPANIES: set[str] = set(COMPANIES.keys())
SUPPORTED_COMPANIES: set[str] = INDEXED_COMPANIES


# Section registry (used by ingest.py's heading matcher and the coverage check)
SECTION_HEADINGS: dict[str, list[str]] = {
    "business": [
        "item 1 business",
    ],
    "risk_factors": [
        "item 1a risk factors",
    ],
    "mdna": [
        "item 7 management s discussion and analysis of financial condition and results of operations",
        "item 7 management discussion and analysis of financial condition and results of operations",
        "item 7 management s discussion and analysis",
        "item 7 management discussion and analysis",
    ],
    "financial_statements": [
        "item 8 financial statements and supplementary data",
        "item 8 financial statements",
    ],
}

REQUIRED_SECTIONS: set[str] = set(SECTION_HEADINGS.keys())

SECTION_DISPLAY_MAP: dict[str, str] = {
    "business": "Business",
    "risk_factors": "Risk Factors",
    "mdna": "MD&A",
    "financial_statements": "Financial Statements",
    "full_filing": "Full Filing",
}

SECTION_NAME_MAP: dict[str, str] = {
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
