from __future__ import annotations
import re

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

SLUG_TO_DISPLAY: dict[str, str] = {
    slug: meta["display"] for slug, meta in COMPANIES.items()
}
SLUG_TO_TICKER: dict[str, str] = {
    slug: meta["ticker"] for slug, meta in COMPANIES.items()
}
TICKER_TO_SLUG: dict[str, str] = {}
for _slug, _meta in COMPANIES.items():
    TICKER_TO_SLUG[_meta["ticker"].upper()] = _slug
    for _extra_ticker in _meta.get("extra_tickers", []):
        TICKER_TO_SLUG[_extra_ticker.upper()] = _slug
del _slug, _meta, _extra_ticker

COMPANY_NAME_MAP: dict[str, str] = {}
for _slug, _meta in COMPANIES.items():
    _display_lower = _meta["display"].lower()
    COMPANY_NAME_MAP[_display_lower] = _slug
    COMPANY_NAME_MAP[_slug.replace("_", " ")] = _slug
    for _alias in _meta["aliases"]:
        COMPANY_NAME_MAP[_alias.lower()] = _slug
del _slug, _meta, _display_lower, _alias

INDEXED_COMPANIES: set[str] = set(COMPANIES.keys())
SUPPORTED_COMPANIES: set[str] = INDEXED_COMPANIES

SECTION_PATTERNS_ITEM: dict[str, re.Pattern] = {
    "business":              re.compile(r"^item\s*1\.?\s*[-\u2014.]?\s*business\b", re.I),
    "risk_factors":          re.compile(r"^item\s*1a\.?\s*[-\u2014.]?\s*risk\s*factors\b", re.I),
    "mdna":                  re.compile(r"^item\s*7\.?\s*[-\u2014.]?\s*management.{0,3}s?\s*discussion", re.I),
    "financial_statements":  re.compile(r"^item\s*8\.?\s*[-\u2014.]?\s*financial\s*statements", re.I),
}

SECTION_PATTERNS_NARRATIVE: dict[str, re.Pattern] = {
    "business":              re.compile(r"^(about\s+[a-z .,&]+|business)\s*$", re.I),
    "risk_factors":          re.compile(r"^risk\s*factors\s*$", re.I),
    "mdna":                  re.compile(r"^management.{0,3}s?\s*discussion\s*and\s*analysis\b", re.I),
    "financial_statements":  re.compile(
        r"^(audited\s+)?(consolidated\s+)?financial\s*statements(\s+and\s+(supplementary\s+)?notes)?\s*$",
        re.I,
    ),
}

SECTION_HEADINGS: dict[str, list[str]] = {
    "business": ["item 1 business"],
    "risk_factors": ["item 1a risk factors"],
    "mdna": ["item 7 management s discussion and analysis of financial condition and results of operations"],
    "financial_statements": ["item 8 financial statements and supplementary data"],
}

REQUIRED_SECTIONS: set[str] = set(SECTION_PATTERNS_ITEM.keys())

SECTION_DISPLAY_MAP: dict[str, str] = {
    "business": "Business",
    "risk_factors": "Risk Factors",
    "mdna": "MD&A",
    "financial_statements": "Financial Statements",
    "full_filing": "Full Filing",
    "amendment_supplement": "Amendment Supplement (non-core Items)",
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