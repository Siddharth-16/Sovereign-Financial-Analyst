from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sec_api import QueryApi, RenderApi

load_dotenv()

API_KEY = os.getenv("SEC_API_KEY")
if not API_KEY:
    raise RuntimeError("SEC_API_KEY not found in environment or .env")

query_api = QueryApi(api_key=API_KEY)
render_api = RenderApi(api_key=API_KEY)

DATA_ROOT = Path("data/raw")

COMPANIES: dict[str, str] = {
    "NVDA": "nvidia",
    "MSFT": "microsoft",
    "AAPL": "apple",
    "AMZN": "amazon",
    "GOOG": "alphabet",
    "META": "meta",
    "AMD": "amd",
    "AVGO": "broadcom",
    "TSLA": "tesla",
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

TARGET_FISCAL_YEARS = {2023, 2024, 2025}
MAX_RESULTS = 12
DOWNLOAD_RETRIES = 3
SLEEP_SECONDS = 0.2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def extract_fiscal_year_from_text(text: str) -> tuple[int | None, str]:
    """
    Return (fiscal_year, source).
    Tries multiple patterns against visible filing text.
    """
    snippet = text[:10000]

    patterns = [
        r"For the fiscal year ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"For the year ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, snippet, flags=re.IGNORECASE)
        if match:
            date_text = match.group(1)
            year_match = re.search(r"(\d{4})$", date_text)
            if year_match:
                return int(year_match.group(1)), "header_text"

    return None, "none"


def fallback_fiscal_year_from_filed_at(filed_at: str) -> tuple[int, str]:
    return int(filed_at[:4]), "filed_at"


def safe_download_filing(filing_url: str) -> str | None:
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            return render_api.get_filing(filing_url)
        except Exception as exc:
            if attempt == DOWNLOAD_RETRIES:
                print(f"  Download failed after {DOWNLOAD_RETRIES} attempts: {exc}")
                return None
            wait = attempt * 1.5
            print(f"  Retry {attempt}/{DOWNLOAD_RETRIES} after error: {exc}")
            time.sleep(wait)
    return None


def find_recent_10k_filings(ticker: str, max_results: int = MAX_RESULTS) -> list[dict[str, Any]]:
    search_query = {
        "query": f'ticker:{ticker} AND formType:"10-K"',
        "from": "0",
        "size": str(max_results),
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    response = query_api.get_filings(search_query)
    return response.get("filings", [])


def choose_target_filings(filings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}

    for filing in filings:
        filing_url = filing.get("linkToFilingDetails") or filing.get("linkToHtml")
        filed_at = filing.get("filedAt", "1900-01-01")

        if not filing_url:
            continue

        html = safe_download_filing(filing_url)
        if not html:
            continue

        visible_text = html_to_text(html)
        fiscal_year, fy_source = extract_fiscal_year_from_text(visible_text)

        if fiscal_year is None:
            fiscal_year, fy_source = fallback_fiscal_year_from_filed_at(filed_at)

        if fiscal_year in TARGET_FISCAL_YEARS and fiscal_year not in selected:
            filing["_downloaded_content"] = html
            filing["_fiscal_year"] = fiscal_year
            filing["_fiscal_year_source"] = fy_source
            selected[fiscal_year] = filing

        if TARGET_FISCAL_YEARS.issubset(selected.keys()):
            break

        time.sleep(SLEEP_SECONDS)

    return [selected[fy] for fy in sorted(selected.keys(), reverse=True)]


def save_filing_html(company_slug: str, fiscal_year: int, html_text: str) -> Path:
    folder = DATA_ROOT / company_slug
    ensure_dir(folder)

    out_path = folder / f"{fiscal_year}_10k.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def save_metadata(
    company_slug: str,
    ticker: str,
    fiscal_year: int,
    filing: dict[str, Any],
) -> Path:
    folder = DATA_ROOT / company_slug
    ensure_dir(folder)

    metadata = {
        "ticker": ticker,
        "company": company_slug,
        "form_type": filing.get("formType"),
        "fiscal_year": fiscal_year,
        "fiscal_year_source": filing.get("_fiscal_year_source"),
        "filed_at": filing.get("filedAt"),
        "accession_no": filing.get("accessionNo"),
        "source_url": filing.get("linkToFilingDetails") or filing.get("linkToHtml"),
    }

    out_path = folder / f"{fiscal_year}_10k.meta.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_path


def download_company_10ks(ticker: str, company_slug: str) -> set[int]:
    print(f"\n=== {ticker} / {company_slug} ===")
    filings = find_recent_10k_filings(ticker)

    if not filings:
        print("  No 10-K filings found.")
        return set()

    selected = choose_target_filings(filings)

    if not selected:
        print("  No target fiscal years found.")
        return set()

    saved_years: set[int] = set()

    for filing in selected:
        fiscal_year = filing["_fiscal_year"]
        html_text = filing["_downloaded_content"]
        filed_at = filing.get("filedAt", "unknown")

        html_path = DATA_ROOT / company_slug / f"{fiscal_year}_10k.html"
        meta_path = DATA_ROOT / company_slug / f"{fiscal_year}_10k.meta.json"

        if html_path.exists() and meta_path.exists():
            print(f"  Skipping FY{fiscal_year} (already exists) -> {html_path}")
            saved_years.add(fiscal_year)
            continue

        out_path = save_filing_html(company_slug, fiscal_year, html_text)
        save_metadata(company_slug, ticker, fiscal_year, filing)

        print(
            f"  Saved FY{fiscal_year} ({filed_at}) "
            f"[source={filing['_fiscal_year_source']}] -> {out_path}"
        )
        saved_years.add(fiscal_year)

    return saved_years


def print_summary(summary: dict[str, set[int]]) -> None:
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)

    for company_slug, years in summary.items():
        years_sorted = sorted(years)
        missing = sorted(TARGET_FISCAL_YEARS - years)
        print(
            f"{company_slug:22} "
            f"downloaded={years_sorted if years_sorted else '[]'} "
            f"missing={missing if missing else '[]'}"
        )


def main() -> None:
    ensure_dir(DATA_ROOT)
    summary: dict[str, set[int]] = {}

    for ticker, company_slug in COMPANIES.items():
        try:
            summary[company_slug] = download_company_10ks(ticker, company_slug)
        except Exception as exc:
            print(f"  Failed for {ticker}: {exc}")
            summary[company_slug] = set()

    print_summary(summary)


if __name__ == "__main__":
    main()