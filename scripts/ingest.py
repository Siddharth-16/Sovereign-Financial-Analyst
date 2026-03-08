from __future__ import annotations

import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHROMA_PATH, EMBED_MODEL

RAW_DATA_ROOT = Path("data/raw")

COMPANY_METADATA = {
    "nvidia": {"company": "Nvidia", "ticker": "NVDA"},
    "microsoft": {"company": "Microsoft", "ticker": "MSFT"},
    "apple": {"company": "Apple", "ticker": "AAPL"},
    "amazon": {"company": "Amazon", "ticker": "AMZN"},
    "alphabet": {"company": "Alphabet", "ticker": "GOOG"},
    "meta": {"company": "Meta", "ticker": "META"},
    "amd": {"company": "AMD", "ticker": "AMD"},
    "broadcom": {"company": "Broadcom", "ticker": "AVGO"},
    "tesla": {"company": "Tesla", "ticker": "TSLA"},
    "caterpillar": {"company": "Caterpillar", "ticker": "CAT"},
    "boeing": {"company": "Boeing", "ticker": "BA"},
    "general_electric": {"company": "General Electric", "ticker": "GE"},
    "jpmorgan_chase": {"company": "JPMorgan Chase", "ticker": "JPM"},
    "goldman_sachs": {"company": "Goldman Sachs", "ticker": "GS"},
    "visa": {"company": "Visa", "ticker": "V"},
    "johnson_and_johnson": {"company": "Johnson & Johnson", "ticker": "JNJ"},
    "eli_lilly": {"company": "Eli Lilly", "ticker": "LLY"},
    "pfizer": {"company": "Pfizer", "ticker": "PFE"},
    "exxonmobil": {"company": "ExxonMobil", "ticker": "XOM"},
    "walmart": {"company": "Walmart", "ticker": "WMT"},
}

SECTION_HEADINGS = {
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

if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN not set. HuggingFace downloads may be rate limited.")


def html_to_text(html: str) -> str:
    """Convert SEC filing HTML into clean visible text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)

def normalize_heading(line: str) -> str:
    line = line.strip().lower()
    line = re.sub(r"[^\w\s]", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line

def parse_filing_metadata(file_path: Path) -> dict:
    company_slug = file_path.parent.name
    filename = file_path.stem  # e.g. 2025_10k

    fiscal_year_str, _ = filename.split("_", 1)
    fiscal_year = int(fiscal_year_str)

    if company_slug not in COMPANY_METADATA:
        raise ValueError(f"Unknown company folder: {company_slug}")

    base = COMPANY_METADATA[company_slug]

    return {
        "company_slug": company_slug,
        "company": base["company"],
        "ticker": base["ticker"],
        "fiscal_year": fiscal_year,
        "filing": "10-K",
        "source_path": str(file_path),
    }

def split_into_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()

    heading_positions: list[tuple[str, int]] = []

    for i, line in enumerate(lines):
        normalized = normalize_heading(line)

        for section_name, heading_variants in SECTION_HEADINGS.items():
            if normalized in heading_variants:
                heading_positions.append((section_name, i))
                break

    # remove duplicates, keep first occurrence of each section
    seen = set()
    deduped: list[tuple[str, int]] = []
    for section_name, line_idx in heading_positions:
        if section_name not in seen:
            deduped.append((section_name, line_idx))
            seen.add(section_name)

    heading_positions = deduped

    if not heading_positions:
        return {"full_filing": text}

    sections: dict[str, str] = {}

    for idx, (section_name, start_line) in enumerate(heading_positions):
        end_line = heading_positions[idx + 1][1] if idx + 1 < len(heading_positions) else len(lines)
        section_text = "\n".join(lines[start_line:end_line]).strip()

        if len(section_text) > 1000:
            sections[section_name] = section_text

    if not sections:
        return {"full_filing": text}

    return sections

def load_html_filing(file_path: Path) -> list[Document]:
    """Load one HTML filing and return section-level LangChain Documents."""
    base_metadata = parse_filing_metadata(file_path)

    html = file_path.read_text(encoding="utf-8", errors="ignore")
    text = html_to_text(html)
    sections = split_into_sections(text)

    print(f"{file_path.name} sections found: {list(sections.keys())}")

    docs: list[Document] = []
    for section_name, section_text in sections.items():
        metadata = {**base_metadata, "section": section_name}
        docs.append(Document(page_content=section_text, metadata=metadata))

    return docs


def collect_documents(data_root: Path) -> list[Document]:
    docs: list[Document] = []

    for file_path in sorted(data_root.glob("*/*_10k.html")):
        try:
            filing_docs = load_html_filing(file_path)
            docs.extend(filing_docs)
            print(f"Loaded: {file_path} ({len(filing_docs)} sections)")
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


def ingest_all() -> None:
    print(f"Scanning filings in {RAW_DATA_ROOT}...")
    documents = collect_documents(RAW_DATA_ROOT)

    if not documents:
        print("No filings found.")
        return

    print(f"Loaded {len(documents)} section documents.")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    BATCH_SIZE = 500

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Inserted {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    print(f"Ingestion complete — stored {len(chunks)} chunks in {CHROMA_PATH}")


if __name__ == "__main__":
    ingest_all()