#ingest.py
from __future__ import annotations

import os
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHROMA_PATH, EMBED_MODEL

RAW_DATA_ROOT = Path("data/raw")

# Map folder slug -> display company name / ticker
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


if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN not set. HuggingFace downloads may be rate limited.")


def html_to_text(html: str) -> str:
    """Convert SEC filing HTML into clean visible text."""
    soup = BeautifulSoup(html, "html.parser")

    # remove unwanted tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def parse_filing_metadata(file_path: Path) -> dict:
    """
    Example path:
    data/raw/nvidia/2025_10k.html
    """
    company_slug = file_path.parent.name
    filename = file_path.stem  # e.g. 2025_10k

    fiscal_year_str, filing_type = filename.split("_", 1)
    fiscal_year = int(fiscal_year_str)
    filing_type = filing_type.upper()  # 10K -> 10K / or keep 10-K below

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


def load_html_filing(file_path: Path) -> Document:
    """Load one HTML filing and return a LangChain Document."""
    metadata = parse_filing_metadata(file_path)

    html = file_path.read_text(encoding="utf-8", errors="ignore")
    text = html_to_text(html)

    return Document(page_content=text, metadata=metadata)


def collect_documents(data_root: Path) -> list[Document]:
    """Walk data/raw and load all *_10k.html filings."""
    docs: list[Document] = []

    for file_path in sorted(data_root.glob("*/*_10k.html")):
        try:
            doc = load_html_filing(file_path)
            docs.append(doc)
            print(f"Loaded: {file_path}")
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

    print(f"Loaded {len(documents)} filings.")

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