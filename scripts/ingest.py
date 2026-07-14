from __future__ import annotations
import os
import sys
import re
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import CHROMA_PATH, EMBED_MODEL
from app.companies import COMPANIES, SECTION_HEADINGS

RAW_DATA_ROOT = Path("data/raw")

COMPANY_METADATA = {
    slug: {"company": meta["display"], "ticker": meta["ticker"]}
    for slug, meta in COMPANIES.items()
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

def build_normalized_text_with_offsets(text: str) -> tuple[str, list[int]]:
    """Collapse the ENTIRE document to a single normalized string, while
    keeping a mapping from each character in the normalized string back to
    its offset in the original text.

    Large-cap 10-Ks are often produced by financial printers that render
    text with heavy inline markup -- sometimes one HTML tag per word (or
    even per line-break), purely for pixel-perfect kerning. BeautifulSoup's
    get_text(separator="\\n") then turns a single heading like
    "Item 1A. Risk Factors" into many separate lines: "Item", "1A.",
    "Risk", "Factors". No fixed-size line window can reliably re-join an
    arbitrary number of fragments.

    The robust fix is to stop treating line breaks as meaningful at all
    when looking for headings: collapse every run of whitespace (including
    newlines) to a single space, lowercase, and strip punctuation, exactly
    like normalize_heading() does for a single line -- but across the
    *whole document* at once. A heading fragmented across any number of
    lines/tags becomes indistinguishable from one written on a single
    line, so exact-substring matching becomes reliable again. The
    character-offset map lets us translate a match's position in the
    normalized string back to a real slice of the original text.
    """
    norm_chars: list[str] = []
    offsets: list[int] = []
    prev_was_space = True  # collapse leading whitespace too

    for i, ch in enumerate(text):
        if ch.isalnum():
            out_ch = ch.lower()
            is_space = False
        else:
            out_ch = " "
            is_space = True

        if is_space:
            if prev_was_space:
                continue
            prev_was_space = True
        else:
            prev_was_space = False

        norm_chars.append(out_ch)
        offsets.append(i)

    return "".join(norm_chars), offsets


def find_heading_matches(text: str) -> dict[str, list[int]]:
    """Section name -> sorted list of ORIGINAL-text character offsets where
    that heading occurs, found via whole-document normalized substring
    search (see build_normalized_text_with_offsets)."""
    normalized, offsets = build_normalized_text_with_offsets(text)

    matches: dict[str, list[int]] = {name: [] for name in SECTION_HEADINGS}

    for section_name, heading_variants in SECTION_HEADINGS.items():
        for variant in heading_variants:
            start = 0
            while True:
                idx = normalized.find(variant, start)
                if idx == -1:
                    break
                matches[section_name].append(offsets[idx])
                start = idx + 1  # allow overlapping/adjacent matches

    return {name: positions for name, positions in matches.items() if positions}


def split_into_sections(text: str) -> dict[str, str]:
    matches = find_heading_matches(text)

    if not matches:
        return {"full_filing": text}

    # For each section, take the LAST matching offset as the real heading.
    # In a standard 10-K, the Table of Contents lists every Item heading
    # before the real body, so earlier matches are TOC/cross-reference
    # noise and the final occurrence is the actual section start.
    heading_positions = [
        (section_name, max(positions)) for section_name, positions in matches.items()
    ]
    heading_positions.sort(key=lambda pair: pair[1])

    sections: dict[str, str] = {}

    for idx, (section_name, start_offset) in enumerate(heading_positions):
        end_offset = heading_positions[idx + 1][1] if idx + 1 < len(heading_positions) else len(text)
        section_text = text[start_offset:end_offset].strip()

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