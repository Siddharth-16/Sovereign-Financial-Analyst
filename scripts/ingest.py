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
from app.companies import (
    COMPANIES,
    SECTION_PATTERNS_ITEM,
    SECTION_PATTERNS_NARRATIVE,
)

RAW_DATA_ROOT = Path("data/raw")

COMPANY_METADATA = {
    slug: {"company": meta["display"], "ticker": meta["ticker"]}
    for slug, meta in COMPANIES.items()
}

if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN not set. HuggingFace downloads may be rate limited.")

MIN_SECTION_CHARS = 1000
_MARKER_PREFIX = "\ue000SECTION_MARKER_"  # private-use-area char, won't collide with real content


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

def html_to_text(html: str) -> str:
    """Convert SEC filing HTML into clean visible text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def normalize_line(line: str) -> str:
    line = line.strip().lower()
    line = re.sub(r"[^\w\s]", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def parse_filing_metadata(file_path: Path) -> dict:
    company_slug = file_path.parent.name
    filename = file_path.stem  # e.g. 2025_10k or 2025_10ka

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

_AMENDMENT_MARKERS = re.compile(
    r"form\s*10-k\s*/\s*a|amendment\s+no\.?\s*\d+\s+to\s+(annual\s+report\s+on\s+)?form\s*10-k",
    re.I,
)
_AMENDMENT_SCOPE_NOTE = re.compile(
    r"does\s+not\s+otherwise\s+(change|update|amend)|"
    r"amend\s+part\s+iii|solely\s+to\b.{0,200}part\s+iii",
    re.I | re.S,
)


def detect_amendment(text: str) -> tuple[bool, str | None]:
    """Return (is_amendment, scope_note). scope_note is a short human-readable
    reason, used for logging / metadata, not for control flow beyond the
    boolean."""
    head = text[:6000]
    if not _AMENDMENT_MARKERS.search(head):
        return False, None
    scope_match = _AMENDMENT_SCOPE_NOTE.search(text[:20000])
    scope = "Part III/IV only (per filing's own explanatory note)" if scope_match else "amendment, scope unconfirmed"
    return True, scope



# Strategy 1: TOC-anchor-based section splitting.
#
# Nearly every SEC-filer-produced 10-K has a hyperlinked Table of Contents
# where each row links (via #anchor) to a bookmark placed at the real start
# of that section. This works even for filers (GE confirmed) whose actual
# body headings never literally say "Item 1A" -- because we're following the
# filer's own navigation structure, not guessing at heading text.

def _row_text_for_link(a_tag) -> str:
    """Text of the table row (or nearby container) that holds this TOC link,
    so 'Item 1A.' and 'Risk Factors.' sitting in separate cells of the same
    row are seen together for classification."""
    row = a_tag.find_parent("tr")
    if row is not None:
        return row.get_text(" ", strip=True)
    # fall back to the link's own text plus following sibling text
    parent = a_tag.find_parent(["p", "div", "li"]) or a_tag
    return parent.get_text(" ", strip=True)


def classify_toc_text(text: str) -> str | None:
    norm = normalize_line(text)
    for section, pattern in SECTION_PATTERNS_ITEM.items():
        if pattern.search(norm):
            return section
    return None


def build_toc_anchor_map(soup: BeautifulSoup) -> dict[str, str]:
    """canonical section name -> anchor id (the '#...' target) found via the
    filing's own hyperlinked TOC. Only the FIRST TOC link classified for a
    given section is kept -- TOC rows are read top-to-bottom same as the
    document, and sub-rows (e.g. 'Overview' under Item 1) won't match the
    section patterns at all, so this is safe."""
    anchor_map: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("#") or len(href) < 2:
            continue
        row_text = _row_text_for_link(a)
        section = classify_toc_text(row_text)
        if section and section not in anchor_map:
            anchor_map[section] = href[1:]
    return anchor_map


def find_anchor_targets(soup: BeautifulSoup, anchor_ids: set[str]) -> dict[str, object]:
    """anchor id -> the BeautifulSoup tag it points to (the bookmark placed
    at the real heading, NOT the TOC link itself)."""
    targets: dict[str, object] = {}
    remaining = set(anchor_ids)
    if not remaining:
        return targets
    for tag in soup.find_all(True):
        tid = tag.get("id")
        tname = tag.get("name")
        for candidate in (tid, tname):
            if candidate and candidate in remaining:
                targets[candidate] = tag
                remaining.discard(candidate)
        if not remaining:
            break
    return targets


def split_by_anchors(html: str) -> dict[str, str] | None:
    """Try TOC-anchor-based splitting. Returns None if fewer than 2 of the 4
    canonical sections could be anchor-resolved (not worth trusting a
    partial anchor map over the regex fallback)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    anchor_map = build_toc_anchor_map(soup)
    if len(anchor_map) < 2:
        return None

    targets = find_anchor_targets(soup, set(anchor_map.values()))
    section_to_target = {
        section: targets[aid] for section, aid in anchor_map.items() if aid in targets
    }
    if len(section_to_target) < 2:
        return None

    for section, tag in section_to_target.items():
        marker = soup.new_string(f"{_MARKER_PREFIX}{section}\n")
        tag.insert_before(marker)

    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    flat = "\n".join(lines)

    parts = re.split(rf"({re.escape(_MARKER_PREFIX)}\w+)", flat)
    sections: dict[str, str] = {}
    current_section = None
    buffer: list[str] = []
    for part in parts:
        m = re.fullmatch(rf"{re.escape(_MARKER_PREFIX)}(\w+)", part)
        if m:
            if current_section is not None:
                sections[current_section] = "\n".join(buffer).strip()
            current_section = m.group(1)
            buffer = []
        else:
            buffer.append(part)
    if current_section is not None:
        sections[current_section] = "\n".join(buffer).strip()

    sections = {k: v for k, v in sections.items() if len(v) > MIN_SECTION_CHARS}
    return sections or None

# Strategy 2: regex heading fallback (used when anchors are missing/broken).
#
# Key fix vs. the old "take the LAST matching offset" rule: that heuristic
# assumed the only duplicate would be an earlier TOC entry, but running
# headers, cross-reference indices, and exhibit lists can just as easily put
# a duplicate AFTER the real section (verified: this is what actually
# quietly corrupted section boundaries in some of the large-filer 10-Ks --
# a late duplicate would win and get treated as the "real" heading,
# discarding almost the entire section).
#
# Instead: for each candidate heading line, score it by how much text
# follows before hitting a candidate heading for a DIFFERENT section. This
# lets repeated running headers for the same section coexist without
# artificially shortening the candidate's content run.

def _find_candidates(lines: list[str], patterns: dict[str, re.Pattern]) -> dict[str, list[int]]:
    candidates: dict[str, list[int]] = {s: [] for s in patterns}
    for i, line in enumerate(lines):
        if len(line) > 200:
            continue  # heading lines are short; a 200+ char line is prose
        norm = normalize_line(line)
        if not norm:
            continue
        for section, pattern in patterns.items():
            if pattern.search(norm):
                candidates[section].append(i)
    return {s: idxs for s, idxs in candidates.items() if idxs}


def split_by_regex(text: str) -> dict[str, str]:
    lines = text.split("\n")

    candidates = _find_candidates(lines, SECTION_PATTERNS_ITEM)

    # Some filers (notably JPMorgan Chase and ExxonMobil) make Item 7/8 in
    # Part II only a short cross-reference and place the real MD&A / financial
    # statements later in a separate financial section. In those filings the
    # Item heading exists, so a "narrative only when Item is missing" fallback
    # never sees the real section heading. Always supplement MD&A and financial
    # statements with narrative candidates; keep the old missing-only behavior
    # for Business/Risk Factors to avoid broad narrative false positives.
    narrative_sections = [
        section
        for section in SECTION_PATTERNS_NARRATIVE
        if section not in candidates or section in {"mdna", "financial_statements"}
    ]
    if narrative_sections:
        narrative_patterns = {
            section: SECTION_PATTERNS_NARRATIVE[section]
            for section in narrative_sections
        }
        narrative_candidates = _find_candidates(lines, narrative_patterns)
        for section, idxs in narrative_candidates.items():
            candidates.setdefault(section, []).extend(idxs)
            candidates[section] = sorted(set(candidates[section]))

    if not candidates:
        return {"full_filing": text}

    line_char_offsets = [0]
    for line in lines:
        line_char_offsets.append(line_char_offsets[-1] + len(line) + 1)  # +1 for the '\n'

    all_positions = sorted(
        (idx, section) for section, idxs in candidates.items() for idx in idxs
    )

    def next_boundary_after(idx: int, section: str) -> int:
        for other_idx, other_section in all_positions:
            if other_idx > idx and other_section != section:
                return other_idx
        return len(lines)

    best_start: dict[str, int] = {}
    for section, idxs in candidates.items():
        scored = []
        for idx in idxs:
            end_idx = next_boundary_after(idx, section)
            char_len = line_char_offsets[end_idx] - line_char_offsets[idx]
            scored.append((char_len, idx))
        scored.sort(reverse=True)  # largest content run (by characters) wins
        best_start[section] = scored[0][1]

    ordered = sorted(best_start.items(), key=lambda kv: kv[1])
    sections: dict[str, str] = {}
    for pos, (section, start_line) in enumerate(ordered):
        end_line = ordered[pos + 1][1] if pos + 1 < len(ordered) else len(lines)
        section_text = "\n".join(lines[start_line:end_line]).strip()
        if len(section_text) > MIN_SECTION_CHARS:
            sections[section] = section_text

    return sections or {"full_filing": text}

def split_into_sections(html: str, text: str) -> tuple[dict[str, str], str]:
    """Returns (sections, strategy_used) for logging/diagnostics."""
    anchor_sections = split_by_anchors(html)
    if anchor_sections:
        regex_sections = split_by_regex(text)
        merged = dict(regex_sections)
        merged.update(anchor_sections)  
        merged.pop("full_filing", None) if len(merged) > 1 else None
        return merged, "anchor"
    return split_by_regex(text), "regex"


def load_html_filing(file_path: Path) -> list[Document]:
    """Load one HTML filing and return section-level LangChain Documents."""
    base_metadata = parse_filing_metadata(file_path)

    html = file_path.read_text(encoding="utf-8", errors="ignore")
    text = html_to_text(html)

    is_amendment, scope_note = detect_amendment(text)
    base_metadata["is_amendment"] = is_amendment
    if is_amendment:
        base_metadata["filing"] = "10-K/A"

    if is_amendment:
        print(
            f"{file_path.name}: detected as a 10-K/A amendment ({scope_note}). "
            f"Skipping Business/Risk Factors/MD&A/Financial Statements extraction -- "
            f"an amendment covering only later Items will not contain them by design. "
            f"If you need full section coverage for this company/fiscal year, ingest the "
            f"ORIGINAL 10-K instead; this file will still be indexed under "
            f"section='amendment_supplement' so its content isn't lost."
        )
        docs = [Document(
            page_content=text,
            metadata={**base_metadata, "section": "amendment_supplement"},
        )]
        return docs

    sections, strategy = split_into_sections(html, text)
    print(f"{file_path.name} sections found ({strategy} strategy): {list(sections.keys())}")

    docs: list[Document] = []
    for section_name, section_text in sections.items():
        metadata = {**base_metadata, "section": section_name}
        docs.append(Document(page_content=section_text, metadata=metadata))

    return docs


def collect_documents(data_root: Path) -> list[Document]:
    docs: list[Document] = []
    patterns = ["*/*_10k.html", "*/*_10ka.html", "*/*_10k_a.html"]
    seen: set[Path] = set()
    for pattern in patterns:
        for file_path in sorted(data_root.glob(pattern)):
            if file_path in seen:
                continue
            seen.add(file_path)
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