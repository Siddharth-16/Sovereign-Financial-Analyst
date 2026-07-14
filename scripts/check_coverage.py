from __future__ import annotations
import os
import sys
from collections import defaultdict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import CHROMA_PATH, EMBED_MODEL
from app.companies import COMPANIES, TARGET_FISCAL_YEARS, REQUIRED_SECTIONS


def load_indexed_metadata(batch_size: int = 500) -> list[dict]:
    """Pull every chunk's metadata out of the persisted Chroma collection."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    all_metadatas: list[dict] = []
    offset = 0
    while True:
        result = db.get(include=["metadatas"], limit=batch_size, offset=offset)
        ids = result.get("ids", []) or []
        if not ids:
            break
        all_metadatas.extend(result.get("metadatas", []) or [])
        offset += len(ids)

    return all_metadatas


def build_coverage_map(metadatas: list[dict]) -> dict[tuple[str, int], set[str]]:
    """(company_slug, fiscal_year) -> set of section names actually indexed."""
    coverage: dict[tuple[str, int], set[str]] = defaultdict(set)

    for md in metadatas:
        slug = md.get("company_slug")
        fiscal_year = md.get("fiscal_year")
        section = md.get("section")

        if slug is None or fiscal_year is None or section is None:
            continue

        coverage[(slug, int(fiscal_year))].add(section)

    return coverage


def main() -> int:
    print(f"Reading Chroma collection at {CHROMA_PATH} ...")
    metadatas = load_indexed_metadata()

    if not metadatas:
        print("No chunks found in the collection. Has ingestion been run? "
              "(python scripts/ingest.py)")
        return 1

    coverage = build_coverage_map(metadatas)

    expected_combos = [
        (slug, fy)
        for slug in COMPANIES
        for fy in sorted(TARGET_FISCAL_YEARS)
    ]

    missing_entirely: list[tuple[str, int]] = []       # no chunks at all
    fell_back_to_full_filing: list[tuple[str, int]] = []  # split silently failed
    partial_sections: list[tuple[str, int, set[str]]] = []  # some sections missing

    for slug, fy in expected_combos:
        sections_found = coverage.get((slug, fy))

        if not sections_found:
            missing_entirely.append((slug, fy))
            continue

        if sections_found == {"full_filing"}:
            fell_back_to_full_filing.append((slug, fy))
            continue

        missing_sections = REQUIRED_SECTIONS - sections_found
        if missing_sections:
            partial_sections.append((slug, fy, missing_sections))

    total = len(expected_combos)
    fully_covered = total - len(missing_entirely) - len(fell_back_to_full_filing) - len(partial_sections)

    print("\n" + "=" * 72)
    print("COVERAGE REPORT")
    print("=" * 72)
    print(f"Expected: {len(COMPANIES)} companies x {len(TARGET_FISCAL_YEARS)} fiscal years "
          f"= {total} filings, each needing {len(REQUIRED_SECTIONS)} sections "
          f"({', '.join(sorted(REQUIRED_SECTIONS))})")
    print(f"Fully covered: {fully_covered}/{total}")

    if missing_entirely:
        print(f"\nNot indexed at all ({len(missing_entirely)}):")
        for slug, fy in missing_entirely:
            print(f"  - {slug} FY{fy}")

    if fell_back_to_full_filing:
        print(f"\nHeading match failed completely — indexed as one 'full_filing' "
              f"blob instead of 4 sections ({len(fell_back_to_full_filing)}):")
        for slug, fy in fell_back_to_full_filing:
            print(f"  - {slug} FY{fy}")

    if partial_sections:
        print(f"\nPartially indexed — some sections missing ({len(partial_sections)}):")
        for slug, fy, missing in partial_sections:
            print(f"  - {slug} FY{fy}: missing {sorted(missing)}")

    print("=" * 72)

    if fully_covered == total:
        print("All companies x fiscal years x sections are fully indexed.")
        return 0

    print(f"{total - fully_covered} filing(s) have incomplete coverage. "
          f"See above for exactly where section splitting is silently failing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())