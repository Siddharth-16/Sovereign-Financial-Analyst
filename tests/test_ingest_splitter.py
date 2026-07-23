from __future__ import annotations
from scripts.ingest import (
    classify_toc_text,
    detect_amendment,
    normalize_line,
    split_by_anchors,
    split_by_regex,
)

_FILLER = " Lorem ipsum dolor sit amet, consectetur adipiscing elit." * 30  # ~1750 chars


# --------------------------------------------------------------------- normalize_line


def test_normalize_line_strips_punctuation_and_case():
    assert normalize_line("Item 1A. Risk Factors") == "item 1a risk factors"


def test_normalize_line_collapses_whitespace():
    assert normalize_line("  Item   1.   Business  ") == "item 1 business"


# --------------------------------------------------------------------- classify_toc_text


def test_classify_toc_text_matches_item_heading():
    assert classify_toc_text("Item 1A. Risk Factors") == "risk_factors"


def test_classify_toc_text_no_match():
    assert classify_toc_text("Exhibit 21.1 Subsidiaries") is None


# --------------------------------------------------------------------- split_by_regex


def test_split_by_regex_finds_all_four_sections():
    text = (
        f"Item 1. Business\n{_FILLER}\n"
        f"Item 1A. Risk Factors\n{_FILLER}\n"
        f"Item 7. Management's Discussion and Analysis\n{_FILLER}\n"
        f"Item 8. Financial Statements and Supplementary Data\n{_FILLER}\n"
    )
    sections = split_by_regex(text)
    assert set(sections.keys()) == {"business", "risk_factors", "mdna", "financial_statements"}


def test_split_by_regex_prefers_largest_content_run_over_toc_mention():
    """This is the exact bug class called out in the roadmap comments: a
    short TOC/cross-reference mention of a heading must lose to the real
    section, which is followed by a much larger run of content before the
    next heading."""
    text = (
        "Item 1A. Risk Factors\n"  
        "see page 12\n"
        f"Item 1. Business\n{_FILLER}\n"
        f"Item 1A. Risk Factors\n{_FILLER}\n" 
        f"Item 7. Management's Discussion and Analysis\n{_FILLER}\n"
    )
    sections = split_by_regex(text)
    assert "risk_factors" in sections
    assert "consectetur" in sections["risk_factors"]


def test_split_by_regex_ignores_long_lines_as_non_headings():
    long_line = "Item 1A. Risk Factors " + "x" * 220
    text = f"{long_line}\n{_FILLER}"
    sections = split_by_regex(text)
    assert list(sections.keys()) == ["full_filing"]


def test_split_by_regex_falls_back_to_full_filing_when_no_headings():
    text = f"Just a plain filing with no Item headings at all.\n{_FILLER}"
    sections = split_by_regex(text)
    assert list(sections.keys()) == ["full_filing"]


def test_split_by_regex_drops_sections_below_min_length():
    text = (
        "Item 1. Business\n"
        "very short\n"
        f"Item 1A. Risk Factors\n{_FILLER}\n"
    )
    sections = split_by_regex(text)
    assert "business" not in sections
    assert "risk_factors" in sections


# --------------------------------------------------------------------- split_by_anchors


def test_split_by_anchors_uses_toc_hyperlinks():
    html = f"""
    <html><body>
    <table>
      <tr><td><a href="#risk_anchor">Item 1A. Risk Factors</a></td></tr>
      <tr><td><a href="#biz_anchor">Item 1. Business</a></td></tr>
    </table>
    <a id="biz_anchor"></a>
    <p>Item 1. Business</p>
    <p>{_FILLER}</p>
    <a id="risk_anchor"></a>
    <p>Item 1A. Risk Factors</p>
    <p>{_FILLER}</p>
    </body></html>
    """
    sections = split_by_anchors(html)
    assert sections is not None
    assert "business" in sections
    assert "risk_factors" in sections


def test_split_by_anchors_returns_none_with_insufficient_toc():
    html = f"""
    <html><body>
    <table><tr><td><a href="#biz_anchor">Item 1. Business</a></td></tr></table>
    <a id="biz_anchor"></a>
    <p>Item 1. Business</p>
    <p>{_FILLER}</p>
    </body></html>
    """
    assert split_by_anchors(html) is None


# --------------------------------------------------------------------- detect_amendment


def test_detect_amendment_true_for_10ka():
    text = f"This Amendment No. 1 to Form 10-K is filed solely to amend Part III\n{_FILLER}"
    is_amendment, scope = detect_amendment(text)
    assert is_amendment is True
    assert scope is not None


def test_detect_amendment_false_for_normal_filing():
    text = f"Item 1. Business\n{_FILLER}"
    is_amendment, scope = detect_amendment(text)
    assert is_amendment is False
    assert scope is None
