from app.tool_args import sanitize_tool_args


def test_reportable_segments_repairs_wrong_financial_section():
    raw = {
        "company": "JPMorgan Chase",
        "question": "main reportable business segments",
        "section": "financial_statements",
    }
    out = sanitize_tool_args(
        "search_filing",
        raw,
        user_input="What are JPMorgan Chase's main reportable business segments?",
    )
    assert out["section"] == "business"


def test_mdna_explicit_language_wins():
    raw = {"company": "Goldman Sachs", "question": "net income drivers", "section": "financial_statements"}
    out = sanitize_tool_args(
        "search_filing",
        raw,
        user_input="What does Goldman Sachs' MD&A discuss about net income drivers?",
    )
    assert out["section"] == "mdna"


def test_balance_sheet_stays_financial():
    raw = {"company": "Walmart", "question": "total liabilities", "section": "business"}
    out = sanitize_tool_args(
        "search_filing",
        raw,
        user_input="What does Walmart's balance sheet show regarding total liabilities?",
    )
    assert out["section"] == "financial_statements"
