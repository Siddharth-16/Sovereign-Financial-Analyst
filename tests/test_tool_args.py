from app.tool_args import sanitize_tool_args


def test_empty_fiscal_year_becomes_none():
    args = sanitize_tool_args(
        "search_filing",
        {
            "company": "Microsoft",
            "question": "revenue growth drivers",
            "section": "mdna",
            "fiscal_year": "",
        },
        user_input="What does Microsoft's MD&A say about revenue growth drivers?",
    )
    assert args["fiscal_year"] is None


def test_numeric_fiscal_year_string_becomes_int():
    args = sanitize_tool_args(
        "search_filing",
        {"company": "Nvidia", "question": "revenue", "fiscal_year": "FY2025"},
        user_input="What was Nvidia revenue in FY2025?",
    )
    assert args["fiscal_year"] == 2025


def test_degenerate_query_falls_back_to_original_user_question():
    user_input = "What does Microsoft's MD&A say about revenue growth drivers?"
    args = sanitize_tool_args(
        "search_filing",
        {
            "company": "Microsoft",
            "question": "What does Microsoft",
            "section": "mdna",
            "fiscal_year": "",
        },
        user_input=user_input,
    )
    assert args["question"] == user_input


def test_short_but_meaningful_query_is_preserved():
    args = sanitize_tool_args(
        "search_filing",
        {"company": "Pfizer", "question": "net income", "section": "financial_statements"},
        user_input="What does Pfizer's income statement show regarding net income?",
    )
    assert args["question"] == "net income"


def test_non_search_tool_args_are_unchanged():
    original = {"ticker": "NVDA"}
    assert sanitize_tool_args("get_stock_price", original, user_input="price") == original
