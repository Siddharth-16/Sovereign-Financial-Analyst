from app.financial_answer import answer_financial_point_lookup, relevant_financial_rows


def test_net_income_prefers_common_shareholder_row():
    content = (
        "(MILLIONS) 2024 2023 2022 "
        "Net income before allocation to noncontrolling interests 8,062 2,158 31,407 "
        "Less: Net income attributable to noncontrolling interests 31 39 35 "
        "Net income attributable to Pfizer Inc. common shareholders $ 8,031 $ 2,119 $ 31,372"
    )
    answer = answer_financial_point_lookup("What does the income statement show regarding net income?", "Pfizer", content)
    assert "8,031" in answer
    assert "Net income attributable to Pfizer Inc. common shareholders" in answer
    assert "8,062" not in answer


def test_capex_returns_all_distinct_explicit_measures():
    content = (
        "(millions of dollars) 2024 2023 "
        "Capital and Exploration Expenditures (Capex) 27,551 26,325 "
        "Additions to property, plant and equipment 24,306 21,919 "
        "Total Cash Capex (Non-GAAP) 25,647 23,228"
    )
    answer = answer_financial_point_lookup("What do the financial statements disclose about capital expenditures?", "ExxonMobil", content)
    assert "27,551" in answer
    assert "24,306" in answer
    assert "25,647" in answer


def test_operating_cash_flow_beats_cash_flow_hedge_prose():
    content = (
        "Cash flow hedges are recorded in other comprehensive income. "
        "(in millions) Net cash provided by (used in) operating activities 23,059 19,950 20,755"
    )
    rows = relevant_financial_rows("cash flow from operations", content)
    assert rows == ["Net cash provided by (used in) operating activities 23,059 19,950 20,755"]


def test_total_liabilities_does_not_misread_total_assets_or_total_equity_line():
    content = (
        "(Amounts in millions) Total assets $ 260,823 $ 252,399 "
        "Total current liabilities 96,584 92,415 Long-term debt 33,401 36,132 "
        "Long-term operating lease obligations 12,825 12,943 "
        "Long-term finance lease obligations 5,923 5,709 "
        "Deferred income taxes and other 14,398 14,629 "
        "Total liabilities, redeemable noncontrolling interest, and shareholders' equity $ 260,823 $ 252,399"
    )
    answer = answer_financial_point_lookup("What does the balance sheet show regarding total liabilities?", "Walmart", content)
    assert "96,584" in answer
    assert "33,401" in answer
    assert "260,823" not in answer


def test_debt_includes_total_and_current_portion():
    content = (
        "(Dollars in millions) Short-term debt and current portion of long-term debt at December 31 consisted of the following: "
        "2024 2023 Unsecured debt $ 850 $ 5,072 Finance lease obligations 86 77 Other notes 342 55 Total $ 1,278 $ 5,204 "
        "Debt at December 31 consisted of the following: Total debt $ 53,864 $ 52,307"
    )
    answer = answer_financial_point_lookup("What does the balance sheet disclose about debt levels?", "Boeing", content)
    assert "53,864" in answer
    assert "1,278" in answer