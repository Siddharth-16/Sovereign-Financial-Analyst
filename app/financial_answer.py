from __future__ import annotations

import re
from typing import Optional

# A single accounting value as it commonly appears in SEC filing text.
# Handles: 24,306 | $ 24,306 | (24,306) | ( 24,306 ) | -24,306
_MONEY = r"(?:\$\s*)?(?:\(\s*)?-?\d[\d,]*(?:\.\d+)?(?:\s*\))?"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _first_row(content: str, anchor: str, max_values: int = 4) -> Optional[str]:
    """Return an exact accounting row beginning at *anchor*.

    We intentionally copy the filing's row instead of numerically interpreting
    it. This prevents a small language model from confusing adjacent rows or
    changing units.
    """
    pattern = rf"{anchor}\s+{_MONEY}(?:\s+{_MONEY}){{0,{max_values - 1}}}"
    match = re.search(pattern, content or "", flags=re.IGNORECASE)
    return _clean(match.group(0)) if match else None


def _short_term_debt_total(content: str) -> Optional[str]:
    """Extract the total current/short-term debt amount from a debt table."""
    pattern = (
        r"Short-term debt and current portion of long-term debt"
        r".{0,600}?\bTotal\s+(" + _MONEY + r")(?:\s+(" + _MONEY + r"))?"
    )
    match = re.search(pattern, content or "", flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    values = " ".join(_clean(v) for v in match.groups() if v)
    return f"Short-term debt and current portion of long-term debt — Total {values}"


def _reported_in_millions(content: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?:dollars\s+in\s+millions|amounts\s+in\s+millions|"
            r"\bin\s+millions\b|\(millions\)|millions\s+of\s+dollars)",
            content or "",
        )
    )


def relevant_financial_rows(question: str, content: str) -> list[str]:
    """Select explicitly labeled rows for simple financial point lookups.

    This is company-agnostic and value-agnostic. It recognizes common
    accounting metrics, then copies matching rows verbatim from retrieved SEC
    filing context. Broad/analytical questions simply return no rows and fall
    back to normal LLM synthesis.
    """
    q = (question or "").lower()
    rows: list[str] = []
    seen: set[str] = set()

    def add(row: Optional[str]) -> None:
        if not row:
            return
        key = row.casefold()
        if key not in seen:
            rows.append(row)
            seen.add(key)

    # Balance-sheet asset point lookups.
    if "total asset" in q or ("asset" in q and "total" in q):
        add(_first_row(content, r"\bTotal assets"))

    # Liability questions often need the explicitly listed components when a
    # standalone "Total liabilities" row is not present in the balance sheet.
    if "liabil" in q:
        for anchor in (
            r"\bTotal current liabilities",
            r"\bLong[- ]term debt",
            r"\bLong[- ]term operating lease obligations",
            r"\bLong[- ]term finance lease obligations",
            r"\bDeferred income taxes and other",
        ):
            add(_first_row(content, anchor))
        # Do not confuse "Total liabilities, ... and shareholders' equity"
        # with a standalone total-liabilities row.
        add(_first_row(content, r"\bTotal liabilities(?!\s*,)"))

    # Statement of cash flows.
    if any(
        phrase in q
        for phrase in (
            "cash flow from operations",
            "cash flows from operations",
            "operating cash flow",
            "cash flow from operating",
            "operating activities",
        )
    ):
        add(
            _first_row(
                content,
                r"\bNet cash provided by \(used in\) operating activities",
            )
        )

    # Debt point lookups.
    if "debt" in q and "liabil" not in q:
        add(_first_row(content, r"\bTotal debt"))
        add(_short_term_debt_total(content))

    # Revenue point lookups. Prefer an explicit Total revenue row; fall back to
    # the top-line Revenue row from an income statement.
    if "revenue" in q:
        add(_first_row(content, r"\bTotal revenue"))
        if not rows:
            add(_first_row(content, r"(?<!of )(?<!total )\bRevenue(?=\s+\$)"))

    # Issuer net income: when both "before allocation to noncontrolling
    # interests" and issuer-attributable net income are present, prefer the
    # latter for an unqualified net-income question.
    if "net income" in q:
        add(
            _first_row(
                content,
                r"\bNet income attributable to (?!noncontrolling\b)"
                r"[^$\n]{0,100}?common shareholders",
            )
        )
        if not rows:
            add(_first_row(content, r"\bNet income(?=\s+\$)"))

    # Capex is not one universal measure. If the filing explicitly presents
    # multiple related measures, preserve all of them instead of silently
    # collapsing them into one number.
    if "capital expenditure" in q or "capital expenditures" in q or "capex" in q:
        for anchor in (
            r"\bCapital and Exploration Expenditures \(Capex\)",
            r"\bAdditions to property, plant and equipment",
            r"\bTotal Cash Capex \(Non-GAAP\)",
        ):
            add(_first_row(content, anchor))

    return rows


def answer_financial_point_lookup(
    question: str,
    company: str,
    content: str,
) -> Optional[str]:
    """Return a deterministic evidence-first answer when possible.

    Returns None for broad financial-statement questions so the normal LLM
    synthesis path remains available.
    """
    rows = relevant_financial_rows(question, content)
    if not rows:
        return None

    unit_note = " The filing reports these table values in millions." if _reported_in_millions(content) else ""
    prefix = (
        f"According to {company}'s retrieved financial statements, the relevant "
        f"reported figures are below (latest period first).{unit_note}"
    )
    rendered = "\n".join(f"- {row}" for row in rows)
    return f"{prefix}\n{rendered}"
