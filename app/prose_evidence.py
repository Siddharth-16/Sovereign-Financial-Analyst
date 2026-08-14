from __future__ import annotations

import html
import re
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.&%-][a-z0-9]+)*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "it", "its", "of", "on", "or", "say", "says", "the",
    "their", "this", "to", "was", "were", "what", "which", "with", "about",
    "company", "companies", "discuss", "discusses", "discussion", "md&a", "mdna",
}

_CONCEPT_EXPANSIONS = {
    "income": {"income", "earnings", "profit", "revenue", "revenues"},
    "earnings": {"earnings", "income", "profit"},
    "revenue": {"revenue", "revenues", "sales"},
    "revenues": {"revenue", "revenues", "sales"},
    "sales": {"sales", "revenue", "revenues"},
    "margin": {"margin", "income", "profit"},
    "trend": {"trend", "trends", "growth", "increase", "increased", "decrease", "decreased", "higher", "lower"},
    "trends": {"trend", "trends", "growth", "increase", "increased", "decrease", "decreased", "higher", "lower"},
    "growth": {"growth", "grew", "increase", "increased", "higher"},
    "driver": {"driver", "drivers", "driven", "primarily", "reflecting", "due"},
    "drivers": {"driver", "drivers", "driven", "primarily", "reflecting", "due"},
    "advertising": {"advertising", "advertiser", "ads"},
    "manufacturing": {"manufacturing", "production", "factory", "factories", "supplier", "suppliers"},
    "production": {"production", "manufacturing", "supplier", "suppliers", "components"},
    "regulatory": {"regulatory", "regulation", "regulations", "laws", "compliance"},
    "environmental": {"environmental", "climate", "emissions", "greenhouse"},
    "privacy": {"privacy", "data", "regulation", "regulatory"},
}

_TREND_RE = re.compile(
    r"\b(?:increase(?:d)?|decrease(?:d)?|higher|lower|grew|growth|declin(?:e|ed)|"
    r"primarily|driven|driver|due\s+to|reflecting|compared)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\$?\s*\d[\d,.]*(?:\s*(?:%|percent|billion|million))?", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _query_terms(question: str) -> tuple[set[str], set[str]]:
    base = {t for t in _tokens(question) if t not in _STOPWORDS}
    expanded = set(base)
    for token in base:
        expanded.update(_CONCEPT_EXPANSIONS.get(token, set()))
    return base, expanded


def _paragraphs(content: str) -> list[str]:
    raw = re.split(r"\n\s*\n", content or "")
    out = []
    for part in raw:
        cleaned = html.unescape(re.sub(r"\s+", " ", part)).strip()
        if len(cleaned) >= 30:
            out.append(cleaned)
    return out


def _passage_score(question: str, passage: str) -> float:
    base, expanded = _query_terms(question)
    passage_tokens = set(_tokens(passage))
    direct = len(base & passage_tokens)
    related = len(expanded & passage_tokens)
    numbers = min(len(_NUMBER_RE.findall(passage)), 10)
    trend_hits = min(len(_TREND_RE.findall(passage)), 8)

    score = direct * 4.0 + related * 1.5 + numbers * 0.55 + trend_hits * 1.15

    q = (question or "").lower()
    p = passage.lower()

    # Questions about trends/drivers should prefer actual period-over-period
    # result discussion over generic forward-looking prose.
    if any(term in q for term in ("trend", "growth", "driver", "margin", "increase", "decrease")):
        if re.search(r"\b20\d{2}\b", p) and numbers:
            score += 4.0
        if any(phrase in p for phrase in ("primarily due to", "primarily driven", "primarily reflecting")):
            score += 4.0
        if any(phrase in p for phrase in ("we expect", "may affect", "could affect", "anticipate that")) and not numbers:
            score -= 3.0

    return score


def select_evidence_passages(
    question: str,
    content: str,
    *,
    max_passages: int = 4,
    max_chars_each: int = 850,
) -> list[str]:
    """Return compact, high-signal passages from retrieved filing context.

    This is deliberately extractive: it preserves names, values, directions and
    stated drivers instead of asking a small model to rediscover them.
    """
    passages = _paragraphs(content)
    ranked = sorted(
        ((-_passage_score(question, p), idx, p) for idx, p in enumerate(passages)),
        key=lambda item: (item[0], item[1]),
    )

    selected: list[str] = []
    seen: set[str] = set()
    for _, _, passage in ranked:
        key = re.sub(r"\W+", " ", passage.casefold()).strip()
        if not key or key in seen:
            continue
        seen.add(key)

        if len(passage) > max_chars_each:
            clipped = passage[:max_chars_each]
            boundary = max(clipped.rfind(". "), clipped.rfind("; "))
            if boundary >= int(max_chars_each * 0.6):
                clipped = clipped[: boundary + 1]
            passage = clipped.rstrip() + "…"

        selected.append(passage)
        if len(selected) >= max_passages:
            break
    return selected


def build_evidence_brief(question: str, content: str) -> str:
    passages = select_evidence_passages(question, content)
    if not passages:
        return ""
    return "\n".join(f"- {passage}" for passage in passages)


def should_preserve_evidence(question: str, section: str | None) -> bool:
    """Use an extractive appendix only as a quantitative MD&A safety net.

    Business and Risk Factors still receive the evidence brief in the synthesis
    prompt, but remain normal prose answers. This avoids bloating every answer.
    """
    if section != "mdna":
        return False
    q = (question or "").lower()
    return any(term in q for term in (
        "trend", "growth", "driver", "margin", "revenue", "sales", "income", "earnings",
    ))


def append_evidence_if_useful(answer: str, brief: str, *, question: str, section: str | None) -> str:
    if not brief or not should_preserve_evidence(question, section):
        return answer

    answer_numbers = {
        re.sub(r"\s+", "", m.group(0).lower())
        for m in _NUMBER_RE.finditer(answer or "")
        if len(re.sub(r"\D", "", m.group(0))) >= 2
    }
    brief_numbers = {
        re.sub(r"\s+", "", m.group(0).lower())
        for m in _NUMBER_RE.finditer(brief)
        if len(re.sub(r"\D", "", m.group(0))) >= 2
    }

    # If synthesis already retained several concrete values from the evidence,
    # leave the concise answer alone. The appendix is for failures like
    # Goldman/Amazon where the small model turns specific evidence into a
    # generic narrative or focuses on the wrong nearby metric.
    if len(answer_numbers & brief_numbers) >= 5:
        return answer
    if len(brief_numbers) < 2:
        return answer

    return (
        f"{answer.rstrip()}\n\n"
        "Supporting filing evidence:\n"
        f"{brief}"
    )

