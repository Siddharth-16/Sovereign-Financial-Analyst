from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError, model_validator


class JudgeParseError(ValueError):
    """Raised when the groundedness judge output is not valid strict JSON."""


class ClaimJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim: str
    supported: StrictBool
    evidence_quote: str

    @model_validator(mode="after")
    def validate_evidence_quote_contract(self):
        if self.supported and not self.evidence_quote.strip():
            raise ValueError("supported claims require a non-empty evidence_quote")
        if not self.supported and self.evidence_quote.strip():
            raise ValueError("unsupported claims must use an empty evidence_quote")
        return self


class GroundednessJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claims: list[ClaimJudgment]


@dataclass(frozen=True)
class AuditedClaim:
    claim: str
    judge_supported: bool
    supported: bool
    evidence_quote: str
    evidence_quote_valid: bool
    validation_error: str | None = None


@dataclass(frozen=True)
class GoldScore:
    status: str
    found: int
    total: int
    matches: list[dict[str, Any]]

    @property
    def recall(self) -> float | None:
        return (self.found / self.total) if self.total else None

    @property
    def question_success(self) -> bool | None:
        return (self.found == self.total) if self.total else None


@dataclass(frozen=True)
class CompletenessScore:
    status: str
    found: int
    total: int
    matches: list[dict[str, Any]]

    @property
    def recall(self) -> float | None:
        return (self.found / self.total) if self.total else None


_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


def normalize_text(text: str, *, remove_punctuation: bool = True) -> str:
    """Normalize text for auditable span matching.

    The normalization is intentionally deterministic: Unicode compatibility
    normalization, quote/dash normalization, case-folding, punctuation-to-space,
    and whitespace collapsing. It does not do stemming or semantic matching.
    """

    value = unicodedata.normalize("NFKC", text or "").translate(_QUOTE_TRANSLATION).casefold()
    if remove_punctuation:
        value = "".join(
            " " if unicodedata.category(ch)[0] in {"P", "S"} else ch
            for ch in value
        )
    return re.sub(r"\s+", " ", value).strip()


def span_is_traceable(span: str, context: str) -> bool:
    normalized_span = normalize_text(span)
    normalized_context = normalize_text(context)
    return bool(normalized_span) and normalized_span in normalized_context


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text


def parse_groundedness_judgment(raw: str) -> GroundednessJudgment:
    cleaned = _strip_json_fence(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"judge returned malformed JSON: {exc.msg}") from exc

    try:
        return GroundednessJudgment.model_validate(payload)
    except ValidationError as exc:
        raise JudgeParseError(f"judge JSON failed strict schema validation: {exc}") from exc


def call_judge_with_repair(
    invoke: Callable[[list[dict[str, str]]], str],
    messages: list[dict[str, str]],
    *,
    repair_system_prompt: str,
) -> tuple[GroundednessJudgment | None, str | None, int, list[str]]:
    """Call a judge once, then perform at most one JSON-repair retry.

    Returns (judgment, error, attempts, raw_outputs). A second invalid response is
    surfaced as JUDGE_ERROR by the caller rather than coerced into a score.
    """

    raw_outputs: list[str] = []
    raw = invoke(messages)
    raw_outputs.append(raw)
    try:
        return parse_groundedness_judgment(raw), None, 1, raw_outputs
    except JudgeParseError as first_error:
        repair_messages = [
            {"role": "system", "content": repair_system_prompt},
            {
                "role": "user",
                "content": (
                    "Repair the following judge output into the required JSON schema. "
                    "Do not change the substantive verdicts; only repair formatting/types.\n\n"
                    f"INVALID OUTPUT:\n{raw}"
                ),
            },
        ]
        repaired = invoke(repair_messages)
        raw_outputs.append(repaired)
        try:
            return parse_groundedness_judgment(repaired), None, 2, raw_outputs
        except JudgeParseError as second_error:
            return (
                None,
                f"initial parse error: {first_error}; repair parse error: {second_error}",
                2,
                raw_outputs,
            )


def audit_claim_evidence(
    judgment: GroundednessJudgment,
    context: str,
) -> list[AuditedClaim]:
    audited: list[AuditedClaim] = []
    for claim in judgment.claims:
        quote_valid = span_is_traceable(claim.evidence_quote, context) if claim.supported else True
        supported = bool(claim.supported and quote_valid)
        audited.append(
            AuditedClaim(
                claim=claim.claim,
                judge_supported=bool(claim.supported),
                supported=supported,
                evidence_quote=claim.evidence_quote,
                evidence_quote_valid=quote_valid,
                validation_error=(
                    None
                    if quote_valid
                    else "judge marked claim supported but evidence_quote is not traceable to CONTEXT"
                ),
            )
        )
    return audited


def groundedness_from_claims(claims: Sequence[AuditedClaim]) -> dict[str, Any]:
    if not claims:
        return {
            "status": "NO_FACTUAL_CLAIMS",
            "supported_claims": 0,
            "total_claims": 0,
            "claim_groundedness": None,
            "answer_grounded": None,
            "unsupported_claims": [],
        }

    supported = sum(1 for claim in claims if claim.supported)
    return {
        "status": "GRADED",
        "supported_claims": supported,
        "total_claims": len(claims),
        "claim_groundedness": supported / len(claims),
        "answer_grounded": supported == len(claims),
        "unsupported_claims": [claim.claim for claim in claims if not claim.supported],
    }


def _gold_item_candidates(item: Any) -> tuple[str, list[str]]:
    if isinstance(item, str):
        return item[:80], [item]
    if not isinstance(item, dict):
        raise TypeError("gold_evidence entries must be strings or objects")

    item_id = str(item.get("id") or item.get("description") or "gold-evidence")
    candidates: list[str] = []
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        candidates.append(text)
    alternatives = item.get("alternatives", [])
    if alternatives is not None:
        if not isinstance(alternatives, list) or not all(isinstance(v, str) for v in alternatives):
            raise TypeError("gold_evidence alternatives must be a list of strings")
        candidates.extend(v for v in alternatives if v.strip())
    if not candidates:
        raise ValueError(f"gold evidence item {item_id!r} contains no text")
    return item_id, candidates


def score_gold_evidence(gold_evidence: Sequence[Any] | None, context: str) -> GoldScore:
    if not gold_evidence:
        return GoldScore(status="UNGRADED", found=0, total=0, matches=[])

    matches: list[dict[str, Any]] = []
    found = 0
    for item in gold_evidence:
        item_id, candidates = _gold_item_candidates(item)
        matched_text = next((candidate for candidate in candidates if span_is_traceable(candidate, context)), None)
        hit = matched_text is not None
        found += int(hit)
        matches.append(
            {
                "id": item_id,
                "found": hit,
                "matched_gold_text": matched_text,
            }
        )
    return GoldScore(status="GRADED", found=found, total=len(gold_evidence), matches=matches)


def _literal_fact_match(pattern: str, answer: str) -> bool:
    return normalize_text(pattern) in normalize_text(answer)


def _fact_matches(fact: Any, answer: str) -> tuple[str, bool, str | None]:
    if isinstance(fact, str):
        return fact[:80], _literal_fact_match(fact, answer), fact if _literal_fact_match(fact, answer) else None
    if not isinstance(fact, dict):
        raise TypeError("expected_facts entries must be strings or objects")

    fact_id = str(fact.get("id") or fact.get("description") or "expected-fact")
    literal_patterns = fact.get("patterns", [])
    regex_patterns = fact.get("regex_patterns", [])
    if not isinstance(literal_patterns, list) or not all(isinstance(v, str) for v in literal_patterns):
        raise TypeError("expected_fact patterns must be a list of strings")
    if not isinstance(regex_patterns, list) or not all(isinstance(v, str) for v in regex_patterns):
        raise TypeError("expected_fact regex_patterns must be a list of strings")
    if not literal_patterns and not regex_patterns:
        raise ValueError(f"expected fact {fact_id!r} contains no matching pattern")

    for pattern in literal_patterns:
        if _literal_fact_match(pattern, answer):
            return fact_id, True, pattern
    for pattern in regex_patterns:
        if re.search(pattern, answer, flags=re.I | re.S):
            return fact_id, True, pattern
    return fact_id, False, None


def score_answer_completeness(expected_facts: Sequence[Any] | None, answer: str) -> CompletenessScore:
    if not expected_facts:
        return CompletenessScore(status="UNGRADED", found=0, total=0, matches=[])

    found = 0
    matches: list[dict[str, Any]] = []
    for fact in expected_facts:
        fact_id, hit, pattern = _fact_matches(fact, answer)
        found += int(hit)
        matches.append({"id": fact_id, "found": hit, "matched_pattern": pattern})
    return CompletenessScore(status="GRADED", found=found, total=len(expected_facts), matches=matches)


_ABSTENTION_PATTERNS = [
    r"\bnot (?:available|indexed|provided|found)\b",
    r"\binformation (?:is|was) unavailable\b",
    r"\bno (?:filing )?(?:data|information|content) (?:is )?available\b",
    r"\bcould not (?:find|retrieve|locate)\b",
    r"\bunable to (?:find|retrieve|locate)\b",
    r"\bdoes not contain (?:the )?(?:requested )?(?:information|data)\b",
]


def detect_abstention(answer: str) -> tuple[bool, str | None]:
    for pattern in _ABSTENTION_PATTERNS:
        match = re.search(pattern, answer or "", flags=re.I)
        if match:
            return True, match.group(0)
    return False, None


def classify_abstention(abstained: bool, answerable: bool | None) -> str:
    if not abstained:
        return "NOT_ABSTAINED"
    if answerable is True:
        return "INCORRECT"
    if answerable is False:
        return "CORRECT"
    return "UNGRADED"


def metadata_retrieval_hit(
    citations: Iterable[str],
    expected_company_display: str,
    expected_section_display: str,
) -> bool:
    company_norm = normalize_text(expected_company_display)
    section_norm = normalize_text(expected_section_display)
    for citation in citations:
        citation_norm = normalize_text(citation)
        if company_norm in citation_norm and section_norm in citation_norm:
            return True
    return False


def extract_agentic_search_trace(tools_invoked: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Extract the exact search_filing results produced during one agent run."""

    contexts: list[str] = []
    citations: list[str] = []
    search_calls: list[dict[str, Any]] = []
    tool_errors = 0

    for call in tools_invoked:
        name = call.get("tool")
        raw_result = call.get("result")
        parsed: dict[str, Any] | None = None
        if isinstance(raw_result, str):
            try:
                candidate = json.loads(raw_result)
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                parsed = None

        if parsed is None or parsed.get("error"):
            tool_errors += 1

        if name != "search_filing":
            continue

        search_calls.append(call)
        if not parsed:
            continue
        content = parsed.get("content")
        if isinstance(content, str) and content:
            contexts.append(content)
        result_citations = parsed.get("citations", [])
        if isinstance(result_citations, list):
            citations.extend(str(c) for c in result_citations)

    return {
        "context": "\n\n".join(contexts),
        "citations": citations,
        "search_calls": search_calls,
        "search_call_count": len(search_calls),
        "total_tool_calls": len(tools_invoked),
        "tool_errors": tool_errors,
    }


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    pct = lambda numerator, denominator: (numerator / denominator) if denominator else None

    company_correct = sum(bool(r.get("routing_company_correct")) for r in results)
    section_correct = sum(bool(r.get("routing_section_correct")) for r in results)
    metadata_hits = sum(bool(r.get("metadata_retrieval_hit")) for r in results)

    evidence_found = sum(int(r.get("evidence_found", 0)) for r in results)
    evidence_total = sum(int(r.get("evidence_total", 0)) for r in results)
    evidence_gradable = [r for r in results if int(r.get("evidence_total", 0)) > 0]
    evidence_question_successes = sum(
        int(r.get("evidence_found", 0)) == int(r.get("evidence_total", 0))
        for r in evidence_gradable
    )

    graded_claim_answers = [
        r for r in results
        if r.get("groundedness_status") == "GRADED" and int(r.get("total_claims", 0)) > 0
    ]
    supported_claims = sum(int(r.get("supported_claims", 0)) for r in graded_claim_answers)
    total_claims = sum(int(r.get("total_claims", 0)) for r in graded_claim_answers)
    answer_grounded_count = sum(bool(r.get("answer_grounded")) for r in graded_claim_answers)
    macro_claim_values = [
        float(r["claim_groundedness"])
        for r in graded_claim_answers
        if r.get("claim_groundedness") is not None
    ]

    facts_found = sum(int(r.get("facts_found", 0)) for r in results)
    facts_total = sum(int(r.get("facts_total", 0)) for r in results)

    judge_errors = sum(r.get("groundedness_status") == "JUDGE_ERROR" for r in results)
    judge_requested = sum(bool(r.get("groundedness_requested")) for r in results)
    no_factual_claims = sum(r.get("groundedness_status") == "NO_FACTUAL_CLAIMS" for r in results)

    abstentions = sum(bool(r.get("abstained")) for r in results)
    correct_abstentions = sum(r.get("abstention_correctness") == "CORRECT" for r in results)
    incorrect_abstentions = sum(r.get("abstention_correctness") == "INCORRECT" for r in results)

    application_latencies = [float(r["application_latency_sec"]) for r in results if r.get("application_latency_sec") is not None]
    judge_latencies = [float(r["judge_latency_sec"]) for r in results if r.get("judge_latency_sec") is not None]

    total_search_calls = sum(int(r.get("search_call_count", 0)) for r in results)
    total_tool_calls = sum(int(r.get("total_tool_calls", 0)) for r in results)
    zero_tool_questions = sum(int(r.get("total_tool_calls", 0)) == 0 for r in results)
    tool_errors = sum(int(r.get("tool_errors", 0)) for r in results)
    filing_questions = [r for r in results if bool(r.get("is_filing_question", True))]
    filing_search_invocations = sum(int(r.get("search_call_count", 0)) > 0 for r in filing_questions)

    summary = {
        "questions": total,
        "company_routing_accuracy": pct(company_correct, total),
        "section_routing_accuracy": pct(section_correct, total),
        "metadata_retrieval_hit_rate": pct(metadata_hits, total),
        "evidence_retrieval_recall_at_k": pct(evidence_found, evidence_total),
        "evidence_items_found": evidence_found,
        "evidence_items_total": evidence_total,
        "evidence_question_success_rate": pct(evidence_question_successes, len(evidence_gradable)),
        "evidence_questions_graded": len(evidence_gradable),
        "answer_groundedness": pct(answer_grounded_count, len(graded_claim_answers)),
        "answers_grounded": answer_grounded_count,
        "answers_groundedness_graded": len(graded_claim_answers),
        "claim_groundedness_micro": pct(supported_claims, total_claims),
        "supported_claims": supported_claims,
        "total_factual_claims": total_claims,
        "claim_groundedness_macro": statistics.fmean(macro_claim_values) if macro_claim_values else None,
        "answer_fact_completeness": pct(facts_found, facts_total),
        "expected_facts_found": facts_found,
        "expected_facts_total": facts_total,
        "abstentions": abstentions,
        "correct_abstentions": correct_abstentions,
        "incorrect_abstentions": incorrect_abstentions,
        "judge_errors": judge_errors,
        "judge_requested": judge_requested,
        "judge_graded": len(graded_claim_answers),
        "judge_ungraded": max(judge_requested - len(graded_claim_answers), 0),
        "judge_coverage": pct(len(graded_claim_answers), judge_requested),
        "judge_error_rate": pct(judge_errors, judge_requested),
        "no_factual_claim_answers": no_factual_claims,
        "avg_search_calls_per_question": pct(total_search_calls, total),
        "total_search_calls": total_search_calls,
        "total_tool_calls": total_tool_calls,
        "zero_tool_call_questions": zero_tool_questions,
        "tool_errors": tool_errors,
        "filing_questions_search_filing_rate": pct(filing_search_invocations, len(filing_questions)),
        "application_latency_mean_sec": statistics.fmean(application_latencies) if application_latencies else None,
        "application_latency_median_sec": statistics.median(application_latencies) if application_latencies else None,
        "application_latency_p95_sec": percentile(application_latencies, 0.95),
        "judge_latency_mean_sec": statistics.fmean(judge_latencies) if judge_latencies else None,
        "judge_latency_median_sec": statistics.median(judge_latencies) if judge_latencies else None,
        "judge_latency_p95_sec": percentile(judge_latencies, 0.95),
    }
    return summary


def dataclass_list(items: Sequence[AuditedClaim]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]