import json

import pytest

from eval.metrics import (
    AuditedClaim,
    GroundednessJudgment,
    JudgeParseError,
    aggregate_results,
    audit_claim_evidence,
    call_judge_with_repair,
    classify_abstention,
    extract_agentic_search_trace,
    groundedness_from_claims,
    parse_groundedness_judgment,
    score_answer_completeness,
    score_gold_evidence,
    span_is_traceable,
)


def test_false_string_cannot_become_boolean_true():
    raw = json.dumps({"claims": [{"claim": "x", "supported": "false", "evidence_quote": ""}]})
    with pytest.raises(JudgeParseError):
        parse_groundedness_judgment(raw)


def test_malformed_judge_json_becomes_parse_error():
    with pytest.raises(JudgeParseError):
        parse_groundedness_judgment('{"claims": [')


def test_one_json_repair_retry():
    outputs = iter([
        '{"claims": [{"claim": "x", "supported": "false", "evidence_quote": ""}]}',
        '{"claims": [{"claim": "x", "supported": false, "evidence_quote": ""}]}',
    ])

    def invoke(_messages):
        return next(outputs)

    judgment, error, attempts, raw_outputs = call_judge_with_repair(
        invoke,
        [{"role": "user", "content": "judge"}],
        repair_system_prompt="repair",
    )
    assert error is None
    assert attempts == 2
    assert len(raw_outputs) == 2
    assert judgment is not None
    assert judgment.claims[0].supported is False


def test_second_invalid_judge_response_is_error():
    outputs = iter(["not json", "still not json"])

    def invoke(_messages):
        return next(outputs)

    judgment, error, attempts, _ = call_judge_with_repair(
        invoke,
        [{"role": "user", "content": "judge"}],
        repair_system_prompt="repair",
    )
    assert judgment is None
    assert error is not None
    assert attempts == 2


def test_zero_factual_claims():
    result = groundedness_from_claims([])
    assert result["status"] == "NO_FACTUAL_CLAIMS"
    assert result["claim_groundedness"] is None
    assert result["answer_grounded"] is None


def test_one_unsupported_claim_among_five():
    claims = [
        AuditedClaim(str(i), True, i < 4, "evidence", True if i < 4 else False)
        for i in range(5)
    ]
    result = groundedness_from_claims(claims)
    assert result["supported_claims"] == 4
    assert result["total_claims"] == 5
    assert result["claim_groundedness"] == pytest.approx(0.8)
    assert result["answer_grounded"] is False


def test_missing_evidence_quote_cannot_count_as_supported():
    judgment = GroundednessJudgment.model_validate(
        {"claims": [{"claim": "Revenue was $10B", "supported": True, "evidence_quote": "Revenue was $10B"}]}
    )
    audited = audit_claim_evidence(judgment, "The filing says revenue was $9B.")
    assert audited[0].judge_supported is True
    assert audited[0].supported is False
    assert audited[0].evidence_quote_valid is False


def test_evidence_quote_normalization_handles_whitespace_quotes_and_punctuation():
    context = 'Net income — attributable to Company was “$10.0 billion”.\nTotal assets: $100 billion.'
    quote = 'Net income - attributable to Company was "$10.0 billion"  Total assets $100 billion'
    assert span_is_traceable(quote, context)


def test_metadata_hit_but_gold_evidence_miss_is_possible():
    score = score_gold_evidence(
        [{"id": "assets", "text": "Total assets were 100 billion"}],
        "Johnson & Johnson Financial Statements. Total liabilities were 80 billion.",
    )
    assert score.status == "GRADED"
    assert score.found == 0
    assert score.total == 1
    assert score.question_success is False


def test_gold_evidence_hit():
    score = score_gold_evidence(
        [{"id": "assets", "text": "Total assets were $100 billion."}],
        "Selected data: Total assets were $100 billion. Other text.",
    )
    assert score.found == 1
    assert score.recall == 1.0


def test_abstention_with_evidence_available_is_incorrect():
    assert classify_abstention(True, True) == "INCORRECT"


def test_abstention_when_evidence_genuinely_unavailable_is_correct():
    assert classify_abstention(True, False) == "CORRECT"


def test_answer_completeness_counts_expected_facts_separately():
    score = score_answer_completeness(
        [
            {"id": "assets", "patterns": ["total assets were 100 billion"]},
            {"id": "liabilities", "regex_patterns": [r"liabilities\s+(?:were|of)\s+\$?80\s+billion"]},
        ],
        "Total assets were $100 billion.",
    )
    assert score.found == 1
    assert score.total == 2
    assert score.recall == 0.5


def test_aggregation_denominators_exclude_judge_errors():
    results = [
        {
            "routing_company_correct": True,
            "routing_section_correct": True,
            "metadata_retrieval_hit": True,
            "groundedness_requested": True,
            "groundedness_status": "GRADED",
            "supported_claims": 4,
            "total_claims": 5,
            "claim_groundedness": 0.8,
            "answer_grounded": False,
            "evidence_found": 1,
            "evidence_total": 1,
            "facts_found": 1,
            "facts_total": 2,
            "abstained": False,
            "search_call_count": 1,
            "total_tool_calls": 1,
            "tool_errors": 0,
            "is_filing_question": True,
            "application_latency_sec": 1.0,
            "judge_latency_sec": 0.5,
        },
        {
            "routing_company_correct": False,
            "routing_section_correct": False,
            "metadata_retrieval_hit": False,
            "groundedness_requested": True,
            "groundedness_status": "JUDGE_ERROR",
            "supported_claims": 0,
            "total_claims": 0,
            "evidence_found": 0,
            "evidence_total": 1,
            "facts_found": 0,
            "facts_total": 1,
            "abstained": True,
            "abstention_correctness": "INCORRECT",
            "search_call_count": 0,
            "total_tool_calls": 0,
            "tool_errors": 0,
            "is_filing_question": True,
            "application_latency_sec": 2.0,
            "judge_latency_sec": 0.7,
        },
    ]
    summary = aggregate_results(results)
    assert summary["claim_groundedness_micro"] == pytest.approx(0.8)
    assert summary["answer_groundedness"] == 0.0
    assert summary["answers_groundedness_graded"] == 1
    assert summary["judge_errors"] == 1
    assert summary["judge_error_rate"] == pytest.approx(0.5)
    assert summary["answer_fact_completeness"] == pytest.approx(1 / 3)


def test_agentic_context_comes_from_actual_tool_result():
    tools_invoked = [
        {
            "tool": "search_filing",
            "args": {"company": "Tesla", "section": "business"},
            "result": json.dumps({"content": "EXACT CONTEXT USED", "citations": ["Tesla citation"]}),
        },
        {
            "tool": "get_stock_price",
            "args": {"ticker": "TSLA"},
            "result": json.dumps({"data": {"latest_price": 1}}),
        },
    ]
    trace = extract_agentic_search_trace(tools_invoked)
    assert trace["context"] == "EXACT CONTEXT USED"
    assert trace["citations"] == ["Tesla citation"]
    assert trace["search_call_count"] == 1
    assert trace["total_tool_calls"] == 2


def test_supported_claim_requires_nonempty_evidence_quote():
    raw = json.dumps({"claims": [{"claim": "x", "supported": True, "evidence_quote": ""}]})
    with pytest.raises(JudgeParseError):
        parse_groundedness_judgment(raw)


def test_metadata_hit_can_coexist_with_gold_evidence_miss():
    from eval.metrics import metadata_retrieval_hit

    assert metadata_retrieval_hit(
        ["Johnson & Johnson 10-K FY2025 – Financial Statements"],
        "Johnson & Johnson",
        "Financial Statements",
    )
    evidence = score_gold_evidence(
        [{"id": "assets", "text": "Total assets were $100 billion"}],
        "The retrieved chunk discusses liabilities only.",
    )
    assert evidence.question_success is False


def test_malformed_judge_json_surfaces_as_judge_error_after_repair():
    from eval.eval import _score_groundedness

    class BadJudge:
        def invoke(self, _messages):
            class Response:
                content = "not valid json"
            return Response()

    result = _score_groundedness("q", "context", "answer", BadJudge())
    assert result["groundedness_status"] == "JUDGE_ERROR"
    assert result["judge_error"]
    assert result["judge_attempts"] == 2


def test_rule_based_capture_records_exact_returned_context_and_restores_function():
    from eval.eval import _extract_rule_based_trace, capture_rule_based_retrieval

    class FakeAgentModule:
        pass

    module = FakeAgentModule()
    calls = []

    def original(query, company, fiscal_year=None, section=None, k=4):
        calls.append((query, company, fiscal_year, section, k))
        return {
            "content": "EXACT RULE BASED CONTEXT",
            "citations": ["Tesla 10-K FY2025 – Business"],
        }

    module.query_financial_reports = original

    with capture_rule_based_retrieval(module) as trace:
        returned = module.query_financial_reports(
            query="q", company="tesla", fiscal_year=None, section="business"
        )
        assert returned["content"] == "EXACT RULE BASED CONTEXT"
        assert len(calls) == 1

    assert module.query_financial_reports is original
    extracted = _extract_rule_based_trace(trace)
    assert extracted["context"] == "EXACT RULE BASED CONTEXT"
    assert extracted["citations"] == ["Tesla 10-K FY2025 – Business"]
    assert extracted["retrieval_call_count"] == 1