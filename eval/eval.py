"""Auditable evaluation harness for Sovereign Financial Analyst.

The benchmark separates four questions that must not be conflated:

1. Routing / metadata hit: did the application search the expected company and
   section, and did returned citation metadata match those labels?
2. Evidence retrieval recall@k: did the exact generation context contain the
   manually annotated gold evidence spans needed to answer the question?
3. Groundedness / faithfulness: are the factual claims in the answer supported
   by the exact context used during that same generation run?
4. Answer completeness: did the answer contain the manually annotated expected
   facts, scored deterministically with literal/regex patterns?

Gold annotations are never invented by this harness. Items without gold evidence
or expected facts are reported as UNGRADED for those metrics.

Examples:
    python eval/eval.py --mode rule_based
    python eval/eval.py --mode agentic
    python eval/eval.py --mode agentic --limit 3
    python eval/eval.py --mode agentic --runs 3
    python eval/eval.py --mode agentic --skip-groundedness
    python eval/eval.py --dataset-module dataset_test --mode agentic
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from metrics import (  # noqa: E402
    aggregate_results,
    audit_claim_evidence,
    call_judge_with_repair,
    classify_abstention,
    dataclass_list,
    detect_abstention,
    extract_agentic_search_trace,
    groundedness_from_claims,
    GroundednessJudgment,
    metadata_retrieval_hit,
    score_answer_completeness,
    score_gold_evidence,
)


JUDGE_SYSTEM_PROMPT = """You are a strict claim-level faithfulness auditor.

You receive QUESTION, CONTEXT, and ANSWER.

Rules:
- Use ONLY CONTEXT. Do not use outside knowledge.
- Extract every meaningful factual claim actually made in ANSWER. Split compound
  statements into atomic claims when they could be supported independently.
- Ignore headings, source labels/citations, pure formatting, and non-factual
  conversational language.
- For each factual claim, set supported=true only if CONTEXT supports it.
- If supported=true, evidence_quote MUST be a verbatim quote copied from CONTEXT.
  Keep the quote as short as possible while still supporting the claim.
- If supported=false, evidence_quote MUST be an empty string.
- Do not grade completeness. Missing facts are handled by a separate metric.
- If ANSWER has no meaningful factual claims, return an empty claims list.

Return ONLY valid JSON in exactly this shape:
{"claims":[{"claim":"...","supported":true,"evidence_quote":"..."}]}

The supported field must be a JSON boolean, never a quoted string.
"""

JUDGE_REPAIR_PROMPT = """You repair JSON produced by a claim-level evaluator.
Return ONLY JSON matching this exact schema:
{"claims":[{"claim":"string","supported":true,"evidence_quote":"string"}]}
The supported field must be a JSON boolean. Do not add or remove substantive
claims or change verdict meaning; repair only syntax, missing required fields,
or invalid JSON types when the intended value is unambiguous.
"""


# ---------------------------------------------------------------------------
# Loading and reproducibility metadata
# ---------------------------------------------------------------------------


def _load_dataset(module_name: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    module = importlib.import_module(module_name)
    questions = list(getattr(module, "EVAL_QUESTIONS"))
    metadata = {
        "dataset_name": getattr(module, "DATASET_NAME", module_name),
        "dataset_version": getattr(module, "DATASET_VERSION", "unversioned"),
        "dataset_split": getattr(module, "DATASET_SPLIT", "dev"),
    }
    return questions, metadata


def _load_app_components() -> dict[str, Any]:
    import app.agent as agent_module
    import app.agentic_router as agentic_module
    import app.config as app_config
    from app.companies import SECTION_DISPLAY_MAP, SLUG_TO_DISPLAY
    from app.tools import normalize_company, normalize_section, query_financial_reports

    production_k = inspect.signature(query_financial_reports).parameters["k"].default
    generation_model = (
        app_config.GROQ_MODEL if app_config.LLM_PROVIDER == "groq" else app_config.OLLAMA_MODEL
    )
    return {
        "agent_module": agent_module,
        "agentic_module": agentic_module,
        "app_config": app_config,
        "SLUG_TO_DISPLAY": SLUG_TO_DISPLAY,
        "SECTION_DISPLAY_MAP": SECTION_DISPLAY_MAP,
        "normalize_company": normalize_company,
        "normalize_section": normalize_section,
        "production_k": production_k,
        "generation_provider": app_config.LLM_PROVIDER,
        "generation_model": generation_model,
        "embedding_model": app_config.EMBED_MODEL,
    }


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                check=True,
                text=True,
                capture_output=True,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    sha = run("rev-parse", "HEAD") or os.getenv("GIT_COMMIT_SHA")
    status = run("status", "--porcelain")
    return {
        "git_commit_sha": sha,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def _judge_metadata(app: dict[str, Any]) -> dict[str, str]:
    """Resolve the judge provider without overriding the app's provider choice.

    JUDGE_LLM_PROVIDER is an optional explicit override. If it is unset, the
    evaluator uses the same LLM_PROVIDER/model configured for the application.
    Merely having a GROQ_API_KEY must never switch the judge to Groq.
    """
    app_config = app["app_config"]
    provider = os.getenv("JUDGE_LLM_PROVIDER", app_config.LLM_PROVIDER).strip().lower()
    if provider not in {"groq", "ollama", "anthropic"}:
        raise ValueError(
            f"Unknown JUDGE_LLM_PROVIDER={provider!r}. Expected 'groq', 'ollama', or 'anthropic'."
        )

    if provider == "groq":
        model = os.getenv("JUDGE_GROQ_MODEL", app_config.GROQ_MODEL)
    elif provider == "anthropic":
        model = os.getenv("JUDGE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    else:
        model = os.getenv("JUDGE_OLLAMA_MODEL", app_config.OLLAMA_MODEL)
    return {
        "judge_provider": provider,
        "judge_model": model,
        "judge_output_mode": "ollama_json_schema" if provider == "ollama" else "prompt_json",
    }


class _JudgeClient:
    """Eval-only judge client.

    Ollama uses its native JSON-schema `format` capability so malformed JSON is
    constrained at generation time. Groq keeps the existing LangChain path.
    """

    def __init__(self, app: dict[str, Any]):
        meta = _judge_metadata(app)
        self.provider = meta["judge_provider"]
        self.model = meta["judge_model"]
        self.base_url = app["app_config"].OLLAMA_BASE_URL.rstrip("/")
        self.timeout_sec = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "180"))
        self.llm = None

        if self.provider == "groq":
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model=self.model, temperature=0)
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=self.model, temperature=0)

    def invoke_text(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "ollama":
            return self._invoke_ollama_structured(messages)

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _invoke_ollama_structured(self, messages: list[dict[str, str]]) -> str:
        # Ollama accepts a JSON schema in `format`; using the exact Pydantic
        # schema prevents the common truncated/fenced/malformed JSON failures
        # seen when a small local model is only prompted to "return JSON".
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": GroundednessJudgment.model_json_schema(),
            "options": {
                "temperature": 0,
                "num_predict": int(os.getenv("JUDGE_NUM_PREDICT", "4096")),
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama judge HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama judge unavailable: {exc.reason}") from exc

        content = body.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama judge returned no message content: {body!r}")
        return content


def _build_judge_llm(app: dict[str, Any]) -> _JudgeClient:
    return _JudgeClient(app)


def _invoke_judge_text(judge: Any, messages: list[dict[str, str]]) -> str:
    """Invoke either the eval JudgeClient or a LangChain-like test/fallback model."""
    if hasattr(judge, "invoke_text"):
        return judge.invoke_text(messages)
    response = judge.invoke(messages)
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Exact generation-context capture
# ---------------------------------------------------------------------------


@contextmanager
def capture_rule_based_retrieval(agent_module: Any) -> Iterator[list[dict[str, Any]]]:
    """Evaluation-only instrumentation around the exact retrieval call.

    app.agent imports query_financial_reports into its module namespace. Replacing
    only that reference for the duration of one ask_agent call lets the harness
    capture the exact returned content without a second retrieval and without any
    production-code change or behavior change.
    """

    original = agent_module.query_financial_reports
    signature = inspect.signature(original)
    trace: list[dict[str, Any]] = []

    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        call_args = dict(bound.arguments)
        started = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            trace.append(
                {
                    "args": call_args,
                    "error": repr(exc),
                    "latency_sec": time.perf_counter() - started,
                }
            )
            raise
        trace.append(
            {
                "args": call_args,
                "result": result,
                "latency_sec": time.perf_counter() - started,
            }
        )
        return result

    agent_module.query_financial_reports = wrapped
    try:
        yield trace
    finally:
        agent_module.query_financial_reports = original


def _extract_rule_based_trace(trace: Sequence[dict[str, Any]]) -> dict[str, Any]:
    contexts: list[str] = []
    citations: list[str] = []
    errors = 0
    for call in trace:
        if call.get("error"):
            errors += 1
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        content = result.get("content")
        if isinstance(content, str) and content:
            contexts.append(content)
        result_citations = result.get("citations", [])
        if isinstance(result_citations, list):
            citations.extend(str(c) for c in result_citations)
    return {
        "context": "\n\n".join(contexts),
        "citations": citations,
        "retrieval_calls": list(trace),
        "retrieval_call_count": len(trace),
        "tool_errors": errors,
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _routing_from_calls(
    calls: Sequence[dict[str, Any]],
    expected_company: str,
    expected_section: str,
    app: dict[str, Any],
    *,
    agentic: bool,
) -> tuple[list[dict[str, Any]], bool, bool]:
    routes: list[dict[str, Any]] = []
    for call in calls:
        args = call.get("args", {}) if agentic else call.get("args", {})
        company = app["normalize_company"](args.get("company"))
        section = app["normalize_section"](args.get("section"))
        routes.append(
            {
                "raw_company": args.get("company"),
                "company": company,
                "raw_section": args.get("section"),
                "section": section,
                "fiscal_year": args.get("fiscal_year"),
                "query": args.get("question") if agentic else args.get("query"),
            }
        )

    if not routes:
        return routes, False, False

    company_correct = all(route["company"] == expected_company for route in routes)
    section_correct = all(route["section"] == expected_section for route in routes)
    return routes, company_correct, section_correct


def _effective_answerable(item: dict[str, Any]) -> bool | None:
    if "answerable" in item:
        value = item["answerable"]
        return value if isinstance(value, bool) else None
    if item.get("gold_evidence"):
        return True
    return None


def _score_groundedness(
    question: str,
    context: str,
    answer: str,
    judge_llm: Any,
) -> dict[str, Any]:
    user_prompt = f"""QUESTION:\n{question}\n\nCONTEXT:\n{context or '(no context)'}\n\nANSWER:\n{answer}\n"""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    started = time.perf_counter()
    try:
        judgment, error, attempts, raw_outputs = call_judge_with_repair(
            lambda msgs: _invoke_judge_text(judge_llm, msgs),
            messages,
            repair_system_prompt=JUDGE_REPAIR_PROMPT,
        )
    except Exception as exc:
        return {
            "groundedness_status": "JUDGE_ERROR",
            "judge_error": f"judge invocation failed: {exc!r}",
            "judge_attempts": 0,
            "judge_raw_outputs": [],
            "judge_latency_sec": time.perf_counter() - started,
            "claims": [],
            "supported_claims": 0,
            "total_claims": 0,
            "claim_groundedness": None,
            "answer_grounded": None,
            "unsupported_claims": [],
        }

    latency = time.perf_counter() - started
    if judgment is None:
        return {
            "groundedness_status": "JUDGE_ERROR",
            "judge_error": error,
            "judge_attempts": attempts,
            "judge_raw_outputs": raw_outputs,
            "judge_latency_sec": latency,
            "claims": [],
            "supported_claims": 0,
            "total_claims": 0,
            "claim_groundedness": None,
            "answer_grounded": None,
            "unsupported_claims": [],
        }

    audited = audit_claim_evidence(judgment, context)
    score = groundedness_from_claims(audited)
    return {
        "groundedness_status": score["status"],
        "judge_error": None,
        "judge_attempts": attempts,
        "judge_raw_outputs": raw_outputs,
        "judge_latency_sec": latency,
        "claims": dataclass_list(audited),
        "supported_claims": score["supported_claims"],
        "total_claims": score["total_claims"],
        "claim_groundedness": score["claim_groundedness"],
        "answer_grounded": score["answer_grounded"],
        "unsupported_claims": score["unsupported_claims"],
    }


def _empty_groundedness(status: str) -> dict[str, Any]:
    return {
        "groundedness_status": status,
        "judge_error": None,
        "judge_attempts": 0,
        "judge_raw_outputs": [],
        "judge_latency_sec": None,
        "claims": [],
        "supported_claims": 0,
        "total_claims": 0,
        "claim_groundedness": None,
        "answer_grounded": None,
        "unsupported_claims": [],
    }


# ---------------------------------------------------------------------------
# One question / one run
# ---------------------------------------------------------------------------


def evaluate_question(
    item: dict[str, Any],
    mode: str,
    app: dict[str, Any],
    judge_llm: Any | None,
    groundedness_enabled: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    answer = ""
    context = ""
    citations: list[str] = []
    actual_routing: list[dict[str, Any]] = []
    retrieval_call_count = 0
    search_call_count = 0
    total_tool_calls = 0
    tool_errors = 0
    application_error: str | None = None
    tool_calls: list[dict[str, Any]] = []

    try:
        if mode == "agentic":
            agentic_result = app["agentic_module"].run_agentic_query(item["question"])
            answer = str(agentic_result.get("answer", ""))
            tool_calls = list(agentic_result.get("tools_invoked", []))
            trace = extract_agentic_search_trace(tool_calls)
            context = trace["context"]
            citations = trace["citations"]
            search_call_count = trace["search_call_count"]
            total_tool_calls = trace["total_tool_calls"]
            tool_errors = trace["tool_errors"]
            actual_routing, company_correct, section_correct = _routing_from_calls(
                trace["search_calls"],
                item["expected_company"],
                item["expected_section"],
                app,
                agentic=True,
            )
        else:
            with capture_rule_based_retrieval(app["agent_module"]) as raw_trace:
                answer, _active_company = app["agent_module"].ask_agent(item["question"])
            trace = _extract_rule_based_trace(raw_trace)
            context = trace["context"]
            citations = trace["citations"]
            retrieval_call_count = trace["retrieval_call_count"]
            tool_errors = trace["tool_errors"]
            actual_routing, company_correct, section_correct = _routing_from_calls(
                raw_trace,
                item["expected_company"],
                item["expected_section"],
                app,
                agentic=False,
            )
    except Exception as exc:
        company_correct = False
        section_correct = False
        application_error = repr(exc)

    application_latency = time.perf_counter() - started

    expected_company_display = app["SLUG_TO_DISPLAY"].get(
        item["expected_company"], item["expected_company"]
    )
    expected_section_display = app["SECTION_DISPLAY_MAP"].get(
        item["expected_section"], item["expected_section"]
    )
    metadata_hit = metadata_retrieval_hit(
        citations,
        expected_company_display,
        expected_section_display,
    )

    evidence_score = score_gold_evidence(item.get("gold_evidence"), context)
    completeness = score_answer_completeness(item.get("expected_facts"), answer)
    abstained, abstention_phrase = detect_abstention(answer)
    answerable = _effective_answerable(item)
    abstention_correctness = classify_abstention(abstained, answerable)

    if groundedness_enabled and judge_llm is not None and not application_error:
        groundedness = _score_groundedness(item["question"], context, answer, judge_llm)
    elif groundedness_enabled and application_error:
        groundedness = _empty_groundedness("SKIPPED")
    else:
        groundedness = _empty_groundedness("SKIPPED")

    result = {
        "id": item["id"],
        "question": item["question"],
        "category": item.get("category"),
        "expected_company": item["expected_company"],
        "expected_section": item["expected_section"],
        "actual_routing": actual_routing,
        "routing_company_correct": company_correct,
        "routing_section_correct": section_correct,
        "retrieved_citations": citations,
        "metadata_retrieval_hit": metadata_hit,
        "gold_evidence": item.get("gold_evidence", []),
        "evidence_status": evidence_score.status,
        "evidence_found": evidence_score.found,
        "evidence_total": evidence_score.total,
        "evidence_retrieval_recall_at_k": evidence_score.recall,
        "evidence_question_success": evidence_score.question_success,
        "retrieved_gold_evidence_match": evidence_score.matches,
        "generation_context": context,
        "answer": answer,
        "expected_facts": item.get("expected_facts", []),
        "completeness_status": completeness.status,
        "facts_found": completeness.found,
        "facts_total": completeness.total,
        "answer_fact_completeness": completeness.recall,
        "completeness_result": completeness.matches,
        "answerable": answerable,
        "abstained": abstained,
        "abstention_phrase": abstention_phrase,
        "abstention_correctness": abstention_correctness,
        "tool_calls": tool_calls,
        "search_call_count": search_call_count,
        "retrieval_call_count": retrieval_call_count,
        "total_tool_calls": total_tool_calls,
        "tool_errors": tool_errors,
        "is_filing_question": item.get("is_filing_question", True),
        "application_latency_sec": application_latency,
        "application_error": application_error,
        "groundedness_requested": groundedness_enabled,
        **groundedness,
    }
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_pct(value: float | None) -> str:
    return "UNGRADED" if value is None else f"{value * 100:.1f}%"


def _fmt_sec(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def print_question_result(result: dict[str, Any]) -> None:
    evidence = (
        "UNGRADED"
        if result["evidence_total"] == 0
        else f"{result['evidence_found']}/{result['evidence_total']} "
        f"({_fmt_pct(result['evidence_retrieval_recall_at_k'])})"
    )
    status = result["groundedness_status"]
    if status == "GRADED":
        grounded = (
            f"{result['supported_claims']}/{result['total_claims']} "
            f"({_fmt_pct(result['claim_groundedness'])})"
        )
    elif status == "NO_FACTUAL_CLAIMS":
        grounded = "NO_FACTUAL_CLAIMS"
    elif status == "JUDGE_ERROR":
        grounded = "JUDGE_ERROR"
    else:
        grounded = "SKIPPED"

    completeness = (
        "UNGRADED"
        if result["facts_total"] == 0
        else f"{result['facts_found']}/{result['facts_total']} "
        f"({_fmt_pct(result['answer_fact_completeness'])})"
    )
    print(
        f"[{result['id']}] metadata_hit={result['metadata_retrieval_hit']} "
        f"evidence={evidence} groundedness={grounded} completeness={completeness} "
        f"latency={result['application_latency_sec']:.2f}s"
    )
    if result.get("application_error"):
        print(f"  application_error={result['application_error']}")
    if result.get("judge_error"):
        print(f"  judge_error={result['judge_error']}")


def print_summary(summary: dict[str, Any], mode: str) -> None:
    print("\n" + "=" * 72)
    print(f"SUMMARY ({mode})")
    print("=" * 72)
    print(f"Questions:                    {summary['questions']}")
    print(f"Company routing accuracy:     {_fmt_pct(summary['company_routing_accuracy'])}")
    print(f"Section routing accuracy:     {_fmt_pct(summary['section_routing_accuracy'])}")
    print("")
    print(f"Metadata retrieval hit:       {_fmt_pct(summary['metadata_retrieval_hit_rate'])}")
    print(
        f"Evidence retrieval recall@k:  {_fmt_pct(summary['evidence_retrieval_recall_at_k'])} "
        f"({summary['evidence_items_found']}/{summary['evidence_items_total']} gold evidence items)"
    )
    print(
        f"Evidence question success:    {_fmt_pct(summary['evidence_question_success_rate'])} "
        f"({summary['evidence_questions_graded']} graded questions)"
    )
    print("")
    print(
        f"Answer groundedness:          {_fmt_pct(summary['answer_groundedness'])} "
        f"({summary['answers_grounded']}/{summary['answers_groundedness_graded']} graded answers)"
    )
    print(
        f"Claim groundedness:           {_fmt_pct(summary['claim_groundedness_micro'])} "
        f"({summary['supported_claims']}/{summary['total_factual_claims']} factual claims)"
    )
    print(
        f"Answer fact completeness:     {_fmt_pct(summary['answer_fact_completeness'])} "
        f"({summary['expected_facts_found']}/{summary['expected_facts_total']} expected facts)"
    )
    print("")
    print(f"Abstentions:                  {summary['abstentions']}")
    print(f"Correct abstentions:          {summary['correct_abstentions']}")
    print(f"Incorrect abstentions:        {summary['incorrect_abstentions']}")
    print(f"Questions groundedness graded:{summary['judge_graded']}")
    print(f"Questions ungraded:           {summary['judge_ungraded']}")
    print(f"Judge errors:                 {summary['judge_errors']}")
    print(f"Judge error rate:             {_fmt_pct(summary['judge_error_rate'])}")
    print(f"Judge coverage:               {_fmt_pct(summary['judge_coverage'])}")
    if mode == "agentic":
        avg_search = summary["avg_search_calls_per_question"]
        avg_search_text = "n/a" if avg_search is None else f"{avg_search:.2f}"
        print("")
        print(f"Avg search calls/question:    {avg_search_text}")
        print(f"Total tool calls:             {summary['total_tool_calls']}")
        print(f"Zero-tool-call questions:     {summary['zero_tool_call_questions']}")
        print(f"Tool errors:                  {summary['tool_errors']}")
        print(
            f"Filing questions using search:{_fmt_pct(summary['filing_questions_search_filing_rate'])}"
        )
    print("")
    print(f"Application latency mean:     {_fmt_sec(summary['application_latency_mean_sec'])}")
    print(f"Application latency median:   {_fmt_sec(summary['application_latency_median_sec'])}")
    print(f"Application latency p95:      {_fmt_sec(summary['application_latency_p95_sec'])}")
    print(f"Judge latency mean:           {_fmt_sec(summary['judge_latency_mean_sec'])}")
    print(f"Judge latency median:         {_fmt_sec(summary['judge_latency_median_sec'])}")
    print(f"Judge latency p95:            {_fmt_sec(summary['judge_latency_p95_sec'])}")
    print("=" * 72)


def _markdown_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        f"# Sovereign Financial Analyst — {metadata['dataset_split']} evaluation",
        "",
        "## Methodology",
        "",
        "- **Metadata retrieval hit** checks expected company + section labels in returned citation metadata. It is not retrieval recall.",
        "- **Evidence retrieval recall@k** is micro-averaged required gold evidence spans found in the exact generation context / total required spans.",
        "- **Claim groundedness** is micro-averaged supported factual claims / total factual claims. A judge-supported claim is downgraded if its evidence quote cannot be traced to the exact context.",
        "- **Answer groundedness** is the fraction of successfully graded factual answers for which every factual claim is supported.",
        "- **Answer fact completeness** is expected facts detected in the answer / total expected facts; it is separate from faithfulness.",
        "- Missing gold annotations and judge failures are **UNGRADED**, not converted to 0 or 1.",
        "",
        "## Reproducibility",
        "",
        f"- Benchmark run ID: `{metadata['benchmark_run_id']}`",
        f"- Git commit: `{metadata.get('git_commit_sha') or 'unavailable'}`",
        f"- Mode: `{metadata['mode']}`",
        f"- k: `{metadata['k']}`",
        f"- Generation model: `{metadata['generation_provider']} / {metadata['generation_model']}`",
        f"- Judge model: `{metadata['judge_provider']} / {metadata['judge_model']}`",
        f"- Embedding model: `{metadata['embedding_model']}`",
        f"- Dataset: `{metadata['dataset_name']} {metadata['dataset_version']}` ({metadata['dataset_split']})",
        f"- Runs: `{metadata['runs']}`",
        "",
    ]

    for run in report["runs"]:
        s = run["summary"]
        lines += [
            f"## Run {run['run_number']}",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Company routing accuracy | {_fmt_pct(s['company_routing_accuracy'])} |",
            f"| Section routing accuracy | {_fmt_pct(s['section_routing_accuracy'])} |",
            f"| Metadata retrieval hit | {_fmt_pct(s['metadata_retrieval_hit_rate'])} |",
            f"| Evidence retrieval recall@k | {_fmt_pct(s['evidence_retrieval_recall_at_k'])} ({s['evidence_items_found']}/{s['evidence_items_total']}) |",
            f"| Answer groundedness | {_fmt_pct(s['answer_groundedness'])} ({s['answers_grounded']}/{s['answers_groundedness_graded']}) |",
            f"| Claim groundedness (micro) | {_fmt_pct(s['claim_groundedness_micro'])} ({s['supported_claims']}/{s['total_factual_claims']}) |",
            f"| Answer fact completeness | {_fmt_pct(s['answer_fact_completeness'])} ({s['expected_facts_found']}/{s['expected_facts_total']}) |",
            f"| Abstentions | {s['abstentions']} |",
            f"| Groundedness questions graded | {s['judge_graded']} |",
            f"| Groundedness questions ungraded | {s['judge_ungraded']} |",
            f"| Judge errors | {s['judge_errors']} |",
            f"| Judge coverage | {_fmt_pct(s['judge_coverage'])} |",
            f"| Application median latency | {_fmt_sec(s['application_latency_median_sec'])} |",
            f"| Application p95 latency | {_fmt_sec(s['application_latency_p95_sec'])} |",
            "",
            "### Failures / ungraded items",
            "",
        ]
        failures = [
            r for r in run["results"]
            if (
                not r["metadata_retrieval_hit"]
                or r["evidence_status"] != "GRADED"
                or r["evidence_question_success"] is False
                or r["groundedness_status"] in {"JUDGE_ERROR"}
                or r["answer_grounded"] is False
                or r["completeness_status"] != "GRADED"
                or (r["answer_fact_completeness"] is not None and r["answer_fact_completeness"] < 1.0)
                or r["abstention_correctness"] == "INCORRECT"
            )
        ]
        if not failures:
            lines.append("None.")
        for r in failures:
            reasons: list[str] = []
            if not r["metadata_retrieval_hit"]:
                reasons.append("metadata retrieval miss")
            if r["evidence_status"] != "GRADED":
                reasons.append("evidence UNGRADED")
            elif r["evidence_question_success"] is False:
                reasons.append("gold evidence miss")
            if r["groundedness_status"] == "JUDGE_ERROR":
                reasons.append("JUDGE_ERROR")
            elif r["answer_grounded"] is False:
                reasons.append("unsupported factual claim(s)")
            if r["completeness_status"] != "GRADED":
                reasons.append("completeness UNGRADED")
            elif r["answer_fact_completeness"] is not None and r["answer_fact_completeness"] < 1.0:
                reasons.append("incomplete answer")
            if r["abstention_correctness"] == "INCORRECT":
                reasons.append("incorrect abstention")
            lines.append(f"- **{r['id']}**: {', '.join(reasons)} — {r['question']}")
        lines.append("")

    if report.get("across_runs"):
        lines += ["## Across-run variability", ""]
        for name, values in report["across_runs"].items():
            mean = values.get("mean")
            std = values.get("std")
            lines.append(
                f"- `{name}`: mean={mean:.4f}, std={std:.4f}, n={values['n']}"
                if mean is not None
                else f"- `{name}`: UNGRADED"
            )
        lines.append("")

    return "\n".join(lines)


def _across_run_stats(runs: Sequence[dict[str, Any]], mode: str) -> dict[str, dict[str, Any]]:
    metrics = {
        "answer_groundedness",
        "claim_groundedness_micro",
        "answer_fact_completeness",
        "evidence_retrieval_recall_at_k",
        "metadata_retrieval_hit_rate",
        "judge_error_rate",
    }
    if mode == "agentic":
        metrics |= {
            "company_routing_accuracy",
            "section_routing_accuracy",
            "avg_search_calls_per_question",
        }

    output: dict[str, dict[str, Any]] = {}
    for metric in sorted(metrics):
        values = [run["summary"].get(metric) for run in runs]
        numeric = [float(v) for v in values if v is not None]
        output[metric] = {
            "mean": statistics.fmean(numeric) if numeric else None,
            "std": statistics.pstdev(numeric) if len(numeric) > 1 else (0.0 if numeric else None),
            "n": len(numeric),
        }
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rule_based", "agentic"], default="rule_based")
    parser.add_argument("--dataset-module", default="dataset")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--skip-groundedness", action="store_true")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Compatibility check only. The harness evaluates the production retrieval k and will "
            "reject a different value rather than silently changing retrieval for the benchmark."
        ),
    )
    parser.add_argument("--id", help="Evaluate one question by ID")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    questions, dataset_meta = _load_dataset(args.dataset_module)
    if args.id:
        questions = [q for q in questions if q.get("id") == args.id]
    if args.category:
        questions = [q for q in questions if q.get("category") == args.category]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("No evaluation questions matched the requested filters.")

    app = _load_app_components()
    production_k = int(app["production_k"])
    if args.k is not None and args.k != production_k:
        raise SystemExit(
            f"Refusing --k {args.k}: production query_financial_reports currently uses k={production_k}. "
            "This harness does not alter retrieval settings merely to change benchmark scores."
        )

    benchmark_run_id = str(uuid.uuid4())
    groundedness_enabled = not args.skip_groundedness
    judge_llm = _build_judge_llm(app) if groundedness_enabled else None

    default_json = EVAL_DIR / ("report_agentic.json" if args.mode == "agentic" else "report.json")
    out_path = args.out or default_json
    md_path = args.markdown or out_path.with_suffix(".md")

    git_meta = _git_metadata()
    judge_meta = _judge_metadata(app)
    report_metadata = {
        **git_meta,
        **dataset_meta,
        **judge_meta,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_run_id": benchmark_run_id,
        "mode": args.mode,
        "k": production_k,
        "generation_provider": app["generation_provider"],
        "generation_model": app["generation_model"],
        "embedding_model": app["embedding_model"],
        "number_of_questions": len(questions),
        "groundedness_enabled": groundedness_enabled,
        "runs": args.runs,
    }

    missing_evidence = sum(not q.get("gold_evidence") for q in questions)
    missing_facts = sum(not q.get("expected_facts") for q in questions)
    if missing_evidence or missing_facts:
        print(
            "Gold-label coverage warning: "
            f"{missing_evidence}/{len(questions)} questions lack gold_evidence; "
            f"{missing_facts}/{len(questions)} lack expected_facts. "
            "Those metrics will be UNGRADED rather than guessed."
        )

    runs: list[dict[str, Any]] = []
    for run_number in range(1, args.runs + 1):
        print(f"\nRun {run_number}/{args.runs} — mode={args.mode}, k={production_k}")
        results: list[dict[str, Any]] = []
        for item in questions:
            result = evaluate_question(
                item,
                args.mode,
                app,
                judge_llm,
                groundedness_enabled,
            )
            results.append(result)
            print_question_result(result)

        summary = aggregate_results(results)
        print_summary(summary, args.mode)
        runs.append({"run_number": run_number, "summary": summary, "results": results})

    report = {
        "metadata": {
            **report_metadata,
            "judge_error_count": sum(run["summary"]["judge_errors"] for run in runs),
        },
        "runs": runs,
        "across_runs": _across_run_stats(runs, args.mode) if args.runs > 1 else None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"\nJSON report: {out_path}")
    print(f"Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())