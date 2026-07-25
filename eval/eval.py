"""
Runs the labeled question set in eval/dataset.py end-to-end through the real
pipeline (app.agent / app.tools) and reports two numbers you can actually
quote in an interview:

  1. Retrieval recall@k  -- for each question, does the pipeline's own
     company/section routing (extract_company / infer_section in agent.py)
     land on the right (company, section), and does the vector search under
     that filter come back with at least one chunk tagged with the expected
     company + section? This exercises the exact routing logic that's
     currently keyword/regex-based, so it doubles as a routing-accuracy
     check -- the same brittle logic flagged in the roadmap.

  2. Groundedness (faithfulness) -- for each question, does the agent's final
     synthesized answer only contain claims supported by the context that was
     actually retrieved for it? Scored with an LLM-as-judge pass using the
     same local Ollama model the app already uses, so it costs nothing extra
     to run. This is reference-free: it doesn't require a hand-written gold
     answer, it just checks the answer against its own retrieved evidence.

Usage (run from the repo root, with your Chroma DB already ingested and
Ollama running -- same requirements as running the Streamlit app):

    python eval/eval.py                        # full run, both metrics
    python eval/eval.py --skip-groundedness     # retrieval only, fast
    python eval/eval.py --category risk_factors # subset by section
    python eval/eval.py --limit 5               # quick smoke test
    python eval/eval.py --k 6                   # override retrieval k
    python eval/eval.py --out eval/report.json  # custom report path

Output: a summary printed to stdout, plus a JSON report (default
eval/report.json) and a Markdown report (default eval/report.md) with
per-question detail suitable for pasting into a README or a PR description.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agent import extract_company, infer_section, ask_agent
from app.agentic_router import run_agentic_query
from app.tools import query_financial_reports, normalize_company
from app.companies import SLUG_TO_DISPLAY, SECTION_DISPLAY_MAP

from dataset import EVAL_QUESTIONS

JUDGE_LLM_PROVIDER = os.getenv("JUDGE_LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "ollama")
JUDGE_OLLAMA_MODEL = os.getenv("JUDGE_OLLAMA_MODEL", "llama3.1")
JUDGE_GROQ_MODEL = os.getenv("JUDGE_GROQ_MODEL", "llama-3.1-8b-instant")


def _build_judge_llm():
    if JUDGE_LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=JUDGE_GROQ_MODEL, temperature=0)
    from langchain_ollama import ChatOllama
    from app.config import OLLAMA_BASE_URL

    return ChatOllama(model=JUDGE_OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


judge_llm = _build_judge_llm()  


# --------------------------------------------------------------------------- data


@dataclass
class QuestionResult:
    id: str
    question: str
    category: str
    expected_company: str
    expected_section: str

    routed_company: Optional[str] = None
    routed_section: Optional[str] = None
    routing_company_correct: bool = False
    routing_section_correct: bool = False

    retrieval_hit: bool = False
    retrieval_citations: list[str] = field(default_factory=list)
    retrieval_note: str = ""

    groundedness_run: bool = False
    grounded: Optional[bool] = None
    unsupported_claims: list[str] = field(default_factory=list)
    judge_reasoning: str = ""
    judge_error: Optional[str] = None

    answer: str = ""
    latency_sec: float = 0.0
    tool_calls_count: int = 0  # agentic mode only; always 0 for rule_based


# --------------------------------------------------------------------- retrieval


def evaluate_retrieval(item: dict, k: int) -> QuestionResult:
    result = QuestionResult(
        id=item["id"],
        question=item["question"],
        category=item["category"],
        expected_company=item["expected_company"],
        expected_section=item["expected_section"],
    )

    # Use the agent's own routing functions -- this is the exact keyword/regex
    # logic a real user query goes through, not a hand-fed filter.
    routed_company = extract_company(item["question"])
    routed_section = infer_section(item["question"])

    result.routed_company = routed_company
    result.routed_section = routed_section
    result.routing_company_correct = routed_company == item["expected_company"]
    result.routing_section_correct = routed_section == item["expected_section"]

    if routed_company is None:
        result.retrieval_note = "routing failed to identify a company; no retrieval attempted"
        return result

    filing_result = query_financial_reports(
        query=item["question"],
        company=routed_company,
        fiscal_year=None,
        section=routed_section,
        k=k,
    )

    citations = filing_result.get("citations", [])
    result.retrieval_citations = citations

    expected_company_display = SLUG_TO_DISPLAY.get(item["expected_company"], item["expected_company"])
    expected_section_display = SECTION_DISPLAY_MAP.get(item["expected_section"], item["expected_section"])

    hit = any(
        expected_company_display.lower() in c.lower() and expected_section_display.lower() in c.lower()
        for c in citations
    )
    result.retrieval_hit = hit

    if not citations:
        result.retrieval_note = "no chunks retrieved under routed filter (possible coverage gap)"
    elif not hit:
        result.retrieval_note = "retrieved chunks did not match expected company/section"

    return result


# --------------------------------------------------------------------- agentic


def evaluate_agentic(item: dict, k: int, run_groundedness: bool) -> QuestionResult:
    result = QuestionResult(
        id=item["id"],
        question=item["question"],
        category=item["category"],
        expected_company=item["expected_company"],
        expected_section=item["expected_section"],
    )

    start = time.time()
    try:
        agentic_result = run_agentic_query(item["question"])
    except Exception as exc:  
        result.answer = ""
        result.judge_error = f"run_agentic_query raised: {exc!r}"
        result.groundedness_run = run_groundedness
        result.latency_sec = time.time() - start
        result.retrieval_note = "agentic call failed; no retrieval attempted"
        return result

    result.answer = agentic_result["answer"]
    result.latency_sec = time.time() - start
    result.tool_calls_count = len(agentic_result["tools_invoked"])

    search_calls = [t for t in agentic_result["tools_invoked"] if t["tool"] == "search_filing"]

    if not search_calls:
        result.retrieval_note = "agent made no search_filing call; no retrieval attempted"
        return result

    first_call = search_calls[0]
    routed_company = normalize_company(first_call["args"].get("company"))
    routed_section = first_call["args"].get("section")

    result.routed_company = routed_company
    result.routed_section = routed_section
    result.routing_company_correct = routed_company == item["expected_company"]
    result.routing_section_correct = (
        routed_section == item["expected_section"] if routed_section else False
    )

    # Pull citations + retrieved text straight from the tool result already
    # produced during the agent's own run -- no second retrieval call needed.
    citations: list[str] = []
    context_parts: list[str] = []
    for call in search_calls:
        try:
            parsed = json.loads(call["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        citations.extend(parsed.get("citations", []))
        content = parsed.get("content", "")
        if content:
            context_parts.append(content)

    result.retrieval_citations = citations

    expected_company_display = SLUG_TO_DISPLAY.get(item["expected_company"], item["expected_company"])
    expected_section_display = SECTION_DISPLAY_MAP.get(item["expected_section"], item["expected_section"])

    hit = any(
        expected_company_display.lower() in c.lower() and expected_section_display.lower() in c.lower()
        for c in citations
    )
    result.retrieval_hit = hit

    if not citations:
        result.retrieval_note = "no citations returned by the agent's tool call(s)"
    elif not hit:
        result.retrieval_note = "retrieved chunks did not match expected company/section"

    if run_groundedness:
        context = "\n\n".join(context_parts)
        try:
            verdict = judge_groundedness(item["question"], context, result.answer)
        except Exception as exc:  
            result.groundedness_run = True
            result.judge_error = f"judge call raised: {exc!r}"
            return result

        result.groundedness_run = True
        result.grounded = verdict["grounded"]
        result.unsupported_claims = verdict["unsupported_claims"]
        result.judge_reasoning = verdict["reasoning"]
        result.judge_error = verdict["error"]

    return result


# ------------------------------------------------------------------- groundedness

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking auditor for a financial research assistant.

You will be given:
- QUESTION: what the user asked
- CONTEXT: the retrieved 10-K excerpt(s) the assistant was allowed to use
- ANSWER: the assistant's final answer

Your ONLY job: determine whether every factual claim actually stated in ANSWER is supported by CONTEXT.
"Supported" means the claim, or something that reasonably implies it, appears in CONTEXT.
Do not use outside knowledge about the real company -- judge only against CONTEXT.

IMPORTANT -- faithfulness vs. completeness, do not conflate them:
- You are grading FAITHFULNESS ONLY: does ANSWER ever state something CONTEXT does not support?
- You are NOT grading completeness. If CONTEXT contains additional facts (e.g. a 4th business
  segment, an extra risk factor) that ANSWER simply does not mention, that is NOT a violation.
  An answer that omits available context but states nothing false is still grounded=true.
- Only set grounded=false if ANSWER contains a specific claim, figure, or detail that CONTEXT
  does not support -- never for something ANSWER left out.
- "unsupported_claims" must only list claims that ANSWER actually made and CONTEXT contradicts
  or is silent on -- never list something that ANSWER did not say.

If CONTEXT says the data is unavailable/not indexed, and ANSWER honestly says the same, that counts as grounded.

Respond with ONLY a JSON object, no markdown fences, no commentary, in exactly this shape:
{"grounded": true or false, "unsupported_claims": ["short phrase", ...], "reasoning": "one sentence"}
"""


def judge_groundedness(question: str, context: str, answer: str) -> dict:
    user_prompt = f"""QUESTION:
{question}

CONTEXT:
{context if context else "(no context was retrieved)"}

ANSWER:
{answer}

Return the JSON verdict now.
"""
    response = judge_llm.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    raw = response.content if hasattr(response, "content") else str(response)
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
        grounded = bool(parsed.get("grounded"))
        unsupported_claims = list(parsed.get("unsupported_claims", []))
        if grounded and unsupported_claims:
            # Self-contradictory verdict: the judge listed a real unsupported
            # claim but still marked the answer grounded. Treat as ungrounded
            # rather than silently trusting the boolean -- this exact failure
            # mode is what inflated a prior report to a false 1.0 score.
            grounded = False
        return {
            "grounded": grounded,
            "unsupported_claims": unsupported_claims,
            "reasoning": str(parsed.get("reasoning", "")),
            "error": None,
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "grounded": None,
            "unsupported_claims": [],
            "reasoning": "",
            "error": f"judge returned non-JSON output: {raw[:200]!r}",
        }


def evaluate_groundedness(item: dict, result: QuestionResult) -> None:
    start = time.time()
    try:
        answer, _ = ask_agent(item["question"])
    except Exception as exc:  
        result.answer = ""
        result.judge_error = f"ask_agent raised: {exc!r}"
        result.groundedness_run = True
        result.latency_sec = time.time() - start
        return

    result.answer = answer
    result.latency_sec = time.time() - start

    routed_company = result.routed_company
    routed_section = result.routed_section
    context = ""
    if routed_company:
        try:
            filing_result = query_financial_reports(
                query=item["question"],
                company=routed_company,
                fiscal_year=None,
                section=routed_section,
            )
            context = filing_result.get("content", "") or ""
        except Exception as exc:  
            result.groundedness_run = True
            result.judge_error = f"context retrieval for judge raised: {exc!r}"
            return

    try:
        verdict = judge_groundedness(item["question"], context, answer)
    except Exception as exc:  
        result.groundedness_run = True
        result.judge_error = f"judge call raised: {exc!r}"
        return

    result.groundedness_run = True
    result.grounded = verdict["grounded"]
    result.unsupported_claims = verdict["unsupported_claims"]
    result.judge_reasoning = verdict["reasoning"]
    result.judge_error = verdict["error"]


# --------------------------------------------------------------------- reporting


def summarize(results: list[QuestionResult]) -> dict:
    total = len(results)
    routed = [r for r in results if r.routed_company is not None]

    retrieval_hits = sum(1 for r in results if r.retrieval_hit)
    company_correct = sum(1 for r in results if r.routing_company_correct)
    section_correct = sum(1 for r in results if r.routing_section_correct)

    graded = [r for r in results if r.groundedness_run and r.grounded is not None]
    grounded_count = sum(1 for r in graded if r.grounded)
    judge_errors = sum(1 for r in results if r.judge_error)

    total_tool_calls = sum(r.tool_calls_count for r in results)
    zero_tool_call_questions = sum(1 for r in results if r.tool_calls_count == 0)

    by_category: dict[str, dict] = {}
    for r in results:
        cat = by_category.setdefault(
            r.category, {"total": 0, "retrieval_hits": 0, "grounded": 0, "graded": 0}
        )
        cat["total"] += 1
        if r.retrieval_hit:
            cat["retrieval_hits"] += 1
        if r.groundedness_run and r.grounded is not None:
            cat["graded"] += 1
            if r.grounded:
                cat["grounded"] += 1

    return {
        "total_questions": total,
        "routing_company_accuracy": round(company_correct / total, 3) if total else 0.0,
        "routing_section_accuracy": round(section_correct / total, 3) if total else 0.0,
        "retrieval_recall_at_k": round(retrieval_hits / total, 3) if total else 0.0,
        "retrieval_hits": retrieval_hits,
        "retrieval_misses": total - retrieval_hits,
        "unrouted_questions": total - len(routed),
        "groundedness_graded": len(graded),
        "groundedness_score": round(grounded_count / len(graded), 3) if graded else None,
        "judge_errors": judge_errors,
        "avg_tool_calls_per_question": round(total_tool_calls / total, 3) if total else 0.0,
        "questions_with_zero_tool_calls": zero_tool_call_questions,
        "by_category": {
            cat: {
                "total": v["total"],
                "retrieval_recall_at_k": round(v["retrieval_hits"] / v["total"], 3) if v["total"] else 0.0,
                "groundedness_score": round(v["grounded"] / v["graded"], 3) if v["graded"] else None,
            }
            for cat, v in by_category.items()
        },
    }


def write_markdown_report(path: Path, summary: dict, results: list[QuestionResult]) -> None:
    lines = []
    lines.append("# Sovereign Financial Analyst -- Phase 1 Eval Report\n")
    lines.append(f"Questions evaluated: **{summary['total_questions']}**\n")
    lines.append("## Headline numbers\n")
    lines.append(f"- **Retrieval recall@k:** {summary['retrieval_recall_at_k'] * 100:.1f}% "
                 f"({summary['retrieval_hits']}/{summary['total_questions']})")
    lines.append(f"- **Company routing accuracy:** {summary['routing_company_accuracy'] * 100:.1f}%")
    lines.append(f"- **Section routing accuracy:** {summary['routing_section_accuracy'] * 100:.1f}%")
    if summary["groundedness_score"] is not None:
        lines.append(f"- **Groundedness (faithfulness):** {summary['groundedness_score'] * 100:.1f}% "
                     f"of graded answers ({summary['groundedness_graded']} graded)")
    else:
        lines.append("- **Groundedness:** not run (use without `--skip-groundedness`)")
    if summary["judge_errors"]:
        lines.append(f"- Judge parse errors: {summary['judge_errors']} (see JSON report for detail)")
    lines.append("")

    lines.append("## By section\n")
    lines.append("| Section | N | Retrieval recall@k | Groundedness |")
    lines.append("|---|---|---|---|")
    for cat, v in summary["by_category"].items():
        g = f"{v['groundedness_score'] * 100:.1f}%" if v["groundedness_score"] is not None else "n/a"
        lines.append(f"| {cat} | {v['total']} | {v['retrieval_recall_at_k'] * 100:.1f}% | {g} |")
    lines.append("")

    lines.append("## Misses\n")
    misses = [r for r in results if not r.retrieval_hit or r.grounded is False]
    if not misses:
        lines.append("None. ")
    for r in misses:
        lines.append(f"- **{r.id}** ({r.category}): \"{r.question}\"")
        if not r.retrieval_hit:
            lines.append(f"  - retrieval miss: routed to (company={r.routed_company}, section={r.routed_section}), "
                         f"expected (company={r.expected_company}, section={r.expected_section}). {r.retrieval_note}")
        if r.grounded is False:
            claims = "; ".join(r.unsupported_claims) if r.unsupported_claims else "(unspecified)"
            lines.append(f"  - groundedness fail: {r.judge_reasoning or ''} unsupported claims: {claims}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_comparison(summary_rb: dict, summary_ag: dict) -> None:
    print("\n" + "=" * 72)
    print("RULE_BASED vs AGENTIC")
    print("=" * 72)
    rows = [
        ("Retrieval recall@k", summary_rb["retrieval_recall_at_k"], summary_ag["retrieval_recall_at_k"]),
        ("Company routing accuracy", summary_rb["routing_company_accuracy"], summary_ag["routing_company_accuracy"]),
        ("Groundedness", summary_rb["groundedness_score"], summary_ag["groundedness_score"]),
    ]
    for label, rb, ag in rows:
        rb_s = f"{rb * 100:.1f}%" if rb is not None else "n/a"
        ag_s = f"{ag * 100:.1f}%" if ag is not None else "n/a"
        print(f"{label:<28} rule_based: {rb_s:<8} agentic: {ag_s}")
    print(f"{'Avg tool calls/question':<28} rule_based: n/a       agentic: {summary_ag['avg_tool_calls_per_question']}")
    print("=" * 72)


def write_comparison_report(path: Path, summary_rb: dict, summary_ag: dict) -> None:
    lines = ["# Sovereign Financial Analyst -- rule_based vs agentic\n"]
    lines.append("| Metric | rule_based | agentic |")
    lines.append("|---|---|---|")

    def fmt(v: Optional[float]) -> str:
        return f"{v * 100:.1f}%" if v is not None else "n/a"

    lines.append(f"| Retrieval recall@k | {fmt(summary_rb['retrieval_recall_at_k'])} | {fmt(summary_ag['retrieval_recall_at_k'])} |")
    lines.append(f"| Company routing accuracy | {fmt(summary_rb['routing_company_accuracy'])} | {fmt(summary_ag['routing_company_accuracy'])} |")
    lines.append(f"| Section routing accuracy | {fmt(summary_rb['routing_section_accuracy'])} | {fmt(summary_ag['routing_section_accuracy'])} |")
    lines.append(f"| Groundedness | {fmt(summary_rb['groundedness_score'])} | {fmt(summary_ag['groundedness_score'])} |")
    lines.append(f"| Avg tool calls / question | n/a | {summary_ag['avg_tool_calls_per_question']} |")
    lines.append(f"| Questions with zero tool calls | n/a | {summary_ag['questions_with_zero_tool_calls']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------------- main


def run_eval(questions: list[dict], mode: str, args: argparse.Namespace, out_path: Path, md_path: Path) -> dict:
    """Runs one full pass (either mode) over `questions`, checkpointing and
    reporting to the given paths. Returns the summary dict."""
    label = "Phase 1 eval" if mode == "rule_based" else "Phase 2 agentic eval"
    print(f"\nRunning {label} ({mode}) on {len(questions)} question(s)"
          f"{' (retrieval only)' if args.skip_groundedness else ''} ...\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[QuestionResult] = []
    for i, item in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {item['id']}: {item['question']}")

        if mode == "rule_based":
            result = evaluate_retrieval(item, k=args.k)
            status = "HIT " if result.retrieval_hit else "MISS"
            print(f"    routing -> company={result.routed_company} section={result.routed_section} "
                  f"| retrieval {status}")
            if not args.skip_groundedness:
                evaluate_groundedness(item, result)
                if result.grounded is not None:
                    print(f"    groundedness -> {'GROUNDED' if result.grounded else 'NOT GROUNDED'}")
                elif result.judge_error:
                    print(f"    groundedness -> judge error: {result.judge_error}")
        else:  # agentic
            result = evaluate_agentic(item, k=args.k, run_groundedness=not args.skip_groundedness)
            status = "HIT " if result.retrieval_hit else "MISS"
            print(f"    agent routed -> company={result.routed_company} section={result.routed_section} "
                  f"| tool_calls={result.tool_calls_count} | retrieval {status}")
            if result.grounded is not None:
                print(f"    groundedness -> {'GROUNDED' if result.grounded else 'NOT GROUNDED'}")
            elif result.judge_error:
                print(f"    groundedness -> judge error: {result.judge_error}")

        results.append(result)

        # Checkpoint after every question -- if Ollama/the process hard-crashes
        # partway through (e.g. a Metal OOM), you still have every result up
        # to that point on disk instead of losing the whole run.
        partial_summary = summarize(results)
        out_path.write_text(
            json.dumps({"mode": mode, "summary": partial_summary, "results": [asdict(r) for r in results],
                        "status": "in_progress" if i < len(questions) else "complete"}, indent=2),
            encoding="utf-8",
        )

        if args.sleep and i < len(questions):
            time.sleep(args.sleep)

    summary = summarize(results)

    print("\n" + "=" * 72)
    print(f"SUMMARY ({mode})")
    print("=" * 72)
    print(f"Retrieval recall@k:        {summary['retrieval_recall_at_k'] * 100:.1f}% "
          f"({summary['retrieval_hits']}/{summary['total_questions']})")
    print(f"Company routing accuracy:  {summary['routing_company_accuracy'] * 100:.1f}%")
    print(f"Section routing accuracy:  {summary['routing_section_accuracy'] * 100:.1f}%")
    if summary["groundedness_score"] is not None:
        print(f"Groundedness score:        {summary['groundedness_score'] * 100:.1f}% "
              f"({summary['groundedness_graded']} graded, {summary['judge_errors']} judge errors)")
    else:
        print("Groundedness score:        not run")
    if mode == "agentic":
        print(f"Avg tool calls/question:   {summary['avg_tool_calls_per_question']}")
    print("=" * 72)
    print(f"\nJSON report written to {out_path}")

    write_markdown_report(md_path, summary, results)
    print(f"Markdown report written to {md_path}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval harness for Sovereign Financial Analyst")
    parser.add_argument("--mode", choices=["rule_based", "agentic", "both"], default="rule_based",
                         help="which routing strategy to evaluate. 'rule_based' (default) preserves the "
                              "original Phase 1 behavior and default file paths. 'agentic' runs the LLM "
                              "tool-calling router instead. 'both' runs each in turn and prints/writes a "
                              "head-to-head comparison.")
    parser.add_argument("--category", choices=["business", "risk_factors", "mdna", "financial_statements"],
                         help="only run questions from this section")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (after filtering)")
    parser.add_argument("--k", type=int, default=4, help="retrieval k (default matches app.tools default of 4)")
    parser.add_argument("--skip-groundedness", action="store_true",
                         help="skip the LLM-as-judge pass; retrieval metrics only (much faster)")
    parser.add_argument("--sleep", type=float, default=0.0,
                         help="seconds to pause between questions (helps if Ollama/GPU backend is unstable "
                              "under back-to-back calls, e.g. macOS Metal OOM crashes)")
    parser.add_argument("--out", type=str, default=None,
                         help="path to write the JSON report (single-mode runs only; ignored for --mode both, "
                              "defaults to eval/report.json for rule_based, eval/report_agentic.json for agentic)")
    parser.add_argument("--md-out", type=str, default=None,
                         help="path to write the Markdown report (single-mode runs only; same default rule as --out)")
    args = parser.parse_args()

    questions = EVAL_QUESTIONS
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        print("No questions match the given filters.")
        return 1

    default_paths = {
        "rule_based": (REPO_ROOT / "eval" / "report.json", REPO_ROOT / "eval" / "report.md"),
        "agentic": (REPO_ROOT / "eval" / "report_agentic.json", REPO_ROOT / "eval" / "report_agentic.md"),
    }

    if args.mode == "both":
        summary_rb = run_eval(questions, "rule_based", args, *default_paths["rule_based"])
        summary_ag = run_eval(questions, "agentic", args, *default_paths["agentic"])
        print_comparison(summary_rb, summary_ag)
        comparison_path = REPO_ROOT / "eval" / "comparison.md"
        write_comparison_report(comparison_path, summary_rb, summary_ag)
        print(f"\nComparison report written to {comparison_path}")
        return 0

    default_json, default_md = default_paths[args.mode]
    out_path = Path(args.out) if args.out else default_json
    md_path = Path(args.md_out) if args.md_out else default_md
    run_eval(questions, args.mode, args, out_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())