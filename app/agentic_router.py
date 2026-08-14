from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.config import SYSTEM_PROMPT
from app.companies import SLUG_TO_DISPLAY
from app.exceptions import OllamaUnavailableError
from app.financial_answer import answer_financial_point_lookup
from app.prose_evidence import append_evidence_if_useful, build_evidence_brief
from app.llm import get_llm
from app.tool_args import sanitize_tool_args
from app.tools import get_stock_performance, normalize_company, query_financial_reports

logger = logging.getLogger("sovereign_fa.agentic")

MAX_TOOL_ROUNDS = 2


@tool
def search_filing(
    company: str,
    question: str,
    section: Optional[str] = None,
    fiscal_year: Optional[int] = None,
) -> str:
    """Search a company's indexed SEC 10-K filing(s) for information relevant
    to the question. Use this for anything about business overview, risk
    factors, MD&A / results of operations, or financial statements.

    Args:
        company: Company name or stock ticker, e.g. "Nvidia" or "NVDA".
        question: The user's question, passed through as the search query.
        section: Optional filter -- one of "business", "risk_factors",
            "mdna", "financial_statements". Omit to search all sections.
        fiscal_year: Optional filter, e.g. 2024. Omit to search all
            indexed fiscal years.
    """
    result = query_financial_reports(
        query=question,
        company=company,
        fiscal_year=fiscal_year,
        section=section,
    )
    return json.dumps(result)


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest 5-day stock price, high/low, and volume for a ticker."""
    result = get_stock_performance(ticker)
    return json.dumps(result)


TOOLS = [search_filing, get_stock_price]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

_llm = get_llm()
_llm_with_tools = _llm.bind_tools(TOOLS)


AGENTIC_ADDENDUM = """

Tool-Calling Instructions
- Decide which tool(s) to call from the user's question.
- Call search_filing for business, risk, MD&A, or financial-statement questions.
- Call get_stock_price only for live/current stock-price or volume questions.
- You may issue multiple tool calls in the same turn when the question requires them.
- After successful tool results, answer the user's question from those results.
"""


SYNTHESIS_RULES = """

SYNTHESIS MODE — tools are no longer available.
Use only the retrieved tool results supplied below.
- Answer the original user question directly in normal prose. Never output a tool call, function name, JSON request, or tool syntax.
- Do not invent facts or values that are absent from the retrieved result.
- For financial-statement questions, prioritize explicitly labeled accounting rows and the latest period shown in the retrieved filing.
- Preserve the units shown in the filing. If values are in millions, either state the value in millions or convert accurately to billions.
- If the user asks for an aggregate that is not explicitly reported, do not guess or silently add unrelated numbers. State that the aggregate is not explicitly shown and report the relevant explicitly listed components instead.
- Do not add comparisons, sections, forecasts, or stock-price discussion unless the user asked for them.
- When a HIGH-PRIORITY EVIDENCE block is supplied, cover its material named entities, numeric values, directions of change, and stated drivers before adding broader context.
- Prefer realized period-over-period results over generic forward-looking trend language when the question asks what happened or what drove a change.
"""


def _extract_company_from_calls(tool_calls: list[dict]) -> Optional[str]:
    for call in tool_calls:
        if call.get("name") == "search_filing":
            company = call.get("args", {}).get("company")
            if company:
                return normalize_company(company)
    return None


def _tool_result_is_error(raw_result: str) -> bool:
    try:
        parsed = json.loads(raw_result)
    except Exception:
        return False
    return isinstance(parsed, dict) and bool(parsed.get("error"))


def _looks_like_tool_request(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False

    lowered = stripped.lower()
    if any(marker in lowered for marker in (
        '"name": "query_financial_reports"',
        '"name":"query_financial_reports"',
        '"name": "search_filing"',
        '"name":"search_filing"',
        '"name": "get_stock_performance"',
        '"name":"get_stock_performance"',
        '"name": "get_stock_price"',
        '"name":"get_stock_price"',
    )):
        return True

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
        except Exception:
            return False
        if isinstance(obj, dict) and "name" in obj and (
            "parameters" in obj or "arguments" in obj or "args" in obj
        ):
            return True

    return False


def _synthesize_from_tool_results(
    user_input: str,
    tools_invoked: list[dict],
    round_num: int,
) -> AIMessage:
    # Simple financial-statement point lookups should not depend on a small
    # generative model choosing between adjacent accounting rows. If retrieval
    # already found an explicitly labeled metric row, answer deterministically
    # from that evidence. Broad financial questions still fall through to LLM
    # synthesis below.
    for record in tools_invoked:
        if record.get("tool") != "search_filing":
            continue
        args = record.get("args") or {}
        if args.get("section") != "financial_statements":
            continue
        try:
            parsed = json.loads(record.get("result") or "{}")
        except Exception:
            continue
        if not isinstance(parsed, dict) or parsed.get("error"):
            continue
        content = parsed.get("content") or ""
        company = args.get("company") or "the company"
        deterministic = answer_financial_point_lookup(
            user_input,
            company,
            content,
        )
        if deterministic:
            return AIMessage(content=deterministic)

    evidence_records: list[dict] = []
    for record in tools_invoked:
        if record.get("tool") != "search_filing":
            continue
        args = record.get("args") or {}
        section = args.get("section")
        if section == "financial_statements":
            continue
        try:
            parsed = json.loads(record.get("result") or "{}")
        except Exception:
            continue
        if not isinstance(parsed, dict) or parsed.get("error"):
            continue
        content = parsed.get("content") or ""
        brief = build_evidence_brief(user_input, content)
        if brief:
            evidence_records.append({
                "section": section,
                "brief": brief,
            })

    rendered_results = []
    for index, record in enumerate(tools_invoked, start=1):
        rendered_results.append(
            f"Tool result {index} ({record['tool']}):\n{record['result']}"
        )

    context = "\n\n".join(rendered_results)
    evidence_context = "\n\n".join(
        f"HIGH-PRIORITY EVIDENCE {index}:\n{item['brief']}"
        for index, item in enumerate(evidence_records, start=1)
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + SYNTHESIS_RULES),
        HumanMessage(
            content=(
                f"Original question:\n{user_input}\n\n"
                + (f"{evidence_context}\n\n" if evidence_context else "")
                + f"Retrieved results:\n{context}\n\n"
                + "Answer the original question now. Cover the high-priority evidence first."
            )
        ),
    ]

    final = _invoke(messages, round_num, force_no_tools=True)

    # Small local models occasionally imitate a tool call even when tools are
    # unbound. One no-tools repair attempt prevents that malformed content from
    # leaking to the user/evaluator.
    if _looks_like_tool_request(final.content):
        messages.extend([
            final,
            HumanMessage(
                content=(
                    "Your previous response attempted to emit tool/function syntax. "
                    "Do not call or name any tool. Return only the direct plain-language "
                    "answer to the original question using the retrieved results above."
                )
            ),
        ])
        final = _invoke(messages, round_num + 1, force_no_tools=True)

    final_content = final.content
    for item in evidence_records:
        final_content = append_evidence_if_useful(
            final_content,
            item["brief"],
            question=user_input,
            section=item.get("section"),
        )

    return AIMessage(content=final_content)


def run_agentic_query(
    user_input: str,
    conversation_company: Optional[str] = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> dict:
    company_hint = ""
    if conversation_company:
        display = SLUG_TO_DISPLAY.get(conversation_company, conversation_company)
        company_hint = (
            f"\n\nConversation context: the user was previously discussing {display}. "
            f"If this question doesn't name a different company, assume they still mean {display}."
        )

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT + AGENTIC_ADDENDUM + company_hint),
        HumanMessage(content=user_input),
    ]

    tools_invoked: list[dict] = []
    resolved_company: Optional[str] = conversation_company

    for round_num in range(max_tool_rounds):
        ai_message = _invoke(messages, round_num)
        messages.append(ai_message)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            # If no tool was ever needed, the model's direct answer is valid.
            if not tools_invoked:
                return {
                    "answer": ai_message.content,
                    "active_company": resolved_company,
                    "tools_invoked": tools_invoked,
                    "rounds": round_num + 1,
                }

            # Tool results exist: synthesize through the unbound model rather
            # than trusting tool-shaped text from a tool-bound generation.
            final = _synthesize_from_tool_results(user_input, tools_invoked, round_num + 1)
            return {
                "answer": final.content,
                "active_company": resolved_company,
                "tools_invoked": tools_invoked,
                "rounds": round_num + 1,
            }

        company_from_calls = _extract_company_from_calls(tool_calls)
        if company_from_calls:
            resolved_company = company_from_calls

        round_had_error = False

        for call in tool_calls:
            name = call["name"]
            raw_args = dict(call.get("args", {}) or {})
            args = sanitize_tool_args(name, raw_args, user_input=user_input)
            tool_fn = TOOLS_BY_NAME.get(name)

            if tool_fn is None:
                tool_result = json.dumps({"error": f"unknown tool '{name}'"})
            else:
                try:
                    tool_result = tool_fn.invoke(args)
                except Exception as exc:
                    logger.warning(
                        "tool_call_failed",
                        extra={"tool": name, "tool_args": args, "error": str(exc)},
                    )
                    tool_result = json.dumps({"error": str(exc)})

            if _tool_result_is_error(tool_result):
                round_had_error = True

            tools_invoked.append({
                "tool": name,
                "raw_args": raw_args,
                "args": args,
                "result": tool_result,
            })
            messages.append(ToolMessage(content=tool_result, tool_call_id=call["id"]))

        # Once the requested tools succeeded, switch to a dedicated unbound
        # synthesis pass. This prevents llama3.2:3b from inventing a second tool
        # request instead of answering from already-good filing context.
        if not round_had_error:
            final = _synthesize_from_tool_results(user_input, tools_invoked, round_num + 1)
            return {
                "answer": final.content,
                "active_company": resolved_company,
                "tools_invoked": tools_invoked,
                "rounds": round_num + 1,
            }

        # Otherwise let the agent see the error and try one corrective round.

    final = _synthesize_from_tool_results(user_input, tools_invoked, max_tool_rounds)
    return {
        "answer": final.content,
        "active_company": resolved_company,
        "tools_invoked": tools_invoked,
        "rounds": max_tool_rounds,
    }


def _invoke(messages: list, round_num: int, force_no_tools: bool = False) -> AIMessage:
    target = _llm if force_no_tools else _llm_with_tools
    try:
        return target.invoke(messages)
    except Exception as exc:
        logger.error(
            "ollama_unreachable",
            extra={"round": round_num, "error": str(exc)},
        )
        raise OllamaUnavailableError(str(exc)) from exc
