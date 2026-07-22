from __future__ import annotations
import json
import logging
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from app.config import OLLAMA_MODEL, SYSTEM_PROMPT
from app.companies import SLUG_TO_DISPLAY
from app.exceptions import OllamaUnavailableError
from app.tools import get_stock_performance, normalize_company, query_financial_reports

logger = logging.getLogger("sovereign_fa.agentic")

MAX_TOOL_ROUNDS = 2  


# --------------------------------------------------------------------------- tools


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
            indexed fiscal years (2023-2025).
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
    """Get the latest 5-day stock price, high/low, and volume for a ticker
    symbol. Use this for anything about live/current stock performance.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA", "AAPL".
    """
    result = get_stock_performance(ticker)
    return json.dumps(result)


TOOLS = [search_filing, get_stock_price]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

_llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
_llm_with_tools = _llm.bind_tools(TOOLS)


AGENTIC_ADDENDUM = """

Tool-Calling Instructions
- You decide which tool(s) to call, if any, based on the user's question.
  Nothing has pre-selected a company or section for you -- read the
  question yourself and choose the arguments.
- Call search_filing for anything about business, risk, MD&A, or financials.
- Call get_stock_price for anything about live stock price/volume.
- You may call both tools in the same turn if the question needs both,
  and you may call search_filing more than once for a comparison across
  two companies.
- After you receive tool results, write ONE final natural-language answer
  synthesizing them, following the Response Format and Grounding Rules
  above. Do not call a tool again unless a previous call errored.
"""


def _extract_company_from_calls(tool_calls: list[dict]) -> Optional[str]:
    for call in tool_calls:
        if call.get("name") == "search_filing":
            company = call.get("args", {}).get("company")
            if company:
                return normalize_company(company)
    return None


def run_agentic_query(
    user_input: str,
    conversation_company: Optional[str] = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> dict:
    """
    Runs the LLM tool-calling loop and returns:
      {
        "answer": str,
        "active_company": Optional[str]  (slug),
        "tools_invoked": [{"tool": str, "args": dict, "result": str (raw JSON)}, ...],
        "rounds": int,
      }
    Raises OllamaUnavailableError if the model can't be reached.
    """
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
            return {
                "answer": ai_message.content,
                "active_company": resolved_company,
                "tools_invoked": tools_invoked,
                "rounds": round_num + 1,
            }

        company_from_calls = _extract_company_from_calls(tool_calls)
        if company_from_calls:
            resolved_company = company_from_calls

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {})
            tool_fn = TOOLS_BY_NAME.get(name)

            if tool_fn is None:
                tool_result = json.dumps({"error": f"unknown tool '{name}'"})
            else:
                try:
                    tool_result = tool_fn.invoke(args)
                except Exception as exc:
                    logger.warning(
                        "tool_call_failed",
                        extra={"tool": name, "args": args, "error": str(exc)},
                    )
                    tool_result = json.dumps({"error": str(exc)})

            tools_invoked.append({"tool": name, "args": args, "result": tool_result})
            messages.append(ToolMessage(content=tool_result, tool_call_id=call["id"]))

    # Ran out of tool-call rounds -- force a final answer using whatever
    # tool results were gathered, rather than looping indefinitely.
    messages.append(
        HumanMessage(
            content="Write your final answer now, using only the tool results above. "
            "Do not call any more tools."
        )
    )
    final = _invoke(messages, max_tool_rounds, force_no_tools=True)

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
