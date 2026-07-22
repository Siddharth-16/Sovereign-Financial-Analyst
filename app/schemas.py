from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ToolInvocation(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, max_length=2000, description="User's natural language question"
    )
    conversation_company: Optional[str] = Field(
        None,
        description="Company slug, ticker, or display name carried over from a prior turn "
        "(e.g. 'nvidia', 'NVDA', 'Nvidia'). Omit on the first turn of a conversation.",
    )
    mode: Literal["rule_based", "agentic"] = Field(
        "rule_based",
        description="'rule_based' uses the keyword/regex router in app.agent (fast, deterministic). "
        "'agentic' lets the LLM itself choose which tools to call via app.agentic_router "
        "(slower, but a true tool-calling agent).",
    )

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "What are the 3 main risks in Nvidia's 10-K?", "mode": "rule_based"},
                {"message": "How is NVDA stock performing?", "conversation_company": "nvidia", "mode": "agentic"},
            ]
        }
    }


class QueryResponse(BaseModel):
    answer: str
    active_company: Optional[str] = Field(None, description="Company slug the answer is scoped to, if any")
    active_company_display: Optional[str] = Field(None, description="Human-readable company name")
    mode: Literal["rule_based", "agentic"]
    tools_invoked: list[ToolInvocation] = Field(
        default_factory=list,
        description="Tools the LLM chose to call, in order (always empty for mode='rule_based', "
        "since that path calls tools deterministically rather than via LLM tool-calling)",
    )
    latency_ms: float


class CompanyInfo(BaseModel):
    slug: str
    display: str
    ticker: str


class CompaniesResponse(BaseModel):
    companies: list[CompanyInfo]
    fiscal_years: list[int]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ollama_model: str
    vector_store: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
