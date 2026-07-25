"""
Run locally:
    uvicorn api.main:app --reload --port 8000

Then:
    curl http://localhost:8000/health

    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"message": "What are the 3 main risks in Nvidia'"'"'s 10-K?"}'

    curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"message": "How is NVDA stock performing?", "mode": "agentic"}'

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.companies import COMPANIES, SLUG_TO_DISPLAY, TARGET_FISCAL_YEARS
from app.config import OLLAMA_MODEL
from app.exceptions import (
    OllamaUnavailableError,
    SovereignError,
    StockDataUnavailableError,
    VectorStoreUnavailableError,
)
from app.logging_config import configure_logging
from app.schemas import (
    CompaniesResponse,
    CompanyInfo,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ToolInvocation,
)

configure_logging()
logger = logging.getLogger("sovereign_fa.api")


def _warmup() -> None:
    """Pre-load the embedding model, vectorstore, and LLM client so the
    first real user query doesn't pay this cost. Runs in a thread so it
    never blocks /health from responding.
    """
    start = time.perf_counter()
    try:
        from app.agent import invoke_llm_with_retry
        from app.tools import query_financial_reports

        for company in ("apple", "nvidia", "microsoft"):
            try:
                query_financial_reports(query="warmup", company=company, k=1)
            except Exception as exc:  # noqa: BLE001 -- one bad company shouldn't abort warmup
                logger.warning("warmup_retrieval_failed", extra={"company_slug": company, "error": str(exc)})

        invoke_llm_with_retry(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            retries=0,
        )
        logger.info(
            "warmup_complete",
            extra={"latency_ms": round((time.perf_counter() - start) * 1000, 1)},
        )
    except Exception as exc:  # noqa: BLE001 -- best-effort warmup, never fatal
        logger.warning(
            "warmup_failed",
            extra={"error": str(exc), "latency_ms": round((time.perf_counter() - start) * 1000, 1)},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", extra={})
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _warmup)
    yield
    logger.info("shutdown", extra={})


app = FastAPI(
    title="Sovereign Financial Analyst API",
    description="RAG service over indexed SEC 10-K filings plus live stock data. "
    "Two routing modes available per request: rule_based (keyword/regex, app.agent) "
    "and agentic (LLM tool-calling, app.agentic_router).",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "latency_ms": latency_ms,
                "status_code": getattr(response, "status_code", 500),
            },
        )


def _error_response(status_code: int, error: str, detail: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, detail=detail).model_dump(),
    )


# --------------------------------------------------------------------- errors


@app.exception_handler(OllamaUnavailableError)
async def ollama_unavailable_handler(request: Request, exc: OllamaUnavailableError) -> JSONResponse:
    logger.error("ollama_unavailable", extra={"error": str(exc)})
    return _error_response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Local LLM (Ollama) is unavailable.",
        "Make sure `ollama serve` is running and the configured model is pulled.",
    )


@app.exception_handler(VectorStoreUnavailableError)
async def vector_store_unavailable_handler(request: Request, exc: VectorStoreUnavailableError) -> JSONResponse:
    logger.error("vector_store_unavailable", extra={"error": str(exc)})
    return _error_response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Vector store is unavailable.",
        str(exc),
    )


@app.exception_handler(StockDataUnavailableError)
async def stock_unavailable_handler(request: Request, exc: StockDataUnavailableError) -> JSONResponse:
    logger.warning("stock_unavailable", extra={"error": str(exc)})
    return _error_response(
        status.HTTP_502_BAD_GATEWAY,
        "Stock data provider is unavailable.",
        str(exc),
    )


@app.exception_handler(SovereignError)
async def generic_domain_error_handler(request: Request, exc: SovereignError) -> JSONResponse:
    logger.error("domain_error", extra={"error": str(exc)})
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error.", str(exc))


# -------------------------------------------------------------------- routes


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness check. Does NOT verify Ollama/Chroma are reachable -- those
    are checked lazily on /query so an empty local dev environment can
    still start the API and inspect /docs."""
    return HealthResponse(status="ok", ollama_model=OLLAMA_MODEL, vector_store="chroma (local)")


@app.get("/companies", response_model=CompaniesResponse, tags=["meta"])
def companies() -> CompaniesResponse:
    """List the 20 indexed companies and fiscal years, for building a
    picker in a client UI instead of hardcoding it there."""
    return CompaniesResponse(
        companies=[
            CompanyInfo(slug=slug, display=meta["display"], ticker=meta["ticker"])
            for slug, meta in COMPANIES.items()
        ],
        fiscal_years=sorted(TARGET_FISCAL_YEARS),
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        502: {"model": ErrorResponse, "description": "Stock data provider unavailable"},
        503: {"model": ErrorResponse, "description": "Ollama or the vector store is unavailable"},
    },
    tags=["query"],
)
def query(payload: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    if payload.mode == "agentic":
        from app.agentic_router import run_agentic_query

        result = run_agentic_query(payload.message, payload.conversation_company)
        active_company = result["active_company"]
        tools_invoked = [ToolInvocation(tool=t["tool"], args=t["args"]) for t in result["tools_invoked"]]
        answer = result["answer"]
    else:
        from app.agent import ask_agent

        answer, active_company = ask_agent(payload.message, payload.conversation_company)
        tools_invoked = []

    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    logger.info(
        "query_answered",
        extra={
            "query": payload.message,
            "company": active_company,
            "mode": payload.mode,
            "tools_invoked": [t.model_dump() for t in tools_invoked],
            "latency_ms": latency_ms,
        },
    )

    return QueryResponse(
        answer=answer,
        active_company=active_company,
        active_company_display=SLUG_TO_DISPLAY.get(active_company, active_company) if active_company else None,
        mode=payload.mode,
        tools_invoked=tools_invoked,
        latency_ms=latency_ms,
    )