# Sovereign Financial Analyst

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED)
![Tests](https://img.shields.io/badge/tests-68%20passing-brightgreen)

**Sovereign Financial Analyst** is a Retrieval-Augmented Generation system for analyzing SEC 10-K filings and live stock data. It answers natural-language questions about corporate filings — risk factors, revenue trends, business segments, financial statements, cross-company comparisons — grounded in retrieved source text, with citations back to the exact filing and section every answer draws from.

The system indexes **20 public companies** across **fiscal years 2023-2025**, section-aware (Business, Risk Factors, MD&A, and Financial Statements retrieved independently rather than as one undifferentiated blob), and supports two interchangeable query-routing strategies and two interchangeable LLM backends, so the same codebase runs entirely offline on a laptop or behind a public API with zero code changes -- only configuration.

```text
What are Nvidia's supply chain risks?
Compare Nvidia and AMD's risk factors.
What is Nvidia's revenue trend from the filing?
How is NVDA stock performing?
```

---

## Contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Query routing: rule-based vs. agentic](#query-routing-rule-based-vs-agentic)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Tech stack](#tech-stack)
- [Local setup](#local-setup)
- [Deployment](#deployment)
- [Environment variables](#environment-variables)
- [CI/CD](#cicd)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)
- [Possible extensions](#possible-extensions)
- [License](#license)

---

## Why this exists

10-K filings are long, dense, and hard to navigate even for people who read them for a living. This project retrieves only the relevant section of the relevant filing for a given question, then has an LLM synthesize a grounded answer from that retrieved text -- not from the model's own training data. Every claim in an answer traces back to a cited filing section, and the system is built to say a fact isn't available in the indexed filings rather than guess when it isn't actually in the retrieved context.

It's also a deliberately end-to-end build: ingestion, retrieval, two routing strategies, an evaluation harness that measures whether the grounding claim actually holds, a REST API decoupled from the UI, a test suite, containerization, and a CI/CD pipeline -- the full path from a RAG prototype to something another engineer could actually run.

---

## Architecture

```
                        +------------------+
                        |   Streamlit UI    |
                        +---------+--------+
                                  |
                        +---------v--------+
                        |  FastAPI service   |   (api/main.py)
                        |  /query /health     |
                        |  /companies          |
                        +---------+--------+
                                  |
                 +----------------+----------------+
                 |                                   |
        +--------v---------+              +---------v----------+
        |  rule_based mode    |              |   agentic mode       |
        |  (app/agent.py)     |              | (app/agentic_router)  |
        |  keyword/regex        |              |  LLM tool-calling      |
        |  routing                |              |  decides tool calls     |
        +--------+---------+              +---------+----------+
                 |                                   |
                 +----------------+----------------+
                                  |
                 +----------------+----------------+
                 |                                   |
        +--------v---------+              +---------v----------+
        | query_financial_    |              | get_stock_performance |
        | reports (Chroma)    |              |      (yfinance)         |
        +--------+---------+              +----------------------+
                 |
        +--------v---------+
        |  LLM synthesis      |   (Ollama, local -- or Groq, hosted)
        |  with citations       |
        +----------------------+
```

- **Ingestion** (`scripts/ingest.py`) pulls 10-K filings via the SEC API, splits them into sections (Business, Risk Factors, MD&A, Financial Statements) rather than treating a filing as one document, chunks each section, embeds the chunks, and persists them to a local ChromaDB store tagged with company, fiscal year, and section metadata.
- **Retrieval** filters by company/section/fiscal-year metadata before running similarity search, so a question about Apple's risk factors only searches Apple's risk factors -- not all 20 companies' filings.
- **Synthesis** takes the retrieved chunks plus the user's question and asks the LLM to answer using only that context, with a citation for every section it drew from.
- **The API and UI are decoupled**: the FastAPI service can be called, tested, and deployed independently of the Streamlit UI. Streamlit is a thin client on top of the same `app.agent` / `app.agentic_router` logic the API exposes.

---

## Query routing: rule-based vs. agentic

The system supports two ways of deciding which tools to call for a given question, selectable per-request via the API's `mode` field:

**`rule_based`** (default) -- keyword and regex matching in `app/agent.py` identifies the company, section, and required tools deterministically. Fast, predictable, and -- per the evaluation numbers below -- currently the more accurate of the two.

**`agentic`** -- the LLM itself decides which tools to call and with what arguments, via real LangChain tool-calling (`app/agentic_router.py`), not a scripted call sequence. Slower and, on the current small local model, somewhat less reliable at both company-name extraction and staying strictly within retrieved context -- but it's a genuine tool-calling agent, not a rule-based system dressed up as one.

Both modes call the same underlying tools (`query_financial_reports`, `get_stock_performance`) and go through the same LLM-synthesis step; the only difference is how the decision to call a tool gets made. This distinction matters because a keyword router that never lets the model choose its own actions isn't accurately described as agentic -- a claim that doesn't survive a technical follow-up question is worse than no claim at all, so both modes exist, are separately evaluated, and are described here for what they actually are.

---

## Evaluation

Retrieval quality and answer groundedness are both measured, not assumed. `eval/eval.py` runs a hand-labeled set of 28 questions spanning all 20 companies and all four sections (Business, Risk Factors, MD&A, Financial Statements) end-to-end through the live system, checks whether retrieval pulled the expected company/section, and grades every answer for groundedness using an LLM-as-judge pass -- with the judge deliberately run on a separate, stronger model from whichever model answers questions, since a small model grading its own answers produces unreliable, sometimes self-contradictory verdicts.

| Metric                     | rule_based | agentic |
| -------------------------- | ---------- | ------- |
| Retrieval recall@k         | **100%**   | 92.9%   |
| Company routing accuracy   | **100%**   | 92.9%   |
| Groundedness (LLM-judged)  | **92.9%**  | 82.1%   |
| Avg. tool calls / question | n/a        | 1.0     |

```bash
python eval/eval.py --mode both
```

Produces `eval/report.json` and `eval/report_agentic.json` with full per-question results (routed company/section, retrieval hit/miss, groundedness verdict, cited unsupported claims where applicable, latency) alongside the summary above.

**Reading the gap honestly:** agentic mode's two retrieval misses trace to a specific, real bug -- the LLM's tool-calling arguments sometimes include the full legal suffix (e.g. "tesla, inc." instead of "tesla"), which isn't normalized before the company lookup, so retrieval correctly returns nothing rather than silently guessing. That's a fixable normalization gap in the agentic path specifically, not a retrieval-index problem -- rule_based's keyword extraction doesn't hit this because it never passes free-form LLM-generated text through as a lookup key.

---

## Testing

68 tests across routing, tools, ingestion's section-splitting logic, and the API layer, all runnable without any external services -- Ollama, ChromaDB, and yfinance are stubbed in `tests/conftest.py` so the suite exercises actual logic rather than mocked-out no-ops, without requiring a running LLM or a built vector store.

```bash
pytest -v
```

---

## Tech stack

**Backend / API** -- FastAPI, Pydantic, Uvicorn, structured JSON logging

**LLM** -- pluggable via `LLM_PROVIDER`: local Ollama (privacy-first, no external calls) or hosted Groq (used for the public deploy, since a public URL can't require visitors to run a local model). Same `app/llm.py` factory, same call sites either way.

**Retrieval** -- ChromaDB, LangChain, HuggingFace `sentence-transformers/all-MiniLM-L6-v2` embeddings

**Orchestration** -- LangChain (rule-based tool calls + true LLM tool-calling for agentic mode)

**Data sources** -- SEC filings via `sec-api`, live market data via `yfinance`

**UI** -- Streamlit

**Testing / quality** -- pytest, ruff

**Containerization** -- Docker (multi-stage: separate `api`/`ui` build targets), Docker Compose for local orchestration

**CI/CD** -- GitHub Actions (lint, then test, then deploy -- gated on tests passing)

---

## Local setup

### Option A -- native (no Docker)

```bash
git clone https://github.com/Siddharth-16/Sovereign-Financial-Analyst.git
cd Sovereign-Financial-Analyst
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install and run [Ollama](https://ollama.ai), then pull a model:

```bash
ollama serve
ollama pull llama3.1
```

Download filings and build the vector store:

```bash
python scripts/data.py
python scripts/ingest.py
```

Section splitting relies on exact heading-string matches, which can fail silently on filings with non-standard HTML formatting. After ingesting, confirm every company times fiscal year actually produced all four sections:

```bash
python scripts/check_coverage.py
```

Run the API and/or UI:

```bash
uvicorn api.main:app --reload   # API at http://localhost:8000/docs
streamlit run ui/ui.py          # UI at http://localhost:8501
```

### Option B -- Docker Compose

Ollama runs natively on your host, not inside a container -- Docker Desktop on macOS/Windows has no GPU access from inside its Linux VM, so a containerized Ollama is CPU-only and noticeably slower. Only the API/UI are containerized; they reach your host's Ollama via `host.docker.internal`.

```bash
ollama serve
ollama pull llama3.2:3b   # or llama3.1 if you have 16GB+ RAM

cp .env.example .env
docker compose up --build api ui
```

- API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

If you're on Linux with a GPU and want Ollama containerized too:

```bash
docker compose --profile containerized-ollama up --build
```

---

## Deployment

The API is containerized separately from the UI (`Dockerfile`, multi-stage, `api`/`ui` targets) so either can be deployed independently.

### Render (no credit card required)

Connect the repo via the Render dashboard: **New -> Blueprint**. Render reads `render.yaml` and creates the service; you'll be prompted once for `GROQ_API_KEY` and `SEC_API_KEY`. The public deploy uses hosted inference (Groq) instead of Ollama, since a public URL can't require visitors to run a local model. Render's free tier doesn't require billing details, sleeps after 15 minutes of inactivity, and takes 30-60 seconds to wake on the next request.

### Fly.io (alternative, requires a card on file)

```bash
flyctl launch --no-deploy --copy-config --name sovereign-financial-analyst
flyctl secrets set GROQ_API_KEY=... SEC_API_KEY=...
flyctl deploy --dockerfile Dockerfile --build-target api
```

### Build the image directly

```bash
docker build --target api -t sovereign-fa-api .
docker build --target ui  -t sovereign-fa-ui .
```

If a `chroma_db/` directory exists locally at build time, it's baked into the image automatically. Otherwise, mount a populated one at runtime.

---

## Environment variables

Full list in `.env.example`. The ones that matter most:

| Variable                      | Default                  | Notes                                                                                                                                                           |
| ----------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLM_PROVIDER`                | `ollama`                 | `ollama` for local/private, `groq` for the public deploy                                                                                                        |
| `OLLAMA_BASE_URL`             | `http://localhost:11434` | Use `host.docker.internal` from inside a container reaching a host Ollama                                                                                       |
| `GROQ_API_KEY` / `GROQ_MODEL` | --                       | Required only when `LLM_PROVIDER=groq`                                                                                                                          |
| `CHROMA_PATH`                 | `./chroma_db`            | Point at a mounted volume in containerized setups                                                                                                               |
| `SEC_API_KEY`                 | --                       | Required for ingestion (`scripts/data.py`, `scripts/ingest.py`); not needed at query time                                                                       |
| `WARMUP_ENABLED`              | `true`                   | Pre-loads the embedding model and LLM client on startup so the first user query isn't slow. Disabled on the Render deploy specifically -- see Known limitations |

---

## CI/CD

`.github/workflows/ci-cd.yml` runs on every push/PR:

1. **Lint** -- `ruff check`, scoped to real bugs (undefined names, unused imports, syntax errors) rather than full style enforcement
2. **Test** -- the 68-test pytest suite, no external services required
3. **Deploy** -- on push to `main` only, after tests pass, via Render's deploy hook -- so deploys are gated on a green test run rather than firing on every push regardless of CI status

---

## Project structure

```
Sovereign-Financial-Analyst/

app/
   agent.py            rule-based query routing + LLM synthesis
   agentic_router.py   LLM tool-calling routing (mode="agentic")
   llm.py              LLM provider factory (ollama / groq)
   tools.py            retrieval + stock tools
   config.py           configuration
   companies.py        single source of truth: company/ticker/section data
   schemas.py          Pydantic request/response models
   exceptions.py       typed exceptions
   logging_config.py   structured JSON logging

api/
   main.py             FastAPI service

ui/
   ui.py               Streamlit interface

eval/
   dataset.py               28-question labeled eval set
   eval.py                    retrieval recall / routing accuracy / LLM-judged groundedness
   report.json                 latest rule_based results
   report_agentic.json         latest agentic results

tests/
   test_routing.py           rule-based routing logic
   test_tools.py              retrieval + stock tools
   test_ingest_splitter.py    section-splitting logic
   test_api.py                 FastAPI endpoints
   conftest.py                  stubs for Ollama/Chroma/yfinance -- no external services needed

scripts/
   data.py                   download SEC 10-K filings
   ingest.py                  filing ingestion + chunking + embedding pipeline
   check_coverage.py          verifies ingestion produced all companies x years x sections
   sync_chroma.py              optional: pull a pre-built chroma_db from S3 on container start

docker/
   entrypoint-api.sh         api container entrypoint
   entrypoint-ui.sh           ui container entrypoint

.github/workflows/
   ci-cd.yml                 lint -> test -> deploy

Dockerfile                   multi-stage build (api / ui targets)
docker-compose.yml            local orchestration
render.yaml                    Render deploy config
fly.toml                        Fly.io deploy config (alternative)
```

---

## Known limitations

Documented here rather than discovered by a reader mid-demo:

- **Render's free tier (512MB) is tight for this stack.** `/health` and light traffic are fine; sustained real query traffic has triggered out-of-memory errors on that tier during testing. The application runs correctly end-to-end -- verified locally and via Docker -- the constraint is specifically the free-tier resource ceiling, not the code.
- **Agentic mode's company-name normalization gap** (see Evaluation above) causes occasional retrieval misses when the LLM's tool-calling arguments include a legal suffix the lookup doesn't normalize away.
- **One documented, repeatable hallucination pattern** on Pfizer risk-factor questions surfaced during evaluation and is tracked as a known limitation rather than silently accepted.
- Financial-statement questions for companies whose statements weren't cleanly extracted during ingestion correctly return that a fact isn't available in the indexed filings rather than guessing -- by design, but worth knowing it happens for a subset of companies.

---

## Possible extensions

- Fix the agentic-mode company-name normalization gap
- Vector reranking on top of the initial similarity search
- Dynamic PDF ingestion from the UI, beyond the pre-ingested 20-company set
- Financial metric extraction into structured (not just prose) output
- Historical trend visualizations
- Earnings call transcript analysis

---

## License

MIT
