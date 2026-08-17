# Sovereign Financial Analyst

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED)
![Tests](https://img.shields.io/badge/tests-108-blue)

**Sovereign Financial Analyst** is an evaluation-driven Retrieval-Augmented Generation system for analyzing SEC 10-K filings and live stock data. It answers natural-language questions about business segments, risk factors, MD&A, financial statements, and market performance using retrieved filing evidence and structured market data.

The system indexes **20 public companies** across **fiscal years 2023–2025** and keeps Business, Risk Factors, MD&A, and Financial Statements as separate retrieval domains. It includes both a deterministic rule-based baseline and an LLM-driven agentic router, plus local Ollama and hosted Groq backends through the same LLM interface.

```text
What are Nvidia's supply chain risks?
Compare Nvidia and AMD's risk factors.
What does Microsoft's MD&A say about revenue growth?
How is NVDA stock performing today?
What are Nvidia's main risks and how is its stock performing today?
```

---

## Contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Query routing: rule-based vs. agentic](#query-routing-rule-based-vs-agentic)
- [Retrieval and synthesis](#retrieval-and-synthesis)
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

10-K filings are long, dense, and difficult to navigate. This project retrieves the relevant company, filing section, and fiscal year for a question, then synthesizes an answer from the retrieved evidence instead of relying on the model's parametric knowledge alone.

The project is deliberately end-to-end: SEC ingestion, section-aware chunking, hybrid retrieval, two routing strategies, evidence-aware synthesis, deterministic financial extraction, live market-data lookup, an evaluation harness, REST API, Streamlit UI, automated tests, containerization, and CI/CD.

The evaluation layer is part of the system rather than an afterthought. Retrieval quality, answer completeness, routing, latency, and groundedness are measured separately so improvements in one area don't hide regressions in another.

---

## Architecture

```text
                         +------------------+
                         |  Streamlit UI    |
                         +--------+---------+
                                  |
                         +--------v---------+
                         | FastAPI service  |
                         | /query /health   |
                         | /companies       |
                         +--------+---------+
                                  |
                 +----------------+----------------+
                 |                                 |
        +--------v---------+              +--------v----------+
        | rule_based mode  |              |   agentic mode    |
        | deterministic    |              | LLM tool routing  |
        | intent + section |              | + arg sanitation  |
        +--------+---------+              +--------+----------+
                 |                                 |
                 +----------------+----------------+
                                  |
                 +----------------+----------------+
                 |                                 |
        +--------v---------+              +--------v----------+
        | Filing retrieval |              | Live market data  |
        | Chroma + BM25    |              | yfinance          |
        +--------+---------+              +--------+----------+
                 |                                 |
        +--------v---------+                       |
        | RRF + dedupe     |                       |
        +--------+---------+                       |
                 |                                 |
        +--------v------------------+              |
        | Evidence processing       |              |
        | - prose distillation      |              |
        | - financial extraction    |              |
        +--------+------------------+              |
                 |                                 |
                 +----------------+----------------+
                                  |
                         +--------v---------+
                         | Final response   |
                         | + citations      |
                         +------------------+
```

Ingestion (`scripts/ingest.py`) downloads and processes the SEC 10-K filings, separates Business, Risk Factors, MD&A, and Financial Statements, chunks the text, embeds it, and persists it to ChromaDB with company, fiscal-year, and section metadata.

Retrieval constrains the search space by metadata first, then combines dense vector search with BM25 lexical retrieval. Reciprocal Rank Fusion (RRF) merges the ranked lists and drops duplicate evidence. Before generation, prose synthesis distills that evidence down so the LLM works from a smaller, more relevant context.

Financial-statement questions mostly skip the LLM: deterministic extraction pulls exact values instead of asking the model to reproduce them from long tables. Stock-only and mixed filing/stock queries go through structured yfinance data instead, with market values formatted deterministically so the model can't quietly drop or rewrite a number.

The API and UI are decoupled. FastAPI can be called or deployed on its own, and Streamlit is just a thin client sitting on top of the same application logic.

---

## Query routing: rule-based vs. agentic

The system supports two query-routing strategies.

### `rule_based`

A deterministic baseline in `app/agent.py` uses company aliases, keyword/regex section inference, and explicit filing/stock intent detection. It's fast and reproducible, and it's kept around as a comparison baseline.

### `agentic`

The main agentic path in `app/agentic_router.py` lets the LLM select tools and generate tool arguments. Arguments are sanitized and normalized before retrieval so free-form model output can't directly become an unsafe or invalid lookup.

Both modes ultimately use the same underlying filing and market-data capabilities. The difference is how the system decides what to retrieve and how the final filing answer gets assembled.

---

## Retrieval and synthesis

The filing path uses a hybrid retrieval pipeline rather than vector similarity alone:

```text
Query
  |
  +--> metadata filter (company / section / fiscal year)
  |
  +--> dense semantic retrieval
  |
  +--> BM25 lexical retrieval
          |
          v
     Reciprocal Rank Fusion
          |
          v
        dedupe
          |
          v
  evidence processing / extraction
          |
          v
      final answer
```

This helps with filing language where exact terminology, accounting labels, or risk-factor wording can matter even when semantic similarity is weak.

For prose-heavy sections like Business, Risk Factors, and MD&A, the system builds an evidence brief before synthesis. For financial statements, deterministic extraction is preferred for exact values when the retrieved context supports it.

---

## Evaluation

The system is evaluated on a **28-question development/regression benchmark** spanning Business, Risk Factors, MD&A, and Financial Statements. The benchmark contains **46 gold evidence items** and **79 expected answer facts**.

| Metric                      |    Rule-Based |       Agentic |
| --------------------------- | ------------: | ------------: |
| Company routing accuracy    |        100.0% |        100.0% |
| Section routing accuracy    |        100.0% |        100.0% |
| Metadata retrieval hit rate |        100.0% |        100.0% |
| Evidence retrieval recall@8 | 63.0% (29/46) | 69.6% (32/46) |
| Answer fact completeness    | 43.0% (34/79) | 75.9% (60/71) |
| Tool errors                 |           N/A |             0 |

The agentic pipeline substantially improves answer completeness. The rule-based path is still useful as a deterministic baseline, and it's faster in end-to-end runs.

The benchmark was used during development and regression testing, so these results shouldn't be read as held-out test performance.

See [`eval/comparison.md`](eval/comparison.md) for the compact rule-based vs. agentic comparison and the final JSON artifacts for per-question details.

### Reproduce the evaluation

```bash
python eval/eval.py --mode rule_based --dataset-module dataset --skip-groundedness
python eval/eval.py --mode agentic --dataset-module dataset --skip-groundedness
```

The automated groundedness judge isn't used for the final reported groundedness numbers — small local judge models were too inconsistent to serve as an authoritative evaluator.

---

## Testing

The pytest suite contains **108 tests** covering routing, tool-argument sanitation, hybrid retrieval, evidence processing, deterministic financial answers, ingestion, evaluation metrics, stock tools, and API behavior.

External dependencies like Ollama, ChromaDB, and yfinance are stubbed where it makes sense, so tests can exercise application logic without every external service needing to be live.

```bash
pytest -q
```

Linting:

```bash
ruff check app api ui scripts eval tests
```

---

## Tech stack

Backend and API run on FastAPI with Pydantic and Uvicorn, plus structured JSON logging. Ollama handles local inference, with Groq available through the same provider interface for hosted deployments. Retrieval uses ChromaDB, HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for embeddings, BM25 for lexical search, and Reciprocal Rank Fusion to merge the two. LangChain handles tool calling for the agentic path; the baseline routes deterministically instead. Filing data comes from SEC 10-Ks, market data from `yfinance`. The UI is Streamlit. Tests run on pytest, linting on ruff. Everything is containerized with Docker and Docker Compose, and CI/CD runs through GitHub Actions.

---

## Local setup

### Option A — native

```bash
git clone https://github.com/Siddharth-16/Sovereign-Financial-Analyst.git
cd Sovereign-Financial-Analyst
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install and start [Ollama](https://ollama.ai), then pull the default local model:

```bash
ollama serve
ollama pull llama3.2:3b
```

Download filings and build the vector store:

```bash
python scripts/data.py
python scripts/ingest.py
```

Check ingestion coverage:

```bash
python scripts/check_coverage.py
```

Run the API and/or UI:

```bash
uvicorn api.main:app --reload   # http://localhost:8000/docs
streamlit run ui/ui.py          # http://localhost:8501
```

### Option B — Docker Compose

Ollama can run natively on the host while the API/UI run in containers.

```bash
ollama serve
ollama pull llama3.2:3b

cp .env.example .env
docker compose up --build api ui
```

- API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

If your environment supports containerized Ollama:

```bash
docker compose --profile containerized-ollama up --build
```

---

## Deployment

The API and UI have separate Docker build targets so they can be deployed independently.

The repository includes configuration for hosted deployment. Public deployments should use a hosted inference provider unless the deployment environment can also run Ollama.

Build images directly with:

```bash
docker build --target api -t sovereign-fa-api .
docker build --target ui -t sovereign-fa-ui .
```

If `chroma_db/` exists at build time it can be included in the image; otherwise mount or provision a populated vector store at runtime.

---

## Environment variables

See `.env.example` for the full list.

| Variable                      | Default                  | Notes                                                                 |
| ----------------------------- | ------------------------ | --------------------------------------------------------------------- |
| `LLM_PROVIDER`                | `ollama`                 | `ollama` for local inference; `groq` for hosted inference             |
| `OLLAMA_BASE_URL`             | `http://localhost:11434` | Use `host.docker.internal` when a container connects to host Ollama   |
| `OLLAMA_MODEL`                | `llama3.2:3b`            | Default local generation model                                        |
| `GROQ_API_KEY` / `GROQ_MODEL` | —                        | Required only when `LLM_PROVIDER=groq`                                |
| `CHROMA_PATH`                 | `./chroma_db`            | Local or mounted ChromaDB path                                        |
| `SEC_API_KEY`                 | —                        | Required for ingestion, not for querying an already-built local index |
| `WARMUP_ENABLED`              | `true`                   | Controls application startup warmup behavior                          |

---

## CI/CD

`.github/workflows/ci-cd.yml` runs the quality gates configured for the project on pushes and pull requests:

1. **Lint** — `ruff check app api ui scripts eval tests`
2. **Test** — the 108-test pytest suite
3. **Deploy** — deployment workflow on the configured branch after earlier checks succeed

---

## Project structure

```text
Sovereign-Financial-Analyst/
├── app/
│   ├── agent.py               # deterministic baseline + response orchestration
│   ├── agentic_router.py      # LLM tool-calling router
│   ├── retrieval.py           # dense + BM25 + RRF hybrid retrieval
│   ├── financial_answer.py    # deterministic financial extraction/answers
│   ├── prose_evidence.py      # prose evidence distillation
│   ├── tool_args.py           # tool argument sanitation/normalization
│   ├── tools.py               # filing and stock tools
│   ├── llm.py                 # LLM provider factory
│   ├── companies.py           # company/ticker/section mappings
│   ├── config.py              # configuration
│   ├── schemas.py             # Pydantic models
│   ├── exceptions.py          # typed exceptions
│   └── logging_config.py      # structured logging
│
├── api/
│   └── main.py                # FastAPI service
│
├── ui/
│   └── ui.py                  # Streamlit interface
│
├── eval/
│   ├── dataset.py             # 28-question development benchmark
│   ├── eval.py                # end-to-end evaluation runner
│   ├── metrics.py             # evaluation metrics
│   ├── comparison.md          # final rule-based vs. agentic summary
│   ├── final_rule_based.json  # final per-question baseline results
│   └── final_agentic.json     # final per-question agentic results
│
├── tests/
│   ├── test_api.py
│   ├── test_eval_metrics.py
│   ├── test_financial_answer.py
│   ├── test_hybrid_retrieval.py
│   ├── test_ingest_splitter.py
│   ├── test_prose_evidence.py
│   ├── test_retrieval_prose.py
│   ├── test_routing.py
│   ├── test_tool_args.py
│   ├── test_tool_args_prose.py
│   ├── test_tools.py
│   └── conftest.py
│
├── scripts/
│   ├── data.py
│   ├── ingest.py
|   ├── sync_chroma.py
│   └── check_coverage.py
│
├── .github/workflows/
│   └── ci-cd.yml
│
├── Dockerfile
├── docker-compose.yml
├── render.yaml
└── README.md
```

---

## Known limitations

- The 28-question benchmark is a **development/regression set**, not a held-out test set. Results may overestimate generalization to a completely unseen question distribution.
- Agentic evidence recall is **69.6%**, so retrieval still misses some required evidence even when company/section routing is correct.
- The local `llama3.2:3b` model keeps the project laptop-friendly, but it can still produce incomplete or unsupported prose. The evaluation framework is built to surface those failures rather than hide them.
- SEC filings vary a lot in HTML/table structure, so ingestion quality depends on how cleanly a filing can be sectioned and parsed.
- Live stock queries depend on `yfinance` availability and reflect the latest data that provider returns, not an exchange-grade real-time feed.
- Public hosted deployments may need more memory and a hosted inference provider than the local setup does.

---

## Possible extensions

- Build a larger held-out benchmark for a cleaner estimate of generalization
- Add structured financial-metric extraction across more statement line items
- Support dynamic filing/PDF ingestion from the UI
- Add historical trend visualizations
- Add earnings-call transcript analysis
- Evaluate a stronger local model while keeping the same frozen retrieval benchmark

---

## License

MIT
