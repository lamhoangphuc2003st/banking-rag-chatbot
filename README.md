# Vietcombank RAG Platform

Production-oriented chatbot platform for public Vietcombank product information. The system is rebuilt as a portfolio-grade AI Engineer project: data pipeline, hybrid retrieval, guarded RAG, evaluation, observability, CI/CD, and deployment assets.

## Scope

- Public information only: loans, credit cards, fees, requirements, FAQ, and support pages.
- No banking transactions, account lookup, balance lookup, credential handling, or personalized financial advice.
- Every answer should be grounded in retrieved sources and include citations.

## Architecture

```text
Vietcombank Website
  -> Polite crawler
  -> Raw/normalized data store
  -> Schema validation
  -> Semantic chunking
  -> Hybrid index: vector + lexical + metadata
  -> FastAPI RAG service
  -> Guardrails + retrieval + rerank + generation
  -> Web chat + admin/evaluation views
  -> Logs, traces, metrics, evaluation reports
```

## Tech Stack

- API: Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy, Alembic.
- RAG: LangGraph-ready pipeline, Qdrant, Redis, PostgreSQL, configurable LLM/embedding/reranker providers.
- Data: httpx/Playwright-ready crawler, Pydantic validation, reproducible chunk/index commands.
- LLMOps: evaluation package, trace-friendly schemas, prompt/model/data versioning.
- Infra: Docker Compose, production Dockerfile, Kubernetes manifests, GitHub Actions.
- Web: Next.js, TypeScript, streaming chat UI.

## Quick Start

```bash
cp .env.example .env
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev,eval,scraping]"
docker compose up -d postgres redis qdrant
make api
```

Run the web app:

```bash
cd apps/web
npm install
npm run dev
```

## Data Pipeline

```bash
make crawl
make chunk
make index
```

Each pipeline stage writes versioned artifacts under `data/` and records source URL, content hash, crawl time, product type, section, and language.

## Evaluation

```bash
make eval
pytest
```

Quality gates should include retrieval recall, MRR, citation correctness, refusal correctness, faithfulness, latency, and token cost.

## Repository Layout

```text
apps/api/                  FastAPI RAG service
apps/web/                  Next.js chat UI
packages/data_pipeline/    crawler, normalizer, chunker, indexer
packages/evals/            retrieval and answer quality evaluation
packages/shared/           shared schemas and utilities
infra/                     Kubernetes, monitoring, deployment assets
docs/                      architecture, data governance, evaluation
tests/                     unit and integration tests
```
