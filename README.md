# Vietcombank RAG Platform

[![CI](https://github.com/lamhoangphuc2003st/banking-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/lamhoangphuc2003st/banking-rag-chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A portfolio-grade Retrieval-Augmented Generation platform that answers questions about
**public Vietcombank products** (loans, cards, savings, transfers, insurance, FAQ) in
Vietnamese — every answer grounded in crawled sources with citations, protected by input
guardrails, and measured by an evaluation suite.

> ⚠️ Unofficial project for public information only. Not affiliated with Vietcombank.
> It performs no transactions and never handles balances, credentials, or personal accounts.

## Demo

<!-- TODO: after deploying (see "Deploy" below), paste your live URL and embed a GIF. -->

- **Live demo:** _add your Vercel/Render URL here._
- **Walkthrough:** _record a 10–15s clip (query → streamed answer → citations), save it to
  `docs/assets/demo.gif`, and embed it here._

## Scope

- Public information only: loans, credit cards, fees, requirements, FAQ, and support pages.
- No banking transactions, account lookup, balance lookup, credential handling, or personalized financial advice.
- Every answer should be grounded in retrieved sources and include citations.

## Architecture

```mermaid
flowchart LR
    subgraph Offline["Data pipeline (offline)"]
        W[Vietcombank site] --> C[Polite crawler] --> N[Normalize + validate]
        N --> K[Semantic chunking] --> IX[(Qdrant index<br/>vector + lexical + metadata)]
    end
    subgraph Online["RAG service (FastAPI)"]
        Q[User query] --> G[Guardrails] --> R[Rewrite / plan / decompose]
        R --> H[Graph + hybrid retrieval] --> RR[Rerank] --> GEN[Grounded generation]
        GEN --> A[Answer + citations]
    end
    IX --- H
    A --> AUD[(Postgres audit)]
    Online --> OBS[Prometheus /metrics + structured logs]
    UI[Next.js streaming chat] --> Q
```

The online pipeline is hand-built async orchestration (no LangGraph) so every stage —
guardrails, query rewrite/planning/decomposition, graph + hybrid retrieval, reranking,
grounded generation — is explicit and independently testable. The non-streaming `answer()`
and streaming `stream_events()` share one implementation, so they can never diverge.

## Tech Stack

- API: Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy, Alembic.
- RAG: custom async orchestration (guardrails → rewrite/plan/decompose → graph + hybrid retrieval → rerank → grounded generation), Qdrant, Redis, PostgreSQL, configurable LLM/embedding/reranker providers via LiteLLM.
- Data: httpx + BeautifulSoup crawler, Pydantic validation, reproducible chunk/index commands.
- LLMOps: offline evaluation (retrieval + guardrails), Prometheus metrics, structured logging.
- Infra: Docker Compose, production Dockerfile, Kubernetes manifests, GitHub Actions CI.
- Web: Next.js, TypeScript, streaming (SSE) chat UI with citations and clarifications.

## Quick Start

```bash
cp .env.example .env
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
docker compose up -d postgres qdrant
make api
```

Before starting the API, set `REDIS_URL` in `.env` to your managed Redis
connection string. Local and production should use the same Redis server; this
repo does not create a Redis container in Docker Compose.

Run the web app:

```bash
cd apps/web
npm install
npm run dev
```

The web app calls `/api/backend/*` by default. Next.js rewrites that path to
`BACKEND_API_URL`, defaulting to `http://localhost:8000`, so local development
does not require exposing a public API URL in the browser bundle.

## Data Pipeline

```bash
make crawl
make crawl-catalogs
make normalize
make normalize-catalogs
make chunk
make chunk-catalogs
make merge-chunks
make index
```

Each pipeline stage writes versioned artifacts under `data/` and records source URL, content hash, crawl time, product type, section, and language.

The API also reads GraphRAG catalog/detail artifacts from `data/` at runtime. In containers, mount the generated data directory to `/app/data` and set `RAG_DATA_ROOT=/app/data`; the Docker image intentionally does not embed raw/normalized/chunk artifacts.

Use `make index` for non-destructive upserts into the configured Qdrant collection. Use `python -m packages.data_pipeline.cli index --recreate` only when you intentionally want to drop and rebuild the collection.

## Redis

Redis is an external managed service for this project. Create one Redis instance
with a provider such as Upstash, Redis Cloud, AWS ElastiCache, Azure Cache for
Redis, or another managed Redis provider. Copy its connection string into:

- Local: `.env` as `REDIS_URL=...`
- Production: the `REDIS_URL` value in your secret manager or Kubernetes Secret

The API uses that same Redis endpoint for RAG cache and rate limiting. Nothing
in this repo starts or owns the Redis server.

If Redis is in another region or a free managed tier, increase the socket
timeouts instead of disabling Redis cache:

```bash
REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS=2
REDIS_SOCKET_TIMEOUT_SECONDS=5
```

## Deploy: Render + Vercel

Backend deploys as a Render Python web service from `render.yaml`. Frontend
deploys as a Vercel Next.js app with root directory `apps/web`.

Before deploying, commit the runtime data artifacts that are intentionally
unignored:

- `data/chunks/vietcombank_chunks.jsonl`
- `data/chunks/vietcombank_products_chunks.jsonl`
- `data/chunks/vietcombank_faq_chunks.jsonl`
- `data/chunks/vietcombank_product_catalogs_chunks.jsonl`
- `data/normalized/vietcombank_product_catalogs_normalized.jsonl`

Backend Render environment variables:

- `DATABASE_URL`
- `REDIS_URL`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- `API_CORS_ORIGINS=https://your-vercel-app.vercel.app`

The Render pre-deploy command runs Alembic migrations and
`python -m packages.data_pipeline.cli verify-runtime --check-external`. Make
sure the Qdrant collection has already been indexed with `make index`; the
pre-deploy check fails if the collection is missing or empty.

Frontend Vercel environment variables:

- `BACKEND_API_URL=https://your-render-service.onrender.com`

Do not set `NEXT_PUBLIC_API_BASE_URL` in production unless you intentionally
want the browser to call Render directly. The default Vercel rewrite avoids
baking the backend URL into client code.

## Cohere Rerank

To enable Cohere reranking, set:

```bash
RERANKER_PROVIDER=cohere
RERANKER_MODEL=rerank-v4.0-fast
COHERE_API_KEY=...
```

If the Cohere key is missing or the provider call fails, the API falls back to
the local retrieval-score ordering so chat responses still complete.

## Evaluation

```bash
make eval             # retrieval: Recall@k, MRR, nDCG@k over a golden set (needs Qdrant)
make eval-guardrails  # guardrails: credential/PII blocking + scope, fully offline
pytest                # unit + API tests
```

Retrieval quality (`packages/evals/retrieval_eval.py`) reports Recall@k, MRR and nDCG@k
against `data/golden/retrieval_golden.jsonl`. Guardrail quality
(`packages/evals/refusal_eval.py`) is deterministic and runs in CI. Answer-quality static
checks live in `packages/evals/answer_eval.py`. See
[docs/evaluation.md](docs/evaluation.md) for what is computed today versus planned.

## Results

Reproduce with `make eval` / `make eval-guardrails`.

**Guardrails** — 18 labelled prompts (safe / secret-disclosure / PII / out-of-scope),
offline and deterministic (`data/reports/guardrail_eval.json`):

| Check | Accuracy | Precision | Recall |
| --- | --- | --- | --- |
| Credential / PII blocking | 1.00 | 1.00 | 1.00 |
| Out-of-scope detection | 0.92 | 0.89 | 1.00 |

No secret or PII disclosure is ever allowed through (false-negative rate 0). The single
scope miss is a known heuristic limitation (a short keyword collision), tracked as future work.

**Retrieval** — 35-query golden set: Recall@10 = 1.00, MRR = 1.00, nDCG@10 = 1.00.

> These perfect scores mean the golden set is currently too small and self-labelled.
> Expanding it with harder, adversarial and negative queries — and reporting realistic
> (< 1.0) numbers — is the top evaluation priority.

## Observability

- `GET /metrics` — Prometheus metrics: request volume and latency (`rag_chat_latency_seconds`),
  retrieved/reranked context depth, refusals by reason, and retrieval cache hit/miss. Scrape
  config in [`infra/prometheus`](infra/prometheus).
- `GET /health/live` and `GET /health/ready` (readiness checks Qdrant, Postgres, Redis and the
  product graph).
- Structured JSON logs (structlog) to stdout; every answer carries a `trace_id`.

## Repository Layout

```text
apps/api/                  FastAPI RAG service
apps/web/                  Next.js chat UI
packages/data_pipeline/    crawler, normalizer, chunker, indexer
packages/evals/            retrieval, guardrail, and answer-quality evaluation
packages/shared/           shared schemas and utilities
infra/                     Kubernetes, monitoring, deployment assets
docs/                      architecture, data governance, evaluation
tests/                     unit and API tests
```

## License

Released under the [MIT License](LICENSE).
