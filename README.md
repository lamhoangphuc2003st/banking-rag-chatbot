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
