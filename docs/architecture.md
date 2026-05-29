# Architecture

## Runtime Components

- `apps/api`: FastAPI service exposing chat, streaming chat, health, and diagnostics endpoints.
- `packages/data_pipeline`: offline and scheduled jobs for crawl, normalize, chunk, and index.
- `packages/evals`: repeatable evaluation jobs.
- `apps/web`: chat UI that streams model output and displays sources.
- `PostgreSQL`: sessions, audit logs, evaluation records, crawl metadata.
- `Redis`: rate limits, exact cache, short-lived session state.
- `Qdrant`: vector search over public Vietcombank chunks.

## RAG Flow

```text
User query
  -> input guardrails
  -> intent and scope routing
  -> optional rewrite
  -> hybrid retrieval
  -> reranking
  -> context packing
  -> grounded generation
  -> output guardrails
  -> citations + audit log
```

## Design Rules

- Do not put business facts directly in prompts; facts must come from retrieved data.
- Store prompt version, model version, data version, and retrieved chunk IDs for each answer.
- Prefer provider abstraction so OpenAI, Gemini, Claude, or self-hosted models can be swapped.
- Treat crawl output as versioned data, not as code.
