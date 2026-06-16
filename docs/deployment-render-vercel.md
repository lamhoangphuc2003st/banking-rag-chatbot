# Render + Vercel Deployment

## Backend: Render

Use the root `render.yaml` as the Render Blueprint. It deploys `bankbot-api` as
a Python web service, runs database migrations in `preDeployCommand`, verifies
runtime data and external services, then starts Uvicorn on Render's `$PORT`.

Required Render environment variables:

- `DATABASE_URL`
- `REDIS_URL`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- `API_CORS_ORIGINS`
- `REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS=2`
- `REDIS_SOCKET_TIMEOUT_SECONDS=5`

Set `API_CORS_ORIGINS` to your Vercel production URL. If the frontend uses the
default Vercel rewrite, browser CORS is mostly avoided, but keeping the origin
locked down is still useful for direct API calls.

Before the first deploy:

1. Generate and review data artifacts with the data pipeline.
2. Commit these runtime artifacts:
   - `data/chunks/vietcombank_chunks.jsonl`
   - `data/chunks/vietcombank_products_chunks.jsonl`
   - `data/chunks/vietcombank_faq_chunks.jsonl`
   - `data/chunks/vietcombank_product_catalogs_chunks.jsonl`
   - `data/normalized/vietcombank_product_catalogs_normalized.jsonl`
3. Create or configure the target Qdrant collection.
4. Run `make index` with production Qdrant credentials.
5. Run `make verify-runtime-external` against the same env values.

Render health checks use `/health/ready`, so deploys fail instead of serving
traffic when graph data, Qdrant, Postgres, or Redis are unavailable.

## Frontend: Vercel

Create a Vercel project with root directory `apps/web`.

Set this Vercel environment variable:

- `BACKEND_API_URL=https://your-render-service.onrender.com`

The client calls `/api/backend/v1/chat/stream` by default. `next.config.mjs`
rewrites `/api/backend/:path*` to `BACKEND_API_URL/:path*`, which keeps the
backend URL out of the client bundle and avoids browser CORS for normal chat
traffic.

Leave `NEXT_PUBLIC_API_BASE_URL` unset in production unless you intentionally
want direct browser-to-Render calls.
