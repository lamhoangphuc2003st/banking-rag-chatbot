# Live Deployment Evaluation — Render backend

Post-deployment evaluation of the deployed API
(`https://bankbot-api.onrender.com`), run with `python -m packages.evals.live_eval`
plus a local pipeline reproduction to root-cause failures. Unlike the offline
suite, this exercises the **full production stack** (guardrails → rewrite/plan →
graph + dense/lexical hybrid → Cohere rerank → LLM generation → citations) over
HTTP. Last run: 2026-07-07, after the generation-robustness fix was deployed and
the OpenAI quota restored.

## Summary

| Area | Verdict |
| --- | --- |
| Availability & dependency readiness | ✅ healthy (all deps green) |
| Retrieval & grounding (full stack) | ✅ source-hit 0.875, citation 0.94, **0 server errors** |
| Guardrails: credential/PII blocking | ✅ 0 leaks, deterministic |
| LLM generation robustness | ✅ fixed & deployed — provider errors degrade to a fallback |
| Operational: OpenAI quota | ✅ restored |
| Out-of-scope detection | ✅ improved — planner verdict now refuses translation/off-topic (was 2/8 answered) |
| Prometheus `/metrics` | ✅ live, counters move with traffic |

**The post-deploy evaluation earned its keep:** it surfaced a critical robustness
bug (LLM/provider errors returned HTTP 500) that the offline suite cannot see. The
bug was root-caused, fixed, unit-tested, deployed, and re-verified live — the
latest full run over 52 requests has **zero server errors**.

## 1. Availability & readiness ✅

- `GET /health/live` → HTTP 200; `GET /health/ready` → HTTP 200, every dependency `ok`:

| Dependency | Status | Detail |
| --- | --- | --- |
| Product graph | ok | 30 categories, 74 products, 74 detail sources |
| Qdrant | ok | collection `vietcombank_public_docs` |
| Postgres | ok | |
| Redis | ok | |

## 2. Retrieval & grounding quality ✅

16 golden queries stratified across difficulty bands, sent to `/v1/chat` (full
hybrid + Cohere rerank + LLM):

| Metric | Value |
| --- | ---: |
| Request success rate (HTTP 200) | **1.00** (0 server errors) |
| Answer rate (non-refusal, non-empty) | 1.00 |
| Citation rate (≥1 source) | 0.94 |
| Source-hit rate (answer cited the expected document) | **0.875** |
| Latency P50 / P95 / max | 7.8s / 12.8s / 54s |

Source-hit rate is end-to-end: a query counts only when the deployed answer cites
one of the labelled relevant source documents. 0.875 with the full dense+rerank
stack is a clear lift over the offline lexical baseline on the same hard queries.
The 54s max is a one-off cold start on the freshly-deployed instance; warm P50 is
~8s. Worst-case latency is driven by "compare across all products" queries — see
recommendation 2.

## 3. LLM generation robustness — bug found, fixed & verified ✅

**What the evaluation found.** In an earlier run the OpenAI quota was exhausted,
and *every* request that reached LLM generation returned **HTTP 500**, while
requests that did not call the LLM (secret/PII blocked at the guardrail,
out-of-scope refused before generation) returned 200. The failure tracked "does
this query call the LLM", nothing else.

**Root cause (reproduced locally).** The generation call was unguarded, so a
provider exception propagated out of the request handler:

```
litellm.exceptions.RateLimitError: OpenAIException - insufficient_quota (429)
  apps/api/app/rag/generation/llm.py   await acompletion(...)          # not guarded
  apps/api/app/rag/pipeline.py         async for token in stream_answer(...)  # not guarded
  → propagated out of stream_events() / answer() → FastAPI returned HTTP 500
```

Any transient LLM condition (quota, rate limit, timeout, provider outage) became
a hard 500 for the user.

**Fix (deployed).** `pipeline.stream_events` now wraps the `stream_answer` loop:
a provider error is logged, the answer degrades to a graceful fallback, citations
are suppressed, and `metadata.generation_failed` flags it — the request completes
as HTTP 200. Verified three ways:

- Unit regression test `test_pipeline_degrades_gracefully_when_generation_fails`.
- Local end-to-end against the (then still-exhausted) quota → graceful fallback
  for every query instead of an unhandled `RateLimitError`.
- Live re-run after deploy → **0 server errors** across all 52 requests.

## 4. Guardrails end-to-end ✅ / ⚠️

- **Credential / PII blocking:** 10/10 secret & PII prompts blocked at the guardrail
  in ~0.1s (before any model call). **0 leaks**, deterministic.
- **Out-of-scope:** 6/8 refused end to end at eval time. The 2 that slipped
  through were a translation request (actively answered) and a movie
  recommendation (redirected via a clarification). **Fixed since:** the explicit
  planner out-of-scope verdict now takes priority over collision-prone
  keyword-overlap evidence, so the translation request is refused (verified
  end to end). See recommendation 1.

## 5. Operational ✅

OpenAI quota restored; generation works. Note that the evaluation's token-heavy
"exhaustive multi-product" queries were what drained the previous budget and also
drive worst-case latency — a usage/budget alert is recommended.

## 6. Metrics endpoint ✅

`GET /metrics` is live; counters move with traffic (values below are for one eval
run on the fresh instance):

| Counter | Sample |
| --- | --- |
| `rag_chat_requests_total` | refusal=false = 36, refusal=true = 16 |
| `rag_refusals_total` | credential_or_secret = 10, out_of_scope = 6 |
| `rag_cache_events_total` | miss = 38 (cold instance, cache empty) |

## Remaining recommendations

1. ✅ **Tighten out-of-scope detection** (§4) — *done: the planner's explicit
   out-of-scope verdict now overrides weak keyword-overlap evidence, so a
   translation request is refused instead of answered. Verified end to end.*
2. Bound "exhaustive multi-product" generation (token + latency budget) to cap P95
   and control cost; add an OpenAI budget alert.

## Reproduce

```bash
python -m packages.evals.live_eval --base-url https://bankbot-api.onrender.com
# or: make eval-live-deploy BASE_URL=https://bankbot-api.onrender.com
```
