# Live Deployment Evaluation — Render backend

Post-deployment evaluation of the deployed API (`https://bankbot-api.onrender.com`)
on 2026-07-07, run with `python -m packages.evals.live_eval` plus a local
pipeline reproduction to root-cause failures. Unlike the offline suite, this
exercises the **full production stack** (guardrails → rewrite/plan → graph +
dense/lexical hybrid → Cohere rerank → LLM generation → citations) over HTTP.

## Summary

| Area | Verdict |
| --- | --- |
| Availability & dependency readiness | ✅ healthy |
| Retrieval & grounding (LLM available) | ✅ good — source-hit 0.875, citation 0.94 |
| Guardrails: credential/PII blocking | ✅ 0 leaks, deterministic |
| LLM generation robustness | ✅ **fixed** — provider errors now degrade to a fallback (was 🔴 HTTP 500; redeploy required) |
| **Operational: OpenAI quota** | 🔴 **exhausted (`insufficient_quota`)** |
| Out-of-scope detection | ⚠️ heuristic; 2 misses, no fabrication |
| Prometheus `/metrics` | ✅ live, counters move with traffic |

The post-deploy evaluation did its job: it surfaced a **critical robustness bug
and an operational issue that the offline suite cannot see**.

## 1. Availability & readiness ✅

- `GET /health/live` → HTTP 200.
- `GET /health/ready` → HTTP 200, every dependency `ok`:

| Dependency | Status | Detail |
| --- | --- | --- |
| Product graph | ok | 30 categories, 74 products, 74 detail sources |
| Qdrant | ok | collection `vietcombank_public_docs` |
| Postgres | ok | |
| Redis | ok | |

The deployment is correctly wired to all external services.

## 2. Retrieval & grounding quality ✅ (measured while the LLM was available)

16 golden queries stratified across difficulty bands sent to `/v1/chat` (full
hybrid + Cohere rerank + LLM). Measured before the OpenAI quota was exhausted:

| Metric | Value |
| --- | ---: |
| Request success rate (HTTP 200) | 1.00 |
| Answer rate (non-refusal, non-empty) | 1.00 |
| Citation rate (≥1 source) | 0.94 |
| Source-hit rate (answer cited the expected document) | **0.875** |
| Latency P50 / P95 / max | 7.9s / 13.4s / 17.7s |

Source-hit rate is end-to-end: a query only counts when the deployed answer
cites one of the labelled relevant source documents. 0.875 with the full
dense+rerank stack is a clear lift over the offline lexical baseline on the same
hard queries, confirming the production retrieval stack adds value.

## 3. 🔴 Critical: LLM/provider errors surface as HTTP 500

**Symptom.** Every request that reaches LLM generation returns **HTTP 500** once
the model provider errors. In a fault-injected run (with the OpenAI quota
exhausted, see §5) the split was exact:

| Query type | Reaches LLM? | Result |
| --- | --- | ---: |
| Retrieval answers (16) | yes | 15 × HTTP 500 |
| Safe / sensitive-keyword answers (18) | yes | HTTP 500 |
| Secret / PII (10) | no — blocked at guardrail | ✅ 200 (refused) |
| Out-of-scope (8) | no — refused before generation | ✅ 200 |

The failure tracks **"does this query call the LLM"**, nothing else.

**Root cause (reproduced locally).** The generation call is not wrapped in any
error handling, so a provider exception propagates out of the request handler:

```
litellm.exceptions.RateLimitError: OpenAIException - insufficient_quota (429)
  apps/api/app/rag/generation/llm.py:89   await acompletion(...)   # not guarded
  apps/api/app/rag/pipeline.py:538        async for token in self.llm.stream_answer(...)  # not guarded
  → propagates out of stream_events() / answer() → FastAPI returns HTTP 500
```

**Impact.** *Any* transient LLM condition — quota, rate limit, timeout, provider
outage — becomes a hard 500 for the user instead of a graceful message. This is
the single most important fix.

**Fix (applied & verified).** `pipeline.stream_events` now wraps the
`stream_answer` loop: a provider error is logged, the answer degrades to a
graceful fallback message, citations are suppressed, and `metadata.generation_failed`
is set — the request completes as HTTP 200 instead of crashing.

- Unit regression test: `test_pipeline_degrades_gracefully_when_generation_fails`
  (a `stream_answer` that raises must yield a fallback, `refusal=False`, no 500).
- End-to-end: re-running the pipeline against the still-exhausted OpenAI quota now
  returns the fallback for every query (previously an unhandled `RateLimitError`).

> ⚠️ The fix is in the codebase but **not yet on Render** — redeploy the backend
> (and restore the OpenAI quota, §5) for the live service to stop returning 500s.

## 4. Guardrails end-to-end ✅ / ⚠️

- **Credential / PII blocking:** 10/10 secret & PII prompts blocked at the guardrail
  in ~0.1s (before any model call). **0 leaks**, deterministic, unaffected by the
  LLM outage — the safety layer is robust.
- **Out-of-scope:** 6/8 refused. The 2 misses (`thời tiết … thế nào`,
  `… lễ … thế nào`) are the known short-keyword substring collision (`thẻ`
  inside `thế`); notably they return the "no information found" fallback rather
  than a fabricated answer.

## 5. 🔴 Operational: OpenAI quota exhausted

Generation now fails with `insufficient_quota` (HTTP 429 from OpenAI). The
evaluation itself consumed the remaining budget — the "exhaustive multi-product"
queries (e.g. *compare conditions across all credit cards*) are token-heavy and
also drove worst-case latency (~48–55s). Actions:

- Restore/upgrade the OpenAI billing plan; add a usage/budget alert.
- Consider capping exhaustive multi-product expansion (token + latency budget).

## 6. Metrics endpoint ✅

`GET /metrics` is live and counters move with traffic:

| Counter | Sample |
| --- | --- |
| `rag_chat_requests_total` | chat/refusal=false = 35, chat/refusal=true = 32 |
| `rag_refusals_total` | credential_or_secret = 20, out_of_scope = 12 |
| `rag_cache_events_total` | hit = 6, miss = 78 |

## Recommendations (priority order)

1. ✅ **Handle LLM/provider errors gracefully** (§3) — *done in code + tested; redeploy to apply.*
2. **Restore OpenAI quota + add budget alerting** (§5).
3. Tighten the out-of-scope heuristic to remove the `thẻ`/`thế` collision (§4).
4. Bound exhaustive multi-product generation to cap p95 latency and token cost.

## Reproduce

```bash
# Full live evaluation (writes data/reports/live_eval.{json,md}):
python -m packages.evals.live_eval --base-url https://bankbot-api.onrender.com
# or: make eval-live-deploy BASE_URL=https://bankbot-api.onrender.com
```
