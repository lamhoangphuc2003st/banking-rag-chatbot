# Evaluation Strategy

The goal of this suite is an evaluation story that is **honest** (realistic
numbers, no fake 1.0s), **reproducible** (regenerates from a clean checkout with
no external services), and **discriminating** (hard cases that actually stress
the system). Live numbers are in [evaluation-results.md](evaluation-results.md);
a manual whole-answer read is in [qualitative-review.md](qualitative-review.md).

Everything here runs offline and deterministically:

```bash
make eval-report      # retrieval + guardrails -> JSON + evaluation-results.md
make eval             # retrieval only (offline lexical baseline)
make eval-guardrails  # guardrails only
make build-golden     # regenerate the retrieval golden set from the corpus
```

## Why offline

The previous suite ranked with the live `HybridRetriever`, which needs a running
Qdrant collection and an embedding provider. That had two failure modes: the
numbers could not be reproduced from the repository, and the golden labels were
copied chunk ids that went **70% stale** after a re-crawl (chunk ids are content
hashes; they change when content changes) — while the committed report still
showed a perfect score from an older index.

Both are fixed by evaluating the *committed corpus*
(`data/chunks/vietcombank_chunks.jsonl`, the same file `make index` upserts into
Qdrant) with the *deployed lexical scorer* (`_rank_lexical_hits` / `_tokenize`
from the production retriever). This measures the real lexical retrieval stage
faithfully, with zero infrastructure. The live path is still available via
`make eval-live` (`--backend hybrid`) when Qdrant is up.

## Retrieval — `packages/evals/retrieval_eval.py`

- **Metrics** at the *document* level (a hit is any chunk from a relevant source
  document): Recall@{1,3,5,10}, MRR, nDCG@10. Document-level relevance is robust
  to re-chunking, which is what stops labels going stale.
- **Difficulty bands** so the report shows a curve, not one inflated number:
  `verbatim` → `no_accent` → `keyword` → `paraphrase` (hard colloquial rewrites
  with a real vocabulary gap). The paraphrase band is the stress test.
- **Negatives:** out-of-scope queries that should retrieve nothing; reported as a
  context-suppression rate.
- **Ground truth is built, not copied.** `packages/evals/build_golden.py` derives
  every label directly from the corpus (queries anchored to real documents), so
  labels cannot silently drift. A test asserts every golden id still exists in
  the corpus.

## Guardrails — `packages/evals/refusal_eval.py` (runs in CI)

Deterministic pure functions (`inspect_query`, `is_likely_supported_domain`), so
no vector store or LLM is needed.

- **Credential/PII blocking:** accuracy, precision, recall, confusion matrix. The
  set includes hard *do-not-over-block* cases — public questions that merely
  mention a sensitive keyword (`OTP là gì?`, `mã CVV nằm ở đâu?`, `quên mật khẩu`).
- **Out-of-scope detection:** accuracy/precision/recall (heuristic; held to a
  realistic bar, not a fake 1.0).

## Answer static checks — `packages/evals/answer_eval.py`

Deterministic checks used to grade generated answers when available: citation
presence on grounded answers, refusal correctness when a refusal is required, and
unsupported high-risk-claim detection (rates/fees/conditions without a source).

## Post-deployment (live) evaluation — `packages/evals/live_eval.py`

The offline suite measures components. Once the backend is deployed (e.g. Render),
some things can only be checked against the running system, over HTTP:

```bash
make eval-live-deploy BASE_URL=https://<service>.onrender.com
# or: python -m packages.evals.live_eval --base-url https://<service>.onrender.com
```

It writes [evaluation-results-live.md](evaluation-results-live.md) and covers:

1. **Availability & readiness** — `/health/live` and `/health/ready` (the latter
   confirms Qdrant, Postgres, Redis and the product graph are actually wired up in
   the deployed environment). This is the first thing to check after any deploy.
2. **End-to-end answers** — real `/v1/chat` calls exercise the *full* production
   stack (guardrails → rewrite/plan → graph + dense/lexical hybrid → Cohere rerank
   → LLM generation → citations) that the offline lexical baseline cannot.
3. **Grounding / citation relevance** — for labelled queries, does the deployed
   answer actually cite the expected source document? (`source_hit_rate`.)
4. **Guardrails end to end** — secret / PII / out-of-scope prompts must be refused
   by the live service, with zero secret/PII leaks.
5. **Latency** — P50/P95/max wall-clock, including Render cold starts (a spun-down
   free/starter service can take ~30–60s on the first request).
6. **Metrics** — scrapes `/metrics` to confirm Prometheus counters
   (`rag_chat_requests_total`, `rag_refusals_total`, `rag_cache_events_total`) are
   live and move with traffic.

Live calls cost real LLM/reranker budget and are slower than offline, so the
harness sends a small stratified sample by default (`--retrieval-limit`); it is a
smoke-and-quality check, not a full sweep.

## Honest limitations

- The reproducible retrieval numbers are a **lexical baseline**. Production adds
  dense embeddings, graph retrieval and a cross-encoder reranker on top, so the
  online system should do at least as well — but that is not what these offline
  numbers measure. Run `make eval-live` against Qdrant to measure the full stack.
- **Scope detection is heuristic.** The remaining misses are short-keyword
  substring collisions (e.g. `thẻ` inside `thế nào`); tracked as future work.
- **Answer faithfulness** (does the answer follow from its citations) is not yet
  scored automatically — it needs an LLM judge and API budget. `deepeval` is
  available in the dev environment as a starting point.

## Planned

- Faithfulness / answer-relevance via an LLM judge over a graded answer set.
- Citation correctness against gold source URLs.
- Token cost per successful answer in an offline report.
- Operational P50/P95 latency and cache-hit rate are already exported live via
  Prometheus (`/metrics`); folding them into a periodic report is pending.

## Quality gates (CI)

- `pytest` runs the offline retrieval eval against reproducible thresholds, the
  golden-integrity check (labels must exist in the corpus), and the guardrail
  golden set (blocking accuracy must be 1.0 with zero false negatives).
- `ruff` and `mypy` must pass.
