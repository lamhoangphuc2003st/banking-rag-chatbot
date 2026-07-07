# Evaluation Strategy

Metrics are split into what is **computed today** (reproducible from this repo) and what is
**planned**, to keep the evaluation story honest.

## Computed today

### Retrieval — `packages/evals/retrieval_eval.py` (`make eval`, needs Qdrant)

- Recall@k
- MRR
- nDCG@k
- Per-case hit/rank breakdown by product type and section

### Guardrails — `packages/evals/refusal_eval.py` (`make eval-guardrails`, offline, runs in CI)

- Credential/PII blocking: accuracy, precision, recall, confusion matrix
- Out-of-scope detection: accuracy, precision, recall
- Golden set: `data/golden/guardrail_golden.jsonl`; report: `data/reports/guardrail_eval.json`

### Answer static checks — `packages/evals/answer_eval.py`

- Citation presence on grounded answers
- Refusal correctness when a refusal is required
- Unsupported high-risk-claim detection (rates/fees/conditions without a source)

## Planned

- Faithfulness / answer-relevance via an LLM judge (needs an LLM and API budget)
- Citation correctness against gold source URLs
- **Retrieval golden-set expansion** with adversarial and negative queries — the current
  35-query set is small and self-labelled, which yields unrealistic ~1.0 scores
- Token cost per successful answer, surfaced in an offline report
- Operational P50/P95 latency and cache-hit rate are exported live via Prometheus
  (`/metrics`); folding them into a periodic report is pending

## Quality gates (CI)

- `make eval-guardrails` fails if credential/PII blocking accuracy drops below its threshold.
- `pytest` covers unit + API behaviour and the guardrail golden set.
- `ruff` and `mypy` must pass.
