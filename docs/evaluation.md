# Evaluation Strategy

## Retrieval Metrics

- Recall@5, Recall@10, Recall@20
- MRR
- nDCG
- Coverage by product type and section

## Answer Metrics

- Faithfulness to retrieved context
- Answer relevance
- Citation correctness
- Refusal correctness for unsafe and out-of-scope prompts
- Vietnamese fluency and formatting

## Operational Metrics

- P50/P95 latency
- Token cost per successful answer
- Cache hit rate
- Retrieval empty rate
- User feedback score

## Quality Gates

CI should fail when:

- Retrieval Recall@10 drops below the configured threshold.
- Citation correctness regresses.
- Unsafe prompts are answered instead of refused.
- Unit tests or schema validation fail.
