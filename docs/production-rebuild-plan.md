# Production Build Plan

## Goal

Build a production-grade RAG chatbot for public Vietcombank information that demonstrates practical AI Engineer skills: data engineering, retrieval, LLM orchestration, evaluation, deployment, and observability.

## Phases

1. Foundation: monorepo structure, Docker Compose, settings, logging, API contracts.
2. Data pipeline: crawl, normalize, validate, deduplicate, chunk, version artifacts.
3. Indexing: embeddings, Qdrant vector index, lexical index, metadata filters.
4. RAG service: guardrails, intent routing, query rewriting, hybrid retrieval, reranking, grounded generation, citations.
5. Evaluation: golden dataset, retrieval metrics, answer metrics, regression report.
6. Frontend: streaming chat, source display, feedback collection.
7. Production hardening: auth, rate limit, PII redaction, audit logs, observability.
8. Deployment: Docker, Kubernetes, CI/CD, load test, final documentation.

## Acceptance Criteria

- The crawler is polite, reproducible, and stores source provenance.
- The chatbot refuses unsupported or unsafe requests.
- Answers include citations and never invent unsupported rates, fees, or eligibility rules.
- Evaluation can be run in CI and fails on regression.
- API has health checks, structured logs, rate limiting hooks, and environment-based configuration.
- Deployment assets are available for local Docker and Kubernetes.
