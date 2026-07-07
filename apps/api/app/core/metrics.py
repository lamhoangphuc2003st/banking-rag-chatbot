"""Application-level Prometheus metrics for the RAG service.

These instruments are exported by the ``/metrics`` endpoint so that request
volume, latency, retrieval depth, refusals and cache efficiency can be scraped
by Prometheus and visualised (see ``infra/prometheus``). Recording is
best-effort: a metrics failure must never affect a chat response.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from prometheus_client import Counter, Histogram

CHAT_REQUESTS = Counter(
    "rag_chat_requests_total",
    "Chat requests handled by the RAG pipeline.",
    labelnames=("endpoint", "refusal"),
)
CHAT_LATENCY = Histogram(
    "rag_chat_latency_seconds",
    "End-to-end chat latency in seconds.",
    labelnames=("endpoint",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
)
RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Chunks retrieved before reranking.",
    buckets=(0, 1, 2, 4, 8, 12, 16, 24, 32, 48),
)
RERANKED_CHUNKS = Histogram(
    "rag_reranked_chunks",
    "Chunks kept as context after reranking.",
    buckets=(0, 1, 2, 4, 6, 8, 12, 16, 24, 32),
)
REFUSALS = Counter(
    "rag_refusals_total",
    "Refused or out-of-scope answers, labelled by reason.",
    labelnames=("reason",),
)
CACHE_EVENTS = Counter(
    "rag_cache_events_total",
    "Retrieval cache hits and misses.",
    labelnames=("event",),
)


def record_chat_metrics(
    *,
    endpoint: str,
    refusal: bool,
    latency_ms: int,
    metadata: dict[str, Any],
) -> None:
    """Record one completed chat turn. Best-effort; never raises."""

    with suppress(Exception):  # pragma: no cover - metrics must never break a response
        CHAT_REQUESTS.labels(endpoint=endpoint, refusal=str(refusal).lower()).inc()
        CHAT_LATENCY.labels(endpoint=endpoint).observe(max(latency_ms, 0) / 1000.0)

        retrieved = metadata.get("retrieved_count")
        if isinstance(retrieved, int):
            RETRIEVED_CHUNKS.observe(retrieved)

        reranked = metadata.get("reranked_count")
        if isinstance(reranked, int):
            RERANKED_CHUNKS.observe(reranked)

        if refusal:
            reason = str(metadata.get("guardrail_reason") or "unknown")
            REFUSALS.labels(reason=reason).inc()


def record_cache_event(*, hit: bool) -> None:
    """Record a retrieval-cache hit or miss. Best-effort; never raises."""

    with suppress(Exception):  # pragma: no cover - metrics must never break a response
        CACHE_EVENTS.labels(event="hit" if hit else "miss").inc()
