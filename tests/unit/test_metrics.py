from __future__ import annotations

from apps.api.app.core.metrics import (
    CACHE_EVENTS,
    CHAT_REQUESTS,
    REFUSALS,
    record_cache_event,
    record_chat_metrics,
)


def test_record_chat_metrics_increments_request_counter() -> None:
    before = CHAT_REQUESTS.labels(endpoint="chat", refusal="false")._value.get()
    record_chat_metrics(
        endpoint="chat",
        refusal=False,
        latency_ms=10,
        metadata={"retrieved_count": 4, "reranked_count": 2},
    )
    after = CHAT_REQUESTS.labels(endpoint="chat", refusal="false")._value.get()
    assert after == before + 1


def test_record_chat_metrics_counts_refusal_reason() -> None:
    before = REFUSALS.labels(reason="out_of_scope")._value.get()
    record_chat_metrics(
        endpoint="chat",
        refusal=True,
        latency_ms=5,
        metadata={"guardrail_reason": "out_of_scope"},
    )
    after = REFUSALS.labels(reason="out_of_scope")._value.get()
    assert after == before + 1


def test_record_cache_event_tracks_hits_and_misses() -> None:
    hit_before = CACHE_EVENTS.labels(event="hit")._value.get()
    miss_before = CACHE_EVENTS.labels(event="miss")._value.get()
    record_cache_event(hit=True)
    record_cache_event(hit=False)
    assert CACHE_EVENTS.labels(event="hit")._value.get() == hit_before + 1
    assert CACHE_EVENTS.labels(event="miss")._value.get() == miss_before + 1
