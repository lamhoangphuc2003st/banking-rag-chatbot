"""API-level tests for the FastAPI app.

These drive the real ASGI app with ``httpx.AsyncClient`` and override the
pipeline, database session and rate limiter dependencies so the HTTP contract
(request validation, response model, streaming, metrics, audit best-effort) is
exercised end-to-end without any external services.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from apps.api.app.db.session import get_db_session
from apps.api.app.main import app, enforce_rate_limit, get_pipeline
from apps.api.app.models.chat import ChatResponse

ANSWER_TEXT = "Vietcombank hỗ trợ vay mua nhà."


class FakePipeline:
    async def answer(self, request: Any) -> ChatResponse:
        return ChatResponse(
            answer=ANSWER_TEXT,
            session_id=request.session_id,
            trace_id="trace-test",
            sources=[],
            refusal=False,
            latency_ms=7,
            metadata={"retrieved_count": 3, "reranked_count": 2},
        )

    async def stream_events(self, request: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "token", "content": "Vietcombank "}
        yield {"type": "token", "content": "hỗ trợ vay mua nhà."}
        yield {"type": "sources", "sources": []}
        yield {
            "type": "metadata",
            "trace_id": "trace-test",
            "refusal": False,
            "latency_ms": 7,
            "metadata": {"retrieved_count": 3, "reranked_count": 2},
        }


class FakeSession:
    def add(self, *_: Any) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


async def _fake_db_session() -> AsyncIterator[FakeSession]:
    yield FakeSession()


@pytest.fixture(autouse=True)
def _override_dependencies() -> Iterator[None]:
    app.dependency_overrides[get_pipeline] = FakePipeline
    app.dependency_overrides[get_db_session] = _fake_db_session
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    try:
        yield
    finally:
        app.dependency_overrides.clear()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_live_returns_ok() -> None:
    async with _client() as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_chat_returns_pipeline_answer() -> None:
    async with _client() as client:
        response = await client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Vay mua nhà thế nào?"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ANSWER_TEXT
    assert body["refusal"] is False
    assert body["metadata"]["retrieved_count"] == 3


async def test_chat_rejects_empty_messages() -> None:
    async with _client() as client:
        response = await client.post("/v1/chat", json={"messages": []})
    assert response.status_code == 422


async def test_chat_stream_emits_server_sent_events() -> None:
    async with _client() as client:
        response = await client.post(
            "/v1/chat/stream",
            json={"messages": [{"role": "user", "content": "Vay mua nhà thế nào?"}]},
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data:" in response.text
    assert "hỗ trợ vay mua nhà" in response.text


async def test_metrics_endpoint_exposes_rag_series() -> None:
    async with _client() as client:
        await client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Vay mua nhà thế nào?"}]},
        )
        response = await client.get("/metrics")
    assert response.status_code == 200
    assert "rag_chat_requests_total" in response.text
