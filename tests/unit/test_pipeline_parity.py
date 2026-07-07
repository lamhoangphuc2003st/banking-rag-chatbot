"""The non-streaming ``answer()`` must never diverge from ``stream_events()``.

``answer()`` is implemented by buffering ``stream_events()``. This test locks
that contract in on the guardrail-refusal path, which short-circuits before any
retriever or LLM call and therefore needs no external services.
"""

from __future__ import annotations

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage, ChatRequest
from apps.api.app.rag.pipeline import RagPipeline


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="local",
        rag_cache_backend="memory",
        openai_api_key=None,
        litellm_api_key=None,
    )


async def test_answer_matches_stream_events_on_guardrail_refusal() -> None:
    pipeline = RagPipeline(_settings())
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Mật khẩu của tôi là abc12345")]
    )

    streamed_tokens: list[str] = []
    metadata_event: dict[str, object] = {}
    async for event in pipeline.stream_events(request):
        if event.get("type") == "token":
            streamed_tokens.append(str(event.get("content") or ""))
        elif event.get("type") == "metadata":
            metadata_event = event

    response = await pipeline.answer(request)

    assert response.refusal is True
    assert response.answer == "".join(streamed_tokens)
    assert response.refusal == bool(metadata_event.get("refusal"))
    assert response.metadata == dict(metadata_event.get("metadata") or {})
