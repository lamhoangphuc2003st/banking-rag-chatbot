from __future__ import annotations

import json

import httpx

from apps.api.app.core.config import Settings
from apps.api.app.rag.retrieval.reranker import COHERE_RERANK_URL, Reranker
from packages.shared.schemas import RetrievedChunk


def _settings(**overrides: object) -> Settings:
    defaults = {
        "_env_file": None,
        "reranker_provider": "local",
        "reranker_model": "rerank-v4.0-fast",
        "rag_cache_backend": "memory",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _chunk(chunk_id: str, *, score: float, title: str = "Title", text: str = "Text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=title,
        source_url=f"https://example.com/{chunk_id}",
        section="product_detail",
        product_type="card",
        text=text,
        score=score,
        metadata={"category_title": "Cards"},
    )


async def test_local_reranker_keeps_retrieval_score_order() -> None:
    reranker = Reranker(_settings())

    result = await reranker.rerank(
        "query",
        [
            _chunk("low", score=0.1),
            _chunk("high", score=0.9),
            _chunk("mid", score=0.5),
        ],
        top_k=2,
    )

    assert [chunk.chunk_id for chunk in result] == ["high", "mid"]


async def test_cohere_reranker_uses_provider_ranking() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        assert str(request.url) == COHERE_RERANK_URL
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.98},
                    {"index": 0, "relevance_score": 0.42},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = Reranker(
            _settings(
                reranker_provider="cohere",
                cohere_api_key="test-key",
                reranker_max_documents=3,
            ),
            http_client=client,
        )
        result = await reranker.rerank(
            "mo the tin dung",
            [
                _chunk("low", score=0.1, title="Low", text="low text"),
                _chunk("high", score=0.9, title="High", text="high text"),
                _chunk("mid", score=0.5, title="Mid", text="mid text"),
            ],
            top_k=2,
        )

    assert requests[0]["model"] == "rerank-v4.0-fast"
    assert requests[0]["query"] == "mo the tin dung"
    assert requests[0]["top_n"] == 2
    assert len(requests[0]["documents"]) == 3
    assert [chunk.chunk_id for chunk in result] == ["low", "high"]
    assert [chunk.score for chunk in result] == [0.98, 0.42]


async def test_cohere_reranker_falls_back_without_api_key() -> None:
    reranker = Reranker(_settings(reranker_provider="cohere", cohere_api_key=None))

    result = await reranker.rerank(
        "query",
        [
            _chunk("low", score=0.1),
            _chunk("high", score=0.9),
        ],
        top_k=2,
    )

    assert [chunk.chunk_id for chunk in result] == ["high", "low"]


async def test_cohere_reranker_falls_back_on_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = Reranker(
            _settings(reranker_provider="cohere", cohere_api_key="test-key"),
            http_client=client,
        )
        result = await reranker.rerank(
            "query",
            [
                _chunk("low", score=0.1),
                _chunk("high", score=0.9),
            ],
            top_k=2,
        )

    assert [chunk.chunk_id for chunk in result] == ["high", "low"]
