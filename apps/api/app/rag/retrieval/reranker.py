from __future__ import annotations

from typing import Any

import httpx

from apps.api.app.core.config import Settings
from apps.api.app.core.logging import get_logger
from packages.shared.schemas import RetrievedChunk

logger = get_logger(__name__)

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
DEFAULT_COHERE_RERANK_MODEL = "rerank-v4.0-fast"


class Reranker:
    def __init__(self, settings: Settings, *, http_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._http_client = http_client
        self._owns_http_client = http_client is None

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int = 6,
    ) -> list[RetrievedChunk]:
        if not chunks or top_k <= 0:
            return []

        provider = self.settings.reranker_provider.strip().casefold()
        if provider not in {"cohere", "cohere_rerank"}:
            return _local_rerank(chunks, top_k=top_k)

        if not self.settings.cohere_api_key:
            logger.warning("cohere_rerank_skipped", reason="missing_api_key")
            return _local_rerank(chunks, top_k=top_k)

        try:
            return await self._cohere_rerank(query, chunks, top_k=top_k)
        except Exception as exc:  # pragma: no cover - provider/network boundary
            logger.warning("cohere_rerank_failed", error=str(exc))
            return _local_rerank(chunks, top_k=top_k)

    async def close(self) -> None:
        if self._http_client is not None and self._owns_http_client:
            await self._http_client.aclose()
        self._http_client = None

    async def _cohere_rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        max_documents = max(1, int(self.settings.reranker_max_documents))
        candidates = _local_rerank(chunks, top_k=min(len(chunks), max_documents))
        documents = [_chunk_to_rerank_document(chunk) for chunk in candidates]
        response = await self._client().post(
            COHERE_RERANK_URL,
            headers={
                "Authorization": f"Bearer {self.settings.cohere_api_key}",
                "Content-Type": "application/json",
                "X-Client-Name": "bank-chatbot",
            },
            json={
                "model": self.settings.reranker_model.strip() or DEFAULT_COHERE_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),
            },
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("Cohere rerank response missing results list.")

        ranked: list[RetrievedChunk] = []
        selected_indices: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            index = _coerce_int(result.get("index"))
            if index is None or index < 0 or index >= len(candidates) or index in selected_indices:
                continue
            selected_indices.add(index)
            relevance_score = _coerce_float(result.get("relevance_score"))
            chunk = candidates[index]
            if relevance_score is not None:
                chunk = chunk.model_copy(update={"score": relevance_score})
            ranked.append(chunk)

        if not ranked:
            raise ValueError("Cohere rerank response did not contain usable rankings.")

        if len(ranked) >= top_k:
            return ranked[:top_k]

        ranked_ids = {chunk.chunk_id for chunk in ranked}
        fallback = [chunk for chunk in _local_rerank(chunks, top_k=top_k) if chunk.chunk_id not in ranked_ids]
        return [*ranked, *fallback][:top_k]

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(float(self.settings.reranker_request_timeout_seconds)),
                follow_redirects=True,
            )
        return self._http_client


def _local_rerank(chunks: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
    # The local fallback keeps retrieval score ordering so local/test runs are deterministic.
    return sorted(chunks, key=lambda item: item.score or 0, reverse=True)[:top_k]


def _chunk_to_rerank_document(chunk: RetrievedChunk) -> str:
    metadata_lines = [
        f"title: {_single_line(chunk.title)}",
        f"section: {_single_line(chunk.section or 'unknown')}",
    ]
    if chunk.product_type:
        metadata_lines.append(f"product_type: {_single_line(chunk.product_type)}")
    category_title = chunk.metadata.get("category_title")
    if isinstance(category_title, str) and category_title.strip():
        metadata_lines.append(f"category_title: {_single_line(category_title)}")
    subquery = chunk.metadata.get("subquery")
    if isinstance(subquery, str) and subquery.strip():
        metadata_lines.append(f"subquery: {_single_line(subquery)}")
    metadata_lines.append("content: |-")
    metadata_lines.extend(f"  {line}" for line in _document_text(chunk).splitlines())
    return "\n".join(metadata_lines)


def _document_text(chunk: RetrievedChunk) -> str:
    text = " ".join(chunk.text.split())
    if len(text) <= 4000:
        return text
    return f"{text[:3997].rstrip()}..."


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
