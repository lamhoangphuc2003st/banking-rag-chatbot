from __future__ import annotations

from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.core.logging import get_logger
from packages.shared.schemas import RetrievedChunk

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid retriever interface.

    The first implementation keeps provider calls isolated so the project can switch
    between Qdrant Cloud, local Qdrant, OpenSearch, or PostgreSQL lexical search
    without changing the API layer.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 12,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        vector_hits = await self._vector_search(query, top_k=top_k, filters=filters)
        lexical_hits = await self._lexical_search(query, top_k=top_k, filters=filters)
        merged = self._merge_hits(vector_hits, lexical_hits)
        return merged[:top_k]

    async def _vector_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        if not self.settings.openai_api_key and self.settings.embedding_provider == "openai":
            logger.warning("vector_search_skipped", reason="missing_embedding_api_key")
            return []

        try:
            query_vector = await self._embed_query(query)
            if not query_vector:
                return []

            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Filter

            client = AsyncQdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=10,
            )

            qdrant_filter = Filter(**filters) if filters else None
            results = await client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:  # pragma: no cover - provider/network boundary
            logger.warning("vector_search_failed", error=str(exc))
            return []

        chunks: list[RetrievedChunk] = []
        for item in results:
            payload = item.payload or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id") or item.id),
                    document_id=str(payload.get("document_id") or ""),
                    title=str(payload.get("title") or "Vietcombank"),
                    source_url=str(payload.get("source_url") or ""),
                    section=payload.get("section"),
                    product_type=payload.get("product_type"),
                    text=str(payload.get("text") or ""),
                    score=float(item.score),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return chunks

    async def _lexical_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        # Placeholder for PostgreSQL full-text or OpenSearch. Keeping the method
        # explicit makes hybrid retrieval measurable once the store is connected.
        _ = (query, top_k, filters)
        return []

    async def _embed_query(self, query: str) -> list[float]:
        from litellm import aembedding

        response = await aembedding(
            model=self.settings.embedding_model,
            input=[query],
            api_key=self.settings.openai_api_key,
        )
        return list(response["data"][0]["embedding"])

    def _merge_hits(
        self,
        vector_hits: list[RetrievedChunk],
        lexical_hits: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}

        for chunk in vector_hits + lexical_hits:
            existing = merged.get(chunk.chunk_id)
            if existing is None or (chunk.score or 0) > (existing.score or 0):
                merged[chunk.chunk_id] = chunk

        return sorted(merged.values(), key=lambda item: item.score or 0, reverse=True)
