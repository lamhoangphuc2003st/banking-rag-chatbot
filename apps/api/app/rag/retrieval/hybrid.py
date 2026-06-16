from __future__ import annotations

import asyncio
import inspect
import json
import re
import unicodedata
from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.core.logging import get_logger
from apps.api.app.rag.cache import AsyncTTLCache, CacheBackend, RedisTTLCache
from packages.shared.schemas import RetrievedChunk

logger = get_logger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
VIETNAMESE_STOPWORDS = {
    "a",
    "anh",
    "bao",
    "bi",
    "cac",
    "can",
    "cho",
    "co",
    "cua",
    "duoc",
    "gi",
    "hang",
    "hay",
    "khach",
    "khong",
    "la",
    "lam",
    "nhu",
    "ngan",
    "o",
    "phai",
    "qua",
    "ra",
    "sao",
    "tai",
    "the",
    "thi",
    "toi",
    "trong",
    "va",
    "vcb",
    "vietcombank",
}


class HybridRetriever:
    """Hybrid retriever interface.

    The first implementation keeps provider calls isolated so the project can switch
    between Qdrant Cloud, local Qdrant, OpenSearch, or PostgreSQL lexical search
    without changing the API layer.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        cache_ttl = settings.rag_cache_ttl_seconds if settings.rag_cache_enabled else 0.0
        cache_entries = settings.rag_cache_max_entries
        self._retrieval_cache = self._create_cache(
            "retrieval",
            max_entries=cache_entries,
            ttl_seconds=cache_ttl,
            encode=_encode_chunks,
            decode=_decode_chunks,
        )
        self._embedding_cache = self._create_cache(
            "embedding",
            max_entries=cache_entries,
            ttl_seconds=cache_ttl,
            encode=_encode_floats,
            decode=_decode_floats,
        )
        self._scroll_cache = self._create_cache(
            "scroll",
            max_entries=max(16, min(cache_entries, 128)),
            ttl_seconds=cache_ttl,
            encode=_encode_chunks,
            decode=_decode_chunks,
        )
        self._qdrant_client: Any | None = None

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 12,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        cache_key = _cache_key(
            "hybrid_retrieve",
            self.settings.qdrant_collection,
            query,
            top_k,
            filters,
        )
        cached = await self._retrieval_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        vector_hits, lexical_hits = await asyncio.gather(
            self._vector_search(query, top_k=top_k, filters=filters),
            self._lexical_search(query, top_k=top_k, filters=filters),
        )
        merged = self._merge_hits(query, vector_hits, lexical_hits)[:top_k]
        if merged:
            await self._retrieval_cache.set(cache_key, tuple(merged))
        return merged

    async def scroll_by_filter(
        self,
        *,
        filters: dict[str, Any],
        limit: int = 1000,
    ) -> list[RetrievedChunk]:
        return await self._scroll_chunks(filters=filters, limit=limit)

    async def close(self) -> None:
        await asyncio.gather(
            self._retrieval_cache.close(),
            self._embedding_cache.close(),
            self._scroll_cache.close(),
            return_exceptions=True,
        )
        client = self._qdrant_client
        self._qdrant_client = None
        if client is None:
            return

        close = getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    def _create_cache(
        self,
        namespace: str,
        *,
        max_entries: int,
        ttl_seconds: float,
        encode: Any,
        decode: Any,
    ) -> CacheBackend:
        backend = self.settings.rag_cache_backend.strip().casefold()
        if backend == "redis":
            if not self.settings.redis_url.strip():
                raise ValueError("REDIS_URL is required when RAG_CACHE_BACKEND=redis.")
            return RedisTTLCache(
                redis_url=self.settings.redis_url,
                namespace=f"bank-chatbot:rag:{namespace}",
                ttl_seconds=ttl_seconds,
                encode=encode,
                decode=decode,
            )
        if backend not in {"", "memory", "inmemory", "local"}:
            logger.warning("unknown_rag_cache_backend", backend=backend, fallback="memory")
        return AsyncTTLCache(max_entries=max_entries, ttl_seconds=ttl_seconds)

    async def _scroll_chunks(
        self,
        *,
        filters: dict[str, Any] | None,
        limit: int,
    ) -> list[RetrievedChunk]:
        cache_key = _cache_key(
            "qdrant_scroll",
            self.settings.qdrant_collection,
            filters,
            limit,
        )
        cached = await self._scroll_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        try:
            from qdrant_client.models import Filter

            qdrant_filter = Filter(**filters) if filters else None
            records, _ = await self._get_qdrant_client().scroll(
                collection_name=self.settings.qdrant_collection,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # pragma: no cover - provider/network boundary
            logger.warning("filter_scroll_failed", error=str(exc))
            return []

        chunks = [
            _payload_to_chunk(payload=record.payload or {}, item_id=record.id, score=0.0)
            for record in records
        ]
        await self._scroll_cache.set(cache_key, tuple(chunks))
        return chunks

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

            from qdrant_client.models import Filter

            qdrant_filter = Filter(**filters) if filters else None
            response = await self._get_qdrant_client().query_points(
                collection_name=self.settings.qdrant_collection,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            results = response.points
        except Exception as exc:  # pragma: no cover - provider/network boundary
            logger.warning("vector_search_failed", error=str(exc))
            return []

        chunks: list[RetrievedChunk] = []
        for item in results:
            payload = item.payload or {}
            chunks.append(
                _payload_to_chunk(
                    payload=payload,
                    item_id=item.id,
                    score=float(item.score),
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
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        candidates = await self._scroll_chunks(filters=filters, limit=1000)
        return await asyncio.to_thread(_rank_lexical_hits, query_tokens, candidates, top_k)

    async def _embed_query(self, query: str) -> list[float]:
        cache_key = _cache_key("embedding", self.settings.embedding_model, query)
        cached = await self._embedding_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        from litellm import aembedding

        response = await aembedding(
            model=self.settings.embedding_model,
            input=[query],
            api_key=self.settings.openai_api_key,
        )
        embedding = list(response["data"][0]["embedding"])
        await self._embedding_cache.set(cache_key, tuple(embedding))
        return embedding

    def _get_qdrant_client(self) -> Any:
        if self._qdrant_client is None:
            from qdrant_client import AsyncQdrantClient

            self._qdrant_client = AsyncQdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=int(self.settings.qdrant_request_timeout_seconds),
            )
        return self._qdrant_client

    def _merge_hits(
        self,
        query: str,
        vector_hits: list[RetrievedChunk],
        lexical_hits: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}

        for chunk in vector_hits + lexical_hits:
            existing = merged.get(chunk.chunk_id)
            if existing is None or (chunk.score or 0) > (existing.score or 0):
                merged[chunk.chunk_id] = chunk

        return self._rerank_by_lexical_match(query, list(merged.values()))

    def _rerank_by_lexical_match(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return sorted(chunks, key=lambda item: item.score or 0, reverse=True)

        ranked: list[RetrievedChunk] = []
        for chunk in chunks:
            lexical_score = _lexical_match_score(query_tokens, chunk)
            if lexical_score:
                chunk = chunk.model_copy(update={"score": (chunk.score or 0.0) + lexical_score})
            ranked.append(chunk)

        return sorted(ranked, key=lambda item: item.score or 0, reverse=True)


def _lexical_match_score(query_tokens: set[str], chunk: RetrievedChunk) -> float:
    title_tokens = _tokenize(chunk.title)
    if not title_tokens:
        return 0.0

    title_overlap = len(query_tokens & title_tokens)
    title_coverage = title_overlap / len(title_tokens)
    query_coverage = title_overlap / len(query_tokens)
    text_tokens = _tokenize(chunk.text[:1200])
    text_coverage = len(query_tokens & text_tokens) / len(query_tokens) if text_tokens else 0.0

    score = (0.25 * title_coverage) + (0.08 * query_coverage) + (0.03 * text_coverage)
    if len(title_tokens) >= 2 and title_tokens.issubset(query_tokens):
        score += 0.45
    return score


def _rank_lexical_hits(
    query_tokens: set[str],
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for chunk in candidates:
        lexical_score = _lexical_match_score(query_tokens, chunk)
        if lexical_score >= 0.12:
            chunks.append(chunk.model_copy(update={"score": lexical_score}))

    return sorted(chunks, key=lambda item: item.score or 0, reverse=True)[:top_k]


def _payload_to_chunk(payload: dict[str, Any], item_id: Any, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id") or item_id),
        document_id=str(payload.get("document_id") or ""),
        title=str(payload.get("title") or "Vietcombank"),
        source_url=str(payload.get("source_url") or ""),
        section=payload.get("section"),
        product_type=payload.get("product_type"),
        text=str(payload.get("text") or ""),
        score=score,
        metadata=dict(payload.get("metadata") or {}),
    )


def _cache_key(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)}"


def _encode_chunks(value: Any) -> str:
    chunks = [chunk.model_dump(mode="json") for chunk in value]
    return json.dumps(chunks, ensure_ascii=False, separators=(",", ":"))


def _decode_chunks(payload: str) -> tuple[RetrievedChunk, ...]:
    values = json.loads(payload)
    if not isinstance(values, list):
        raise ValueError("cached chunks payload must be a list")
    return tuple(RetrievedChunk.model_validate(value) for value in values)


def _encode_floats(value: Any) -> str:
    return json.dumps([float(item) for item in value], separators=(",", ":"))


def _decode_floats(payload: str) -> tuple[float, ...]:
    values = json.loads(payload)
    if not isinstance(values, list):
        raise ValueError("cached embedding payload must be a list")
    return tuple(float(item) for item in values)


def _tokenize(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("đ", "d")
    return {
        token
        for token in TOKEN_PATTERN.findall(ascii_text)
        if len(token) > 1 and token not in VIETNAMESE_STOPWORDS
    }
