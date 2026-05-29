from __future__ import annotations

from apps.api.app.core.config import Settings
from packages.shared.schemas import RetrievedChunk


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int = 6,
    ) -> list[RetrievedChunk]:
        _ = query
        # Provider-specific rerankers can be added here. The default keeps the
        # retrieval score ordering so the pipeline is deterministic in local mode.
        return sorted(chunks, key=lambda item: item.score or 0, reverse=True)[:top_k]
