from __future__ import annotations

import json
from pathlib import Path

from apps.api.app.core.config import Settings
from packages.shared.schemas import DocumentChunk


class QdrantIndexer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def index_file(self, chunks_path: Path, batch_size: int = 64) -> int:
        chunks = _read_chunks(chunks_path)
        if not chunks:
            return 0

        vectors = await self._embed([chunk.text for chunk in chunks])

        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = AsyncQdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=30,
        )
        await client.recreate_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "source_url": chunk.source_url,
                        "text": chunk.text,
                        "content_hash": chunk.content_hash,
                        "language": chunk.language,
                        "product_type": chunk.product_type,
                        "section": chunk.section,
                        "metadata": chunk.metadata,
                    },
                )
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            await client.upsert(collection_name=self.settings.qdrant_collection, points=points)
            total += len(points)
        return total

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.settings.embedding_provider == "openai" and not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")

        from litellm import aembedding

        response = await aembedding(
            model=self.settings.embedding_model,
            input=texts,
            api_key=self.settings.openai_api_key,
        )
        return [list(item["embedding"]) for item in response["data"]]


def _read_chunks(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(DocumentChunk.model_validate(json.loads(line)))
    return chunks
