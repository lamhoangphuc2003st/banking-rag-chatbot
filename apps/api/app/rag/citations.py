from __future__ import annotations

from apps.api.app.models.chat import SourceCitation
from packages.shared.schemas import RetrievedChunk


def build_citations(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    seen: set[str] = set()

    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        citations.append(
            SourceCitation(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source_url=chunk.source_url,
                section=chunk.section,
                score=chunk.score,
            )
        )

    return citations
