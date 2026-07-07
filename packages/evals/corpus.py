"""Offline corpus access for reproducible evaluation.

The production :class:`~apps.api.app.rag.retrieval.hybrid.HybridRetriever` needs a
running Qdrant instance and an embedding provider. That makes it impossible to
reproduce retrieval numbers from a clean checkout (or in CI), and it is why the
previous golden results could never be regenerated from the committed data.

This module loads the *same* chunk corpus that gets indexed into Qdrant
(``data/chunks/vietcombank_chunks.jsonl``) into memory and ranks it with the
*same* lexical scorer the production retriever uses (``_rank_lexical_hits`` /
``_tokenize``). It therefore measures the deployed lexical retrieval stage
faithfully, while running fully offline with no external services.

The production system layers dense embeddings, graph retrieval and a
cross-encoder reranker on top of this lexical stage, so the online numbers are
expected to be at least as good as what this harness reports.
"""

from __future__ import annotations

import json
from pathlib import Path

from apps.api.app.rag.retrieval.hybrid import _rank_lexical_hits, _tokenize
from packages.shared.schemas import RetrievedChunk

# The merged corpus that ``make index`` upserts into Qdrant. Evaluating against
# this exact file guarantees golden labels can never silently drift from what is
# actually served.
DEFAULT_CORPUS_PATH = Path("data/chunks/vietcombank_chunks.jsonl")


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> list[RetrievedChunk]:
    """Load the committed chunk corpus as retrievable candidates."""

    chunks: list[RetrievedChunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(payload["chunk_id"]),
                    document_id=str(payload.get("document_id") or ""),
                    title=str(payload.get("title") or ""),
                    source_url=str(payload.get("source_url") or ""),
                    section=payload.get("section"),
                    product_type=payload.get("product_type"),
                    text=str(payload.get("text") or ""),
                    score=0.0,
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
    return chunks


class LexicalCorpusRetriever:
    """Rank the local corpus with the production lexical scorer.

    Exposes the same ``retrieve`` coroutine shape as
    :class:`~apps.api.app.rag.retrieval.hybrid.HybridRetriever` so the evaluation
    driver can treat the offline and live backends interchangeably.
    """

    def __init__(self, corpus: list[RetrievedChunk]) -> None:
        self._corpus = corpus

    async def retrieve(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        return _rank_lexical_hits(_tokenize(query), self._corpus, top_k)


def corpus_chunk_ids(corpus: list[RetrievedChunk]) -> set[str]:
    return {chunk.chunk_id for chunk in corpus}


def chunk_ids_for_source_url(corpus: list[RetrievedChunk]) -> dict[str, list[str]]:
    """Group chunk ids by their source document URL (the relevance unit)."""

    grouped: dict[str, list[str]] = {}
    for chunk in corpus:
        if not chunk.source_url:
            continue
        grouped.setdefault(chunk.source_url, []).append(chunk.chunk_id)
    return grouped
