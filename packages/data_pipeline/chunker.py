from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from packages.shared.schemas import DocumentChunk, NormalizedDocument


def chunk_file(input_path: Path, output_path: Path, max_chars: int = 1200, overlap: int = 160) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            document = NormalizedDocument.model_validate_json(line)
            for chunk in chunk_document(document, max_chars=max_chars, overlap=overlap):
                target.write(chunk.model_dump_json() + "\n")
                count += 1
    return count


def chunk_document(
    document: NormalizedDocument,
    *,
    max_chars: int = 1200,
    overlap: int = 160,
) -> list[DocumentChunk]:
    paragraphs = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", document.text) if item.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current} {paragraph}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph[-max_chars:] if len(paragraph) > max_chars else paragraph

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap)

    return [
        DocumentChunk(
            chunk_id=stable_chunk_id(document.document_id, index, text),
            document_id=document.document_id,
            title=document.title,
            source_url=document.source_url,
            text=text,
            content_hash=document.content_hash,
            language=document.language,
            product_type=document.product_type,
            section=document.section,
            chunk_index=index,
            metadata=document.metadata,
        )
        for index, text in enumerate(chunks)
    ]


def stable_chunk_id(document_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{index}:{text}".encode("utf-8")).hexdigest()
    return digest[:32]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    output = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:], strict=False):
        prefix = previous[-overlap:].strip()
        output.append(f"{prefix} {current}".strip())
    return output


def read_chunks(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(DocumentChunk.model_validate(json.loads(line)))
    return chunks
