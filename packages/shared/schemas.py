from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    source_url: str
    html: str
    status_code: int
    content_hash: str
    crawled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    document_id: str
    source_url: str
    title: str
    text: str
    content_hash: str
    crawled_at: datetime
    language: str = "vi"
    product_type: str | None = None
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source_url: str
    text: str
    content_hash: str
    language: str = "vi"
    product_type: str | None = None
    section: str | None = None
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source_url: str
    text: str
    score: float | None = None
    section: str | None = None
    product_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
