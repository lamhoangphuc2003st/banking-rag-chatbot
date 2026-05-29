from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceCitation(BaseModel):
    chunk_id: str
    title: str
    source_url: str
    section: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    trace_id: str
    sources: list[SourceCitation] = Field(default_factory=list)
    refusal: bool = False
    latency_ms: int
    metadata: dict[str, Any] = Field(default_factory=dict)
