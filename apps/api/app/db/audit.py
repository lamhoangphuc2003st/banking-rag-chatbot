from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.db.models import ChatAuditLog
from apps.api.app.models.chat import ChatRequest, ChatResponse


async def save_chat_audit(
    session: AsyncSession,
    request: ChatRequest,
    response: ChatResponse,
) -> None:
    query = next((message.content for message in reversed(request.messages) if message.role == "user"), "")
    record = ChatAuditLog(
        session_id=response.session_id,
        trace_id=response.trace_id,
        query=query,
        answer=response.answer,
        sources={"items": [source.model_dump() for source in response.sources]},
        latency_ms=response.latency_ms,
    )
    session.add(record)
    await session.commit()
