from __future__ import annotations

import time
from uuid import uuid4

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatRequest, ChatResponse
from apps.api.app.rag.citations import build_citations
from apps.api.app.rag.generation.llm import LLMClient
from apps.api.app.rag.guardrails import inspect_query, is_likely_supported_domain
from apps.api.app.rag.retrieval.hybrid import HybridRetriever
from apps.api.app.rag.retrieval.reranker import Reranker


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever = HybridRetriever(settings)
        self.reranker = Reranker(settings)
        self.llm = LLMClient(settings)

    async def answer(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        trace_id = str(uuid4())
        question = self._latest_user_message(request)

        guardrail = inspect_query(question)
        if not guardrail.allowed:
            return ChatResponse(
                answer=guardrail.safe_response or "Tôi không thể xử lý yêu cầu này.",
                session_id=request.session_id,
                trace_id=trace_id,
                refusal=True,
                latency_ms=self._elapsed_ms(started),
                metadata={"guardrail_reason": guardrail.reason},
            )

        if not is_likely_supported_domain(question):
            return ChatResponse(
                answer=(
                    "Tôi chỉ hỗ trợ tra cứu thông tin công khai liên quan đến Vietcombank "
                    "trong phạm vi dữ liệu đã được index."
                ),
                session_id=request.session_id,
                trace_id=trace_id,
                refusal=True,
                latency_ms=self._elapsed_ms(started),
                metadata={"guardrail_reason": "out_of_scope"},
            )

        retrieved = await self.retriever.retrieve(question, top_k=12)
        reranked = await self.reranker.rerank(question, retrieved, top_k=6)
        history = self._format_history(request)
        answer = await self.llm.generate_answer(
            question=question,
            history=history,
            chunks=reranked,
        )

        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            trace_id=trace_id,
            sources=build_citations(reranked),
            refusal=False,
            latency_ms=self._elapsed_ms(started),
            metadata={
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "llm_model": self.settings.llm_model,
                "embedding_model": self.settings.embedding_model,
                "data_collection": self.settings.qdrant_collection,
            },
        )

    async def stream(self, request: ChatRequest):
        response = await self.answer(request)
        yield response.answer

    def _latest_user_message(self, request: ChatRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content.strip()
        return ""

    def _format_history(self, request: ChatRequest) -> str:
        history = request.messages[:-1][-8:]
        return "\n".join(f"{item.role}: {item.content}" for item in history)

    def _elapsed_ms(self, started: float) -> int:
        return int((time.perf_counter() - started) * 1000)
