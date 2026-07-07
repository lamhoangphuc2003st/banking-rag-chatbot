from __future__ import annotations

from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage, ChatRequest
from apps.api.app.rag.pipeline import RagPipeline, _context_top_k_for_plan
from apps.api.app.rag.planner import QueryPlan
from apps.api.app.rag.query_decompose import QueryDecomposer, QueryDecompositionResult
from apps.api.app.rag.retrieval.graph import GraphRetrievalResult, GraphSubjectOption
from packages.shared.schemas import RetrievedChunk


class FakeGraphRetriever:
    def __init__(self) -> None:
        self.options = (
            GraphSubjectOption(
                title="Vietcombank Vibe Platinum",
                subject_type="product",
                url="https://www.vietcombank.com.vn/vibe-platinum",
                product_type="card",
                category_title="Thẻ tín dụng",
            ),
            GraphSubjectOption(
                title="Vietcombank Mastercard® Debit",
                subject_type="product",
                url="https://www.vietcombank.com.vn/mastercard-debit",
                product_type="card",
                category_title="Thẻ thanh toán",
            ),
        )

    def match_subjects(self, query: str, *, limit: int = 8) -> tuple[GraphSubjectOption, ...]:
        normalized = query.casefold()
        matches = [
            option
            for option in self.options
            if _subject_key(option.title) in _subject_key(normalized)
        ]
        return tuple(matches[:limit])

    def retrieve(
        self,
        query: str,
        *,
        history: list[ChatMessage],
        top_k: int = 12,
    ) -> GraphRetrievalResult:
        _ = history, top_k
        chunks = [
            RetrievedChunk(
                chunk_id=f"graph:{_subject_key(option.title).replace(' ', '-')}",
                document_id=f"doc:{_subject_key(option.title).replace(' ', '-')}",
                title=option.title,
                source_url=option.url,
                text=f"{option.title}: context for {query}",
                score=2.0,
                section="product_detail",
                product_type="card",
                metadata={"retrieval_source": "graph"},
            )
            for option in self.match_subjects(query, limit=8)
        ]
        return GraphRetrievalResult(chunks=chunks, route="graph" if chunks else "default")

    def latest_subject(self, history: list[ChatMessage]) -> str | None:
        _ = history
        return None

    def suggest_subjects(self, query: str, *, limit: int = 5) -> tuple[GraphSubjectOption, ...]:
        return self.match_subjects(query, limit=limit)


class FakeRetriever:
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 12,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        _ = top_k, filters
        title = (
            "Vietcombank Mastercard® Debit"
            if "Mastercard" in query
            else "Vietcombank Vibe Platinum"
        )
        return [
            RetrievedChunk(
                chunk_id=f"vector:{_subject_key(title).replace(' ', '-')}",
                document_id=f"doc:{_subject_key(title).replace(' ', '-')}",
                title=title,
                source_url=f"https://www.vietcombank.com.vn/{_subject_key(title).replace(' ', '-')}",
                text=f"{title}: retrieved evidence for {query}",
                score=1.0,
                section="product_detail",
                product_type="card",
            )
        ]

    async def scroll_by_filter(
        self,
        *,
        filters: dict[str, Any],
        limit: int = 1000,
    ) -> list[RetrievedChunk]:
        _ = filters, limit
        return []


class FakeLLM:
    def __init__(self) -> None:
        self.question = ""
        self.chunks: list[RetrievedChunk] = []

    async def generate_answer(self, **kwargs: Any) -> str:
        self.question = str(kwargs["question"])
        self.chunks = list(kwargs["chunks"])
        return "Vietcombank Vibe Platinum và Vietcombank Mastercard® Debit"

    async def stream_answer(self, **kwargs: Any) -> Any:
        self.question = str(kwargs["question"])
        self.chunks = list(kwargs["chunks"])
        yield "Vietcombank Vibe Platinum và Vietcombank Mastercard® Debit"


def test_query_decomposer_splits_multi_field_multi_product_question() -> None:
    graph_retriever = FakeGraphRetriever()
    decomposer = QueryDecomposer(Settings(_env_file=None, llm_provider="local"))

    result = _run(
        decomposer.decompose(
            question=(
                "Lợi ích và điều kiện mở thẻ của thẻ Vietcombank Vibe Platinum "
                "và Vietcombank Mastercard® Debit"
            ),
            history=[],
            graph_result=GraphRetrievalResult(chunks=[], route="default"),
            query_plan=QueryPlan(
                intent="direct_answer",
                route="planner_direct",
                reason="single_step_retrieval_is_sufficient",
            ),
            graph_retriever=graph_retriever,  # type: ignore[arg-type]
        )
    )

    assert result.route == "local_field_subject"
    assert list(result.subqueries) == [
        "Lợi ích và ưu đãi của Vietcombank Vibe Platinum là gì?",
        "Điều kiện mở hoặc sử dụng của Vietcombank Vibe Platinum là gì?",
        "Lợi ích và ưu đãi của Vietcombank Mastercard® Debit là gì?",
        "Điều kiện mở hoặc sử dụng của Vietcombank Mastercard® Debit là gì?",
    ]


async def test_pipeline_passes_grouped_subquery_context_to_llm() -> None:
    pipeline = RagPipeline(
        Settings(_env_file=None, llm_provider="local", rag_cache_backend="memory")
    )
    fake_llm = FakeLLM()
    pipeline.graph_retriever = FakeGraphRetriever()  # type: ignore[assignment]
    pipeline.retriever = FakeRetriever()  # type: ignore[assignment]
    pipeline.llm = fake_llm  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Lợi ích và điều kiện mở thẻ của thẻ Vietcombank Vibe Platinum "
                        "và Vietcombank Mastercard® Debit"
                    ),
                )
            ]
        )
    )

    assert response.metadata["query_decomposition_applied"] is True
    assert response.metadata["subquery_count"] == 4
    assert "Câu hỏi đã được tách" in fake_llm.question
    assert {chunk.metadata.get("subquery") for chunk in fake_llm.chunks} == set(
        response.metadata["subqueries"]
    )


def test_decomposition_context_budget_keeps_each_subquery_context() -> None:
    query_plan = QueryPlan(
        intent="direct_answer",
        route="planner_direct",
        reason="single_step_retrieval_is_sufficient",
        context_top_k=6,
    )
    decomposition = QueryDecompositionResult(
        original_query="complex",
        subqueries=("a", "b", "c", "d"),
        route="local_field_subject",
    )

    assert _context_top_k_for_plan(query_plan, decomposition) == 20


def _subject_key(text: str) -> str:
    return (
        text.casefold()
        .replace("®", "")
        .replace("vietcombank", "")
        .replace("thẻ", "")
        .strip()
    )


def _run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
