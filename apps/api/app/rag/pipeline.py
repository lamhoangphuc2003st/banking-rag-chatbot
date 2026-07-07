from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from apps.api.app.core.config import Settings
from apps.api.app.core.logging import get_logger
from apps.api.app.models.chat import ChatMessage, ChatRequest, ChatResponse, SourceCitation
from apps.api.app.rag.citations import build_citations
from apps.api.app.rag.exchange_rates import ExchangeRateService
from apps.api.app.rag.generation.llm import LLMClient
from apps.api.app.rag.guardrails import inspect_query, is_likely_supported_domain
from apps.api.app.rag.planner import QueryPlan, QueryPlanner
from apps.api.app.rag.query_decompose import QueryDecomposer, QueryDecompositionResult
from apps.api.app.rag.query_rewrite import QueryRewriter, QueryRewriteResult
from apps.api.app.rag.retrieval.graph import (
    GraphRetrievalResult,
    GraphSubjectOption,
    ProductGraphRetriever,
)
from apps.api.app.rag.retrieval.hybrid import HybridRetriever
from apps.api.app.rag.retrieval.reranker import Reranker
from packages.shared.schemas import RetrievedChunk

logger = get_logger(__name__)

CATALOG_QUERY_MARKERS = (
    "bao gom",
    "cac goi",
    "cac goi tren",
    "cac san pham",
    "co gi",
    "co cac",
    "co nhung",
    "danh sach",
    "dich vu nao",
    "gom cac",
    "gom nhung",
    "goi nao",
    "goi tren",
    "hien co",
    "liet ke",
    "moi goi",
    "moi san pham",
    "nhung goi",
    "nhung dich vu",
    "nhung san pham",
    "nhung goi tren",
    "nhung san pham tren",
    "san pham dich vu nao",
    "san pham nao",
    "san pham tren",
    "tat ca san pham",
    "tung goi",
    "tung san pham",
)

TYPE_ONLY_CATALOG_MARKERS = (
    "cac loai",
    "cac nhom",
    "loai",
    "nhom",
)

PRODUCT_TYPE_QUERY_MARKERS = {
    "insurance": ("bao hiem", "fwd"),
    "card": ("cac the", "loai the", "nhung the", "the ghi no", "the tin dung"),
    "loan": ("khoan vay", "vay"),
    "saving": ("tien gui", "tiet kiem"),
    "transfer": ("chuyen khoan", "chuyen tien", "chuyen va nhan tien", "kieu hoi", "nhan tien"),
    "digital_banking": ("digibank", "ngan hang so", "sms banking"),
    "account": ("tai khoan",),
    "investment": ("chung khoan", "dau tu", "quy"),
}

CONTEXTUAL_FOLLOW_UP_MARKERS = (
    "ben tren",
    "cai do",
    "cai nay",
    "cai tren",
    "cua no",
    "dich vu do",
    "dich vu nay",
    "goi tren",
    "goi do",
    "goi nay",
    "mo the do",
    "nhom do",
    "nhom nay",
    "san pham do",
    "san pham nay",
    "san pham tren",
    "the do",
    "the nay",
    "ve no",
    "vua noi",
)

FALLBACK_SUBJECT_PATTERNS = (
    re.compile(
        r"(?:\bthe\s+)?(Vietcombank\s+.+?)(?=\s+la\b|\s+cho\s+toi\b|[,.?!\n]|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(FWD\s+.+?)(?=\s+la\b|\s+cho\s+toi\b|[,.?!\n]|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(VCB\s+.+?)(?=\s+la\b|\s+cho\s+toi\b|[,.?!\n]|$)",
        flags=re.IGNORECASE,
    ),
)

DEFAULT_DATA_ROOT = Path(os.getenv("RAG_DATA_ROOT") or Path(__file__).resolve().parents[4] / "data")
CATALOG_CHUNK_PATH = DEFAULT_DATA_ROOT / "chunks" / "vietcombank_product_catalogs_chunks.jsonl"
EVIDENCE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
EVIDENCE_STOPWORDS = {
    "anh",
    "bao",
    "cac",
    "cho",
    "co",
    "cua",
    "duoc",
    "gi",
    "hay",
    "hoi",
    "khong",
    "la",
    "lam",
    "minh",
    "mot",
    "nao",
    "neu",
    "nhu",
    "o",
    "phai",
    "qua",
    "quy",
    "sao",
    "su",
    "tai",
    "the",
    "thi",
    "toi",
    "trong",
    "va",
    "ve",
}

CONDITION_FIELD_SECTION_HEADINGS = (
    "Điều kiện khách hàng",
    "Đối tượng khách hàng",
    "Điều kiện mở thẻ",
    "Điều kiện phát hành thẻ",
    "Điều kiện sử dụng",
    "Điều kiện tham gia",
    "Đối tượng tham gia",
    "Đối tượng sử dụng",
    "Người được bảo hiểm",
    "Bên mua bảo hiểm",
)
CONDITION_ITEM_START_PATTERN = re.compile(
    r"\s+(?=(?:"
    r"Công dân|Khách hàng|HĐLĐ|Hợp đồng lao động|Có nhu cầu|Có thu nhập|Có tài sản|"
    r"Người nước ngoài|Người được bảo hiểm|Bên mua bảo hiểm|Cá nhân|Tổ chức|"
    r"Độ tuổi|Tuổi|Hội viên|Sinh viên|Công viên chức|Đáp ứng"
    r")\b)"
)
FIELD_SECTION_BOUNDARY_PATTERN = (
    r"\s+\[(?:Section|Tab|FAQ|Linked Resource|Product|Highlights|QueryComposition|GraphRAG)\]\s+"
)


@dataclass(frozen=True)
class PreparedQuery:
    rewrite_result: QueryRewriteResult
    graph_result: GraphRetrievalResult | None
    retrieval_query: str


@dataclass(frozen=True)
class RetrievalExecution:
    chunks: list[RetrievedChunk]
    route: str
    expanded_product_count: int = 0
    expanded_chunk_count: int = 0


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.query_rewriter = QueryRewriter(settings)
        self.query_decomposer = QueryDecomposer(settings)
        self.query_planner = QueryPlanner(settings)
        self.exchange_rates = ExchangeRateService(
            xml_url=settings.vietcombank_exchange_rate_xml_url,
            user_agent=settings.crawler_user_agent,
        )
        self.graph_retriever = ProductGraphRetriever(
            Path(settings.rag_data_root) if settings.rag_data_root else None
        )
        self.retriever = HybridRetriever(settings)
        self.reranker = Reranker(settings)
        self.llm = LLMClient(settings)

    async def close(self) -> None:
        await asyncio.gather(
            self.retriever.close(),
            self.reranker.close(),
            return_exceptions=True,
        )

    async def answer(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming answer.

        This buffers :meth:`stream_events` — the single source of truth for the
        pipeline — so the streaming and non-streaming paths can never diverge.
        """

        answer_parts: list[str] = []
        sources: list[SourceCitation] = []
        metadata_event: dict[str, Any] = {}
        async for event in self.stream_events(request):
            event_type = event.get("type")
            if event_type == "token":
                answer_parts.append(str(event.get("content") or ""))
            elif event_type == "sources":
                sources = [
                    SourceCitation.model_validate(source)
                    for source in event.get("sources", [])
                ]
            elif event_type == "metadata":
                metadata_event = event

        return ChatResponse(
            answer="".join(answer_parts),
            session_id=request.session_id,
            trace_id=str(metadata_event.get("trace_id") or ""),
            sources=sources,
            refusal=bool(metadata_event.get("refusal")),
            latency_ms=int(metadata_event.get("latency_ms") or 0),
            metadata=dict(metadata_event.get("metadata") or {}),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        async for event in self.stream_events(request):
            if event.get("type") == "token":
                yield str(event.get("content") or "")

    async def stream_events(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        trace_id = str(uuid4())
        question = self._latest_user_message(request)

        guardrail = inspect_query(question)
        if not guardrail.allowed:
            yield {
                "type": "token",
                "content": guardrail.safe_response or "Tôi không thể xử lý yêu cầu này.",
            }
            yield {
                "type": "sources",
                "sources": [],
            }
            yield {
                "type": "metadata",
                "trace_id": trace_id,
                "refusal": True,
                "latency_ms": self._elapsed_ms(started),
                "metadata": {"guardrail_reason": guardrail.reason},
            }
            return

        exchange_rate_answer = await self.exchange_rates.answer_query(question)
        if exchange_rate_answer is not None:
            yield {
                "type": "token",
                "content": exchange_rate_answer.answer,
            }
            yield {
                "type": "sources",
                "sources": [source.model_dump() for source in exchange_rate_answer.sources],
            }
            yield {
                "type": "metadata",
                "trace_id": trace_id,
                "refusal": False,
                "latency_ms": self._elapsed_ms(started),
                "metadata": exchange_rate_answer.metadata,
            }
            return

        history_messages = request.messages[:-1]
        prepared_query = await self._prepare_query(question, history_messages)
        rewrite_result = prepared_query.rewrite_result

        if rewrite_result.needs_clarification:
            yield {
                "type": "token",
                "content": rewrite_result.clarification_question
                or "Bạn vui lòng nêu rõ sản phẩm, nhóm sản phẩm hoặc dịch vụ cần tra cứu.",
            }
            yield {
                "type": "sources",
                "sources": [],
            }
            yield {
                "type": "metadata",
                "trace_id": trace_id,
                "refusal": False,
                "latency_ms": self._elapsed_ms(started),
                "metadata": {
                    "retrieved_count": 0,
                    "reranked_count": 0,
                    "retrieval_route": rewrite_result.route,
                    "llm_model": self.settings.llm_model,
                    "embedding_model": self.settings.embedding_model,
                    "data_collection": self.settings.qdrant_collection,
                    "retrieval_query": prepared_query.retrieval_query,
                    "query_was_resolved": prepared_query.retrieval_query != question,
                    "clarification_required": True,
                    **_query_rewrite_metadata(rewrite_result),
                },
            }
            return

        graph_result = prepared_query.graph_result
        if graph_result is None:
            raise RuntimeError("prepared query did not include graph retrieval result")
        retrieval_query = prepared_query.retrieval_query
        data_root = _pipeline_data_root(self.settings, self.graph_retriever)
        exact_faq_chunks = _exact_faq_chunks_for_query(
            retrieval_query,
            data_root=data_root,
        )

        force_exact_faq = bool(graph_result.clarification and exact_faq_chunks)

        if graph_result.clarification and not force_exact_faq:
            yield {
                "type": "token",
                "content": _clarification_answer(
                    graph_result.clarification,
                    graph_result.clarification_options,
                ),
            }
            yield {
                "type": "sources",
                "sources": [],
            }
            yield {
                "type": "metadata",
                "trace_id": trace_id,
                "refusal": False,
                "latency_ms": self._elapsed_ms(started),
                "metadata": {
                    "retrieved_count": 0,
                    "reranked_count": 0,
                    "retrieval_route": graph_result.route,
                    "llm_model": self.settings.llm_model,
                    "embedding_model": self.settings.embedding_model,
                    "data_collection": self.settings.qdrant_collection,
                    "retrieval_query": retrieval_query,
                    "query_was_resolved": retrieval_query != question,
                    "clarification_required": True,
                    **_query_rewrite_metadata(rewrite_result),
                    "clarification_options": _subject_options_metadata(
                        graph_result.clarification_options
                    ),
                },
            }
            return

        type_only_answer = (
            None if force_exact_faq else _type_only_catalog_answer(question, graph_result.chunks)
        )
        if type_only_answer:
            citation_chunks = _type_only_catalog_chunks(graph_result.chunks)
            citation_query = _type_only_catalog_citation_query(citation_chunks)
            yield {
                "type": "token",
                "content": type_only_answer,
            }
            yield {
                "type": "sources",
                "sources": [
                    source.model_dump()
                    for source in build_citations(citation_chunks, query=citation_query)
                ],
            }
            yield {
                "type": "metadata",
                "trace_id": trace_id,
                "refusal": False,
                "latency_ms": self._elapsed_ms(started),
                "metadata": {
                    "retrieved_count": len(citation_chunks),
                    "reranked_count": len(citation_chunks),
                    "retrieval_route": f"{graph_result.route}:type_only_catalog",
                    "llm_model": self.settings.llm_model,
                    "embedding_model": self.settings.embedding_model,
                    "data_collection": self.settings.qdrant_collection,
                    "retrieval_query": retrieval_query,
                    "query_was_resolved": retrieval_query != question,
                    **_query_rewrite_metadata(rewrite_result),
                },
            }
            return

        retrieval_prefetch_task = (
            None
            if force_exact_faq
            else asyncio.create_task(self._retrieve(retrieval_query, top_k=12))
        )
        try:
            query_plan = (
                _exact_faq_query_plan(exact_faq_chunks)
                if force_exact_faq
                else await self.query_planner.plan(
                    question=retrieval_query,
                    history=history_messages,
                    graph_result=graph_result,
                    graph_retriever=self.graph_retriever,
                )
            )
            if query_plan.needs_clarification:
                if exact_faq_chunks:
                    query_plan = _exact_faq_query_plan(exact_faq_chunks)
                else:
                    for event in self._planner_clarification_events(
                        query_plan=query_plan,
                        rewrite_result=rewrite_result,
                        retrieval_query=retrieval_query,
                        question=question,
                        trace_id=trace_id,
                        started=started,
                    ):
                        yield event
                    return
            elif exact_faq_chunks and _catalog_retrieval_filter(retrieval_query) is not None:
                query_plan = _exact_faq_query_plan(exact_faq_chunks)
            decomposition = await self.query_decomposer.decompose(
                question=retrieval_query,
                history=history_messages,
                graph_result=graph_result,
                query_plan=query_plan,
                graph_retriever=self.graph_retriever,
            )
            if decomposition.applied:
                await _cancel_retrieval_prefetch(retrieval_prefetch_task)
                retrieval_prefetch_task = None
                retrieval = await self._retrieve_for_decomposition(decomposition, query_plan)
            else:
                retrieval = await self._retrieve_for_plan(
                    retrieval_query,
                    graph_result,
                    query_plan=query_plan,
                    retrieve_task=retrieval_prefetch_task,
                )
                retrieval_prefetch_task = None
        finally:
            await _cancel_retrieval_prefetch(retrieval_prefetch_task)
        if query_plan.route == "planner_exact_faq":
            retrieval = RetrievalExecution(
                chunks=_merge_retrieved_chunks(exact_faq_chunks, retrieval.chunks),
                route=f"exact_faq+{retrieval.route}",
                expanded_product_count=retrieval.expanded_product_count,
                expanded_chunk_count=retrieval.expanded_chunk_count,
            )
        context_top_k = _context_top_k_for_plan(query_plan, decomposition)
        reranked = await self.reranker.rerank(
            retrieval_query,
            retrieval.chunks,
            top_k=context_top_k,
        )
        if query_plan.route == "planner_exact_faq":
            reranked = _merge_retrieved_chunks(exact_faq_chunks, reranked)
        reranked = _expand_exact_faq_context(
            retrieval_query,
            reranked,
            data_root=data_root,
        )
        if _should_refuse_out_of_scope(question, retrieval_query, reranked, query_plan):
            yield {
                "type": "token",
                "content": _out_of_scope_answer(),
            }
            yield {
                "type": "sources",
                "sources": [],
            }
            yield {
                "type": "metadata",
                "trace_id": trace_id,
                "refusal": True,
                "latency_ms": self._elapsed_ms(started),
                "metadata": {
                    "guardrail_reason": "out_of_scope",
                    "retrieved_count": len(retrieval.chunks),
                    "reranked_count": len(reranked),
                    "retrieval_route": retrieval.route,
                    "llm_model": self.settings.llm_model,
                    "embedding_model": self.settings.embedding_model,
                    "data_collection": self.settings.qdrant_collection,
                    "retrieval_query": retrieval_query,
                    "query_was_resolved": retrieval_query != question,
                    **_query_plan_metadata(query_plan, retrieval),
                    **_query_decomposition_metadata(decomposition),
                    **_query_rewrite_metadata(rewrite_result),
                },
            }
            return
        history = self._format_history(request)
        answer_parts: list[str] = []
        answer_question = _answer_question_for_decomposition(
            _answer_question_for_plan(
                question,
                retrieval_query,
                rewrite_result,
                query_plan,
            ),
            decomposition,
        )

        deterministic_answer = _requested_field_answer_from_chunks(query_plan, reranked)
        generation_failed = False
        used_fallback = False
        if deterministic_answer is not None:
            answer_parts.append(deterministic_answer)
            yield {"type": "token", "content": deterministic_answer}
        else:
            try:
                async for token in self.llm.stream_answer(
                    question=answer_question,
                    history=history,
                    chunks=reranked,
                ):
                    answer_parts.append(token)
                    yield {"type": "token", "content": token}
            except Exception as exc:  # provider/network boundary — must never surface as HTTP 500
                # A generation failure (LLM quota/rate-limit/timeout/outage) degrades
                # to a graceful message instead of crashing the request. If tokens were
                # already streamed, keep the partial answer rather than discarding it.
                generation_failed = True
                logger.warning("generation_failed", trace_id=trace_id, error=str(exc))
                if not answer_parts:
                    used_fallback = True
                    fallback = _generation_fallback_answer()
                    answer_parts.append(fallback)
                    yield {"type": "token", "content": fallback}

        citations = (
            []
            if used_fallback
            else build_citations(
                reranked,
                query=""
                if decomposition.applied
                else _citation_query_for_plan(retrieval_query, query_plan),
                answer="".join(answer_parts),
                include_grouped_catalog_items=True,
            )
        )

        yield {
            "type": "sources",
            "sources": [source.model_dump() for source in citations],
        }
        yield {
            "type": "metadata",
            "trace_id": trace_id,
            "refusal": False,
            "latency_ms": self._elapsed_ms(started),
            "metadata": {
                "retrieved_count": len(retrieval.chunks),
                "reranked_count": len(reranked),
                "retrieval_route": retrieval.route,
                "llm_model": self.settings.llm_model,
                "embedding_model": self.settings.embedding_model,
                "data_collection": self.settings.qdrant_collection,
                "retrieval_query": retrieval_query,
                "query_was_resolved": retrieval_query != question,
                "generation_failed": generation_failed,
                **_query_plan_metadata(query_plan, retrieval),
                **_query_decomposition_metadata(decomposition),
                **_query_rewrite_metadata(rewrite_result),
            },
        }

    async def _prepare_query(
        self,
        question: str,
        history: list[ChatMessage],
    ) -> PreparedQuery:
        rewrite_result = await self.query_rewriter.rewrite(
            question=question,
            history=history,
            graph_retriever=self.graph_retriever,
        )
        if rewrite_result.needs_clarification:
            return PreparedQuery(
                rewrite_result=rewrite_result,
                graph_result=None,
                retrieval_query=rewrite_result.rewritten_query or question,
            )

        effective_question = rewrite_result.rewritten_query or question
        graph_result = await self._retrieve_graph(
            effective_question,
            history=history,
            top_k=12,
        )
        retrieval_query = _retrieval_query_from_graph_result(
            effective_question,
            history,
            graph_result.resolved_query,
        )
        return PreparedQuery(
            rewrite_result=rewrite_result,
            graph_result=graph_result,
            retrieval_query=retrieval_query,
        )

    async def _retrieve_graph(
        self,
        query: str,
        *,
        history: list[ChatMessage],
        top_k: int,
    ) -> GraphRetrievalResult:
        return await asyncio.to_thread(
            self.graph_retriever.retrieve,
            query,
            history=history,
            top_k=top_k,
        )

    async def _retrieve_for_plan(
        self,
        question: str,
        graph_result: GraphRetrievalResult,
        query_plan: QueryPlan,
        *,
        retrieve_task: asyncio.Task[tuple[list[RetrievedChunk], str]] | None = None,
    ) -> RetrievalExecution:
        retrieve_task = retrieve_task or asyncio.create_task(self._retrieve(question, top_k=12))
        graph_chunks = list(graph_result.chunks)
        graph_route = graph_result.route
        expanded_chunks: list[RetrievedChunk] = []

        try:
            if query_plan.expands_product_details:
                if not _has_catalog_chunks(graph_chunks):
                    seed_query = _catalog_seed_query(query_plan.product_type)
                    if seed_query:
                        seeded_graph_result = await self._retrieve_graph(
                            seed_query,
                            history=[],
                            top_k=max(12, query_plan.max_subjects),
                        )
                        if seeded_graph_result.chunks:
                            graph_chunks = _merge_retrieved_chunks(
                                graph_chunks,
                                seeded_graph_result.chunks,
                            )
                            graph_route = f"{graph_route}+planner_catalog_seed:{seeded_graph_result.route}"

                if query_plan.compose_product_dossiers:
                    expanded_chunks = await asyncio.to_thread(
                        self.graph_retriever.product_dossier_chunks_for_catalog_chunks,
                        graph_chunks,
                        query=question,
                        max_products=query_plan.max_subjects,
                        requested_field=query_plan.requested_field,
                        max_chars_per_product=query_plan.max_chars_per_subject,
                    )
                else:
                    expanded_chunks = await asyncio.to_thread(
                        self.graph_retriever.product_detail_chunks_for_catalog_chunks,
                        graph_chunks,
                        query=question,
                        max_products=query_plan.max_subjects,
                        top_k_per_product=query_plan.detail_chunks_per_subject,
                        requested_field=query_plan.requested_field,
                    )
                graph_chunks = _merge_retrieved_chunks(graph_chunks, expanded_chunks)
        except Exception:
            await _cancel_retrieval_prefetch(retrieve_task)
            raise

        retrieved, retrieval_route = await retrieve_task

        if graph_chunks:
            retrieved = _merge_retrieved_chunks(graph_chunks, retrieved)
            retrieval_route = f"{graph_route}+{retrieval_route}"

        if query_plan.expands_product_details and expanded_chunks:
            retrieval_route = f"{query_plan.route}+{retrieval_route}"

        return RetrievalExecution(
            chunks=retrieved,
            route=retrieval_route,
            expanded_product_count=_expanded_product_count(expanded_chunks),
            expanded_chunk_count=_expanded_source_chunk_count(expanded_chunks),
        )

    async def _retrieve_for_decomposition(
        self,
        decomposition: QueryDecompositionResult,
        query_plan: QueryPlan,
    ) -> RetrievalExecution:
        per_subquery_top_k = max(3, min(5, query_plan.context_top_k))

        async def retrieve_one(index: int, subquery: str) -> RetrievalExecution:
            subquery_graph_result = await self._retrieve_graph(
                subquery,
                history=[],
                top_k=12,
            )
            retrieval = await self._retrieve_for_plan(subquery, subquery_graph_result, query_plan)
            reranked = await self.reranker.rerank(
                subquery,
                retrieval.chunks,
                top_k=per_subquery_top_k,
            )
            return RetrievalExecution(
                chunks=[
                    _tag_chunk_for_subquery(chunk, subquery=subquery, index=index)
                    for chunk in reranked
                ],
                route=retrieval.route,
                expanded_product_count=retrieval.expanded_product_count,
                expanded_chunk_count=retrieval.expanded_chunk_count,
            )

        subquery_results = await asyncio.gather(
            *(
                retrieve_one(index, subquery)
                for index, subquery in enumerate(decomposition.subqueries, start=1)
            )
        )
        chunks = [
            chunk
            for result in subquery_results
            for chunk in result.chunks
        ]
        route = "query_decomposition[" + ",".join(result.route for result in subquery_results) + "]"
        return RetrievalExecution(
            chunks=chunks,
            route=route,
            expanded_product_count=sum(
                result.expanded_product_count for result in subquery_results
            ),
            expanded_chunk_count=sum(
                result.expanded_chunk_count for result in subquery_results
            ),
        )

    async def _retrieve(self, question: str, *, top_k: int) -> tuple[list[RetrievedChunk], str]:
        catalog_filter = _catalog_retrieval_filter(question)
        if catalog_filter is not None:
            product_type = _catalog_product_type_from_filter(catalog_filter)
            preferred_category_keys = _preferred_catalog_category_keys(question, product_type)
            retrieved = await self.retriever.retrieve(question, top_k=top_k, filters=catalog_filter)
            if retrieved:
                catalog_hits = _filter_catalog_chunks(
                    retrieved,
                    product_type=product_type,
                    preferred_category_keys=preferred_category_keys,
                )
                if preferred_category_keys:
                    catalog_hits = await self._expand_preferred_catalog_hits(
                        question=question,
                        catalog_filter=catalog_filter,
                        product_type=product_type,
                        preferred_category_keys=preferred_category_keys,
                        catalog_hits=catalog_hits,
                        top_k=top_k,
                    )
                return catalog_hits[:top_k] if catalog_hits else retrieved, "product_catalog"

            fallback = await self.retriever.retrieve(question, top_k=max(top_k, 24))
            catalog_hits = _filter_catalog_chunks(
                fallback,
                product_type=product_type,
                preferred_category_keys=preferred_category_keys,
            )
            if catalog_hits:
                if preferred_category_keys:
                    catalog_hits = await self._expand_preferred_catalog_hits(
                        question=question,
                        catalog_filter=catalog_filter,
                        product_type=product_type,
                        preferred_category_keys=preferred_category_keys,
                        catalog_hits=catalog_hits,
                        top_k=top_k,
                    )
                return catalog_hits[:top_k], "product_catalog"
            return fallback[:top_k], "default"

        return await self.retriever.retrieve(question, top_k=top_k), "default"

    async def _expand_preferred_catalog_hits(
        self,
        *,
        question: str,
        catalog_filter: dict[str, Any],
        product_type: str | None,
        preferred_category_keys: list[str],
        catalog_hits: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        expanded = list(catalog_hits)
        seen_chunk_ids = {chunk.chunk_id for chunk in expanded}
        missing_category_keys = _missing_catalog_category_keys(expanded, preferred_category_keys)
        if not missing_category_keys:
            return _filter_catalog_chunks(
                expanded,
                product_type=product_type,
                preferred_category_keys=preferred_category_keys,
            )

        catalog_pool = await self.retriever.scroll_by_filter(filters=catalog_filter)
        for chunk in _catalog_chunks_matching_category_keys(
            catalog_pool,
            product_type=product_type,
            preferred_category_keys=missing_category_keys,
        ):
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            expanded.append(chunk)

        missing_category_keys = _missing_catalog_category_keys(expanded, preferred_category_keys)

        category_hit_groups = await asyncio.gather(
            *(
                self.retriever.retrieve(
                    f"{question} {category_key}",
                    top_k=max(top_k, 12),
                    filters=catalog_filter,
                )
                for category_key in missing_category_keys
            )
        )
        for category_key, category_hits in zip(
            missing_category_keys,
            category_hit_groups,
            strict=False,
        ):
            for chunk in _catalog_chunks_matching_category_keys(
                category_hits,
                product_type=product_type,
                preferred_category_keys=[category_key],
            ):
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                expanded.append(chunk)

        return _filter_catalog_chunks(
            expanded,
            product_type=product_type,
            preferred_category_keys=preferred_category_keys,
        )

    def _planner_clarification_events(
        self,
        *,
        query_plan: QueryPlan,
        rewrite_result: QueryRewriteResult,
        retrieval_query: str,
        question: str,
        trace_id: str,
        started: float,
    ) -> list[dict[str, Any]]:
        retrieval = RetrievalExecution(chunks=[], route=query_plan.route)
        return [
            {
                "type": "token",
                "content": query_plan.clarification_question
                or "Bạn vui lòng nêu rõ sản phẩm, nhóm sản phẩm hoặc dịch vụ cần tra cứu.",
            },
            {
                "type": "sources",
                "sources": [],
            },
            {
                "type": "metadata",
                "trace_id": trace_id,
                "refusal": False,
                "latency_ms": self._elapsed_ms(started),
                "metadata": {
                    "retrieved_count": 0,
                    "reranked_count": 0,
                    "retrieval_route": query_plan.route,
                    "llm_model": self.settings.llm_model,
                    "embedding_model": self.settings.embedding_model,
                    "data_collection": self.settings.qdrant_collection,
                    "retrieval_query": retrieval_query,
                    "query_was_resolved": retrieval_query != question,
                    "clarification_required": True,
                    **_query_plan_metadata(query_plan, retrieval),
                    **_query_rewrite_metadata(rewrite_result),
                    "clarification_options": _subject_options_metadata(
                        query_plan.clarification_options
                    ),
                },
            },
        ]

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


async def _cancel_retrieval_prefetch(
    task: asyncio.Task[tuple[list[RetrievedChunk], str]] | None,
) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return
    except Exception:
        return


def _retrieval_query_from_graph_result(
    question: str,
    history: list[ChatMessage],
    graph_resolved_query: str,
) -> str:
    if graph_resolved_query and graph_resolved_query != question:
        return graph_resolved_query
    return _resolve_retrieval_query(question, history)


def _merge_retrieved_chunks(
    graph_chunks: list[RetrievedChunk],
    retrieved_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    merged: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()
    for chunk in graph_chunks + retrieved_chunks:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        merged.append(chunk)
    return merged


def _tag_chunk_for_subquery(
    chunk: RetrievedChunk,
    *,
    subquery: str,
    index: int,
) -> RetrievedChunk:
    metadata = {
        **chunk.metadata,
        "subquery": subquery,
        "subquery_index": index,
        "original_chunk_id": chunk.chunk_id,
    }
    return chunk.model_copy(
        update={
            "chunk_id": f"{chunk.chunk_id}:subquery:{index}",
            "metadata": metadata,
        }
    )


def _has_catalog_chunks(chunks: list[RetrievedChunk]) -> bool:
    return any(chunk.section == "product_catalog" for chunk in chunks)


def _catalog_seed_query(product_type: str | None) -> str | None:
    return {
        "account": "cac goi tai khoan hien co",
        "card": "cac loai the Vietcombank hien co",
        "digital_banking": "cac dich vu ngan hang so hien co",
        "insurance": "cac goi bao hiem hien co",
        "investment": "cac san pham dau tu hien co",
        "loan": "cac goi vay hien co",
        "saving": "cac goi tiet kiem hien co",
        "transfer": "cac dich vu chuyen va nhan tien hien co",
    }.get(product_type or "")


def _query_rewrite_metadata(rewrite_result: QueryRewriteResult) -> dict[str, Any]:
    return {
        "query_rewrite_route": rewrite_result.route,
        "query_rewrite_applied": rewrite_result.query_was_rewritten,
        "original_query": rewrite_result.original_query,
        "rewritten_query": rewrite_result.rewritten_query,
        "query_rewrite_confidence": rewrite_result.confidence,
        "query_rewrite_reason": rewrite_result.reason,
        "clarification_options": _subject_options_metadata(rewrite_result.clarification_options),
    }


def _query_decomposition_metadata(
    decomposition: QueryDecompositionResult,
) -> dict[str, Any]:
    return {
        "query_decomposition_applied": decomposition.applied,
        "query_decomposition_route": decomposition.route,
        "query_decomposition_reason": decomposition.reason,
        "query_decomposition_confidence": decomposition.confidence,
        "subqueries": list(decomposition.subqueries),
        "subquery_count": len(decomposition.subqueries),
    }


def _query_plan_metadata(
    query_plan: QueryPlan,
    retrieval: RetrievalExecution,
) -> dict[str, Any]:
    return {
        "retrieval_plan_intent": query_plan.intent,
        "retrieval_plan_route": query_plan.route,
        "retrieval_plan_reason": query_plan.reason,
        "retrieval_plan_confidence": query_plan.confidence,
        "retrieval_plan_product_type": query_plan.product_type,
        "retrieval_plan_requested_field": query_plan.requested_field,
        "retrieval_plan_composes_product_dossiers": query_plan.compose_product_dossiers,
        "retrieval_plan_needs_clarification": query_plan.needs_clarification,
        "retrieval_plan_clarification_options": _subject_options_metadata(
            query_plan.clarification_options
        ),
        "expanded_product_count": retrieval.expanded_product_count,
        "expanded_chunk_count": retrieval.expanded_chunk_count,
        "planned_context_top_k": query_plan.context_top_k,
    }


def _context_top_k_for_plan(
    query_plan: QueryPlan,
    decomposition: QueryDecompositionResult,
) -> int:
    if not decomposition.applied:
        return query_plan.context_top_k
    return min(30, max(query_plan.context_top_k, len(decomposition.subqueries) * 5))


def _pipeline_data_root(settings: Settings, graph_retriever: Any) -> Path:
    data_root = getattr(graph_retriever, "data_root", None)
    if isinstance(data_root, Path):
        return data_root
    if settings.rag_data_root:
        return Path(settings.rag_data_root)
    return Path(__file__).resolve().parents[4] / "data"


def _expand_exact_faq_context(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    data_root: Path,
) -> list[RetrievedChunk]:
    if not chunks:
        return chunks

    top_chunk = chunks[0]
    if top_chunk.section != "faq" or not _faq_title_matches_query(query, top_chunk):
        return chunks

    source_chunks = _faq_chunks_by_source_url(data_root).get(top_chunk.source_url)
    if not source_chunks or len(source_chunks) <= 1:
        return chunks

    best_score = top_chunk.score or 0.0
    expanded = [
        chunk.model_copy(update={"score": best_score - (index * 0.0001)})
        for index, chunk in enumerate(source_chunks)
    ]
    expanded_ids = {chunk.chunk_id for chunk in expanded}
    expanded_source = top_chunk.source_url
    remaining = [
        chunk
        for chunk in chunks
        if chunk.chunk_id not in expanded_ids and chunk.source_url != expanded_source
    ]
    return [*expanded, *remaining]


def _exact_faq_query_plan(chunks: list[RetrievedChunk]) -> QueryPlan:
    product_types = {chunk.product_type for chunk in chunks if chunk.product_type}
    return QueryPlan(
        intent="direct_answer",
        route="planner_exact_faq",
        reason="exact_faq_title_match",
        product_type=next(iter(product_types)) if len(product_types) == 1 else None,
        context_top_k=max(6, min(12, len(chunks) + 4)),
        confidence=1.0,
    )


def _exact_faq_chunks_for_query(
    query: str,
    *,
    data_root: Path,
    max_sources: int = 3,
) -> list[RetrievedChunk]:
    query_tokens = _evidence_tokens(query)
    matches: list[tuple[float, int, int, str, tuple[RetrievedChunk, ...]]] = []
    for source_url, source_chunks in _faq_chunks_by_source_url(data_root).items():
        if not source_chunks:
            continue
        title_score = _faq_title_match_score(query, source_chunks[0])
        if title_score < 0.72:
            continue
        title_length = len(_evidence_tokens(source_chunks[0].title))
        category_score = _faq_category_match_score(query_tokens, source_chunks[0])
        matches.append((title_score, category_score, title_length, source_url, source_chunks))

    if not matches:
        return []

    matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_score = matches[0][0]
    selected = [
        match
        for match in matches
        if match[0] >= max(0.72, best_score - 0.05)
    ][:max_sources]

    chunks: list[RetrievedChunk] = []
    for source_index, (_, _, _, _, source_chunks) in enumerate(selected):
        for chunk_index, chunk in enumerate(source_chunks):
            chunks.append(
                chunk.model_copy(
                    update={
                        "score": 3.0 - (source_index * 0.01) - (chunk_index * 0.0001),
                    }
                )
            )
    return chunks


def _faq_category_match_score(query_tokens: set[str], chunk: RetrievedChunk) -> int:
    category = chunk.metadata.get("category") or chunk.metadata.get("category_slug")
    if not isinstance(category, str) or not category.strip():
        return 0
    return len(query_tokens & _evidence_tokens(category))


@lru_cache(maxsize=4)
def _faq_chunks_by_source_url(data_root: Path) -> dict[str, tuple[RetrievedChunk, ...]]:
    chunk_paths = _faq_chunk_paths(data_root)
    if not chunk_paths:
        return {}

    grouped: dict[str, list[tuple[int, RetrievedChunk]]] = {}
    seen_chunk_ids: set[str] = set()
    for chunks_path in chunk_paths:
        with chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("section") != "faq":
                    continue

                source_url = str(payload.get("source_url") or "")
                chunk_id = str(payload.get("chunk_id") or "")
                if not source_url or chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunk_index = int(payload.get("chunk_index") or 0)
                metadata = dict(payload.get("metadata") or {})
                metadata.setdefault("chunk_index", chunk_index)
                grouped.setdefault(source_url, []).append(
                    (
                        chunk_index,
                        RetrievedChunk(
                            chunk_id=chunk_id,
                            document_id=str(payload.get("document_id") or ""),
                            title=str(payload.get("title") or "Vietcombank FAQ"),
                            source_url=source_url,
                            section=payload.get("section"),
                            product_type=payload.get("product_type"),
                            text=str(payload.get("text") or ""),
                            score=0.0,
                            metadata=metadata,
                        ),
                    )
                )

    return {
        source_url: tuple(chunk for _, chunk in sorted(items, key=lambda item: item[0]))
        for source_url, items in grouped.items()
    }


def _faq_chunk_paths(data_root: Path) -> tuple[Path, ...]:
    chunks_dir = data_root / "chunks"
    candidates = (
        chunks_dir / "vietcombank_faq_chunks.jsonl",
        chunks_dir / "vietcombank_chunks.jsonl",
    )
    return tuple(path for path in candidates if path.exists())


def _faq_title_matches_query(query: str, chunk: RetrievedChunk) -> bool:
    return _faq_title_match_score(query, chunk) >= 0.6


def _faq_title_match_score(query: str, chunk: RetrievedChunk) -> float:
    query_tokens = _evidence_tokens(query)
    title_tokens = _evidence_tokens(chunk.title)
    if not query_tokens or not title_tokens:
        return 0.0

    title_overlap = query_tokens & title_tokens
    return len(title_overlap) / len(title_tokens)


def _should_refuse_out_of_scope(
    original_question: str,
    retrieval_query: str,
    chunks: list[RetrievedChunk],
    query_plan: QueryPlan,
) -> bool:
    query = f"{original_question} {retrieval_query}"
    if _has_vietcombank_evidence(query, chunks):
        return False
    if _planner_marked_out_of_scope(query_plan):
        return True
    return not is_likely_supported_domain(query)


def _planner_marked_out_of_scope(query_plan: QueryPlan) -> bool:
    if query_plan.product_type is not None:
        return False
    normalized_reason = _normalize_query_key(query_plan.reason)
    return any(
        marker in normalized_reason
        for marker in (
            "khong lien quan",
            "ngoai pham vi",
            "not related",
            "out of scope",
            "outside scope",
            "unrelated",
        )
    )


def _has_vietcombank_evidence(query: str, chunks: list[RetrievedChunk]) -> bool:
    query_tokens = _evidence_tokens(query)
    if not query_tokens:
        return False

    return any(
        _is_vietcombank_source(chunk.source_url)
        and _chunk_matches_query(query_tokens, chunk)
        for chunk in chunks[:8]
    )


def _chunk_matches_query(query_tokens: set[str], chunk: RetrievedChunk) -> bool:
    title_tokens = _evidence_tokens(chunk.title)
    text_tokens = _evidence_tokens(chunk.text[:1200])
    title_overlap = query_tokens & title_tokens
    total_overlap = title_overlap | (query_tokens & text_tokens)
    if not title_overlap:
        return False

    query_coverage = len(total_overlap) / len(query_tokens)
    title_coverage = len(title_overlap) / len(title_tokens) if title_tokens else 0.0

    if len(query_tokens) <= 2 and query_coverage >= 0.5:
        return True
    return (
        query_coverage >= 0.45 and len(title_overlap) >= 2
        or (len(title_overlap) >= 2 and title_coverage >= 0.45)
    )


def _is_vietcombank_source(source_url: str) -> bool:
    normalized = source_url.casefold()
    return "vietcombank.com.vn" in normalized


def _evidence_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(character for character in normalized if not unicodedata.combining(character))
    ascii_text = ascii_text.replace("đ", "d")
    return {
        token
        for token in EVIDENCE_TOKEN_PATTERN.findall(ascii_text)
        if len(token) > 1 and token not in EVIDENCE_STOPWORDS
    }


def _out_of_scope_answer() -> str:
    return (
        "Tôi chỉ hỗ trợ tra cứu thông tin công khai liên quan đến Vietcombank "
        "trong phạm vi dữ liệu đã được index."
    )


def _generation_fallback_answer() -> str:
    return (
        "Xin lỗi, hệ thống đang tạm thời quá tải hoặc gặp sự cố khi tạo câu trả lời. "
        "Bạn vui lòng thử lại sau ít phút."
    )


def _expanded_product_count(chunks: list[RetrievedChunk]) -> int:
    return sum(
        1
        for chunk in chunks
        if (
            chunk.chunk_id.startswith("graph:product:")
            and chunk.metadata.get("retrieval_source") == "graph"
        )
        or (
            chunk.chunk_id.startswith("composed:product:")
            and chunk.metadata.get("retrieval_source") == "query_composition"
        )
    )


def _expanded_source_chunk_count(chunks: list[RetrievedChunk]) -> int:
    count = 0
    for chunk in chunks:
        source_chunk_ids = chunk.metadata.get("source_chunk_ids")
        if isinstance(source_chunk_ids, list):
            count += len(source_chunk_ids)
        else:
            count += 1
    return count


def _citation_query_for_plan(retrieval_query: str, query_plan: QueryPlan) -> str:
    if query_plan.expands_product_details:
        return ""
    return retrieval_query


def _subject_options_metadata(options: tuple[GraphSubjectOption, ...]) -> list[dict[str, str | None]]:
    return [
        {
            "title": option.title,
            "type": option.subject_type,
            "url": option.url,
            "product_type": option.product_type,
            "category_title": option.category_title,
            "parent_title": option.parent_title,
        }
        for option in options
    ]


def _clarification_answer(
    message: str,
    options: tuple[GraphSubjectOption, ...],
) -> str:
    if not options:
        return message
    lines = [message, "Bạn có thể chọn một trong các mục liên quan sau:"]
    for index, option in enumerate(options, start=1):
        label = "nhóm" if option.subject_type == "category" else "sản phẩm"
        lines.append(f"{index}. {option.title} ({label})")
    return "\n".join(lines)


def _requested_field_answer_from_chunks(
    query_plan: QueryPlan,
    chunks: list[RetrievedChunk],
) -> str | None:
    if (
        query_plan.intent != "exhaustive_product_details"
        or query_plan.requested_field != "condition"
        or not query_plan.compose_product_dossiers
    ):
        return None

    dossiers = [
        chunk
        for chunk in chunks
        if chunk.section == "product_detail"
        and chunk.metadata.get("retrieval_source") == "query_composition"
    ]
    if len(dossiers) < 2:
        return None

    product_conditions: list[tuple[RetrievedChunk, list[str]]] = []
    found_condition_count = 0
    seen_titles: set[str] = set()
    for chunk in dossiers:
        normalized_title = _normalize_query_key(chunk.title)
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        items = _condition_items_from_text(chunk.text)
        if items:
            found_condition_count += 1
        product_conditions.append((chunk, items))

    if found_condition_count == 0:
        return None

    lines = [_condition_answer_intro(product_conditions)]
    for index, (chunk, items) in enumerate(product_conditions, start=1):
        lines.append(f"{index}. {chunk.title}")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- Chưa thấy điều kiện/đối tượng cụ thể trong nguồn hiện có cho sản phẩm này.")
    return "\n".join(lines)


def _condition_answer_intro(
    product_conditions: list[tuple[RetrievedChunk, list[str]]],
) -> str:
    first_chunk = product_conditions[0][0]
    category = str(first_chunk.metadata.get("category_title") or "").strip()
    if first_chunk.product_type == "loan":
        if category:
            return (
                f"Có. Vietcombank có các gói {_lower_first(category)}. "
                "Điều kiện vay theo từng gói:"
            )
        return "Có. Điều kiện vay theo từng gói:"
    return "Điều kiện/đối tượng theo từng sản phẩm/gói:"


def _condition_items_from_text(text: str) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for body in _condition_section_bodies(text):
        for item in _split_condition_items(body):
            cleaned = _clean_condition_item(item)
            if not cleaned:
                continue
            key = _normalize_query_key(cleaned)
            if key in seen:
                continue
            seen.add(key)
            items.append(cleaned)
    return items


def _condition_section_bodies(text: str) -> list[str]:
    flat_text = _flatten_field_text(text)
    bodies: list[str] = []
    for heading in CONDITION_FIELD_SECTION_HEADINGS:
        pattern = re.compile(
            rf"(?:^|\s)\[Section\]\s+{re.escape(heading)}\s+(?P<body>.*?)(?={FIELD_SECTION_BOUNDARY_PATTERN}|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        bodies.extend(match.group("body") for match in pattern.finditer(flat_text))
    return bodies


def _split_condition_items(body: str) -> list[str]:
    body = _flatten_field_text(body)
    if not body:
        return []
    candidates = CONDITION_ITEM_START_PATTERN.split(body)
    if len(candidates) == 1:
        candidates = re.split(r";\s+|\n+", body)
    return [candidate for candidate in candidates if candidate.strip()]


def _clean_condition_item(item: str) -> str:
    item = re.split(FIELD_SECTION_BOUNDARY_PATTERN, item, maxsplit=1, flags=re.IGNORECASE)[0]
    item = re.sub(r"\s+", " ", item).strip(" -:;,.")
    item = item.replace("​​​​​​​", "")
    if not item:
        return ""
    if len(item) > 420:
        item = item[:417].rstrip(" ,.;") + "..."
    return item


def _flatten_field_text(text: str) -> str:
    return " ".join(text.split())


def _lower_first(text: str) -> str:
    if not text:
        return text
    return f"{text[:1].casefold()}{text[1:]}"


def _answer_question(
    original_question: str,
    retrieval_query: str,
    rewrite_result: QueryRewriteResult,
) -> str:
    if rewrite_result.query_was_rewritten:
        return rewrite_result.rewritten_query
    if retrieval_query != original_question:
        return retrieval_query
    return original_question


def _answer_question_for_plan(
    original_question: str,
    retrieval_query: str,
    rewrite_result: QueryRewriteResult,
    query_plan: QueryPlan,
) -> str:
    question = _answer_question(original_question, retrieval_query, rewrite_result)
    if query_plan.intent != "exhaustive_product_details":
        return question

    instructions = [
        "Yêu cầu xử lý: câu hỏi cần tổng hợp nhiều sản phẩm.",
        "Hãy trả lời theo từng sản phẩm/gói dựa trên các nguồn đã truy xuất.",
        "Nêu rõ thông tin nào chưa thấy trong nguồn hiện có, không suy đoán.",
    ]
    if query_plan.compose_product_dossiers:
        instructions.append(
            "Mỗi context có nhãn [QueryComposition] là hồ sơ đã được ghép riêng cho một sản phẩm; "
            "hãy đọc từng hồ sơ đó để trả lời, không chỉ dựa vào dòng catalog summary."
        )
    if query_plan.requested_field == "condition":
        instructions.append(
            "Trọng tâm là điều kiện/đối tượng/yêu cầu sử dụng sản phẩm. "
            "Không dùng mục Hồ sơ, Quy trình, Biểu phí hoặc Hướng dẫn để thay thế điều kiện; "
            "với mỗi sản phẩm/gói có đoạn Điều kiện hoặc Đối tượng trong context, bắt buộc nêu riêng đoạn đó; "
            "không thay bằng Mức vay, Thời hạn vay, Lãi suất hoặc tóm tắt catalog."
        )

    return (
        f"{question}\n\n"
        + " ".join(instructions)
    )


def _answer_question_for_decomposition(
    question: str,
    decomposition: QueryDecompositionResult,
) -> str:
    if not decomposition.applied:
        return question

    subquery_lines = "\n".join(
        f"{index}. {subquery}"
        for index, subquery in enumerate(decomposition.subqueries, start=1)
    )
    return (
        f"{question}\n\n"
        "Câu hỏi đã được tách thành các ý tra cứu sau. "
        "Hãy trả lời đủ từng ý dựa trên context tương ứng; nếu context của một ý không có bằng chứng thì bỏ qua phần không có bằng chứng, không suy đoán.\n"
        "Khong tron context giua cac san pham/subquery; moi y chi duoc tra loi tu context co SUBQUERY tuong ung.\n"
        f"{subquery_lines}"
    )


def _type_only_catalog_answer(query: str, chunks: list[RetrievedChunk]) -> str | None:
    if not _is_type_only_catalog_query(query):
        return None

    parent_chunk = _type_only_parent_catalog_chunk(chunks)
    if parent_chunk is None:
        return None

    group_titles = _catalog_group_titles(parent_chunk)
    if len(group_titles) <= 1:
        return None

    category_title = str(parent_chunk.metadata.get("category_title") or "").strip()
    normalized_query = _normalize_query_key(query)
    if parent_chunk.product_type == "card":
        label = "loại thẻ"
    elif "loai" in normalized_query:
        label = f"loại {category_title.casefold()}".strip()
    else:
        label = f"nhóm {category_title.casefold()}".strip()

    lines = [f"Vietcombank hiện có các {label} chính gồm:"]
    lines.extend(f"{index}. {title}" for index, title in enumerate(group_titles, start=1))
    return "\n".join(lines)


def _is_type_only_catalog_query(query: str) -> bool:
    normalized = _normalize_query_key(query)
    if not any(_has_phrase(normalized, marker) for marker in TYPE_ONLY_CATALOG_MARKERS):
        return False
    return _infer_product_type_from_query(normalized) is not None


def _type_only_catalog_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    parent_chunk = _type_only_parent_catalog_chunk(chunks)
    return [parent_chunk] if parent_chunk is not None else []


def _type_only_parent_catalog_chunk(chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    for chunk in chunks:
        if chunk.section != "product_catalog":
            continue
        if chunk.metadata.get("parent_category_title"):
            continue
        if len(_catalog_group_titles(chunk)) > 1:
            return chunk
    return None


def _type_only_catalog_citation_query(chunks: list[RetrievedChunk]) -> str:
    return " ".join(title for chunk in chunks for title in _catalog_group_titles(chunk))


def _catalog_group_titles(chunk: RetrievedChunk) -> list[str]:
    items = chunk.metadata.get("items")
    if not isinstance(items, list):
        return []

    titles: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("category") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


def _catalog_retrieval_filter(query: str) -> dict[str, Any] | None:
    normalized = _normalize_query_key(query)
    product_type = _infer_product_type_from_query(normalized)
    if not _is_catalog_query(normalized, product_type=product_type):
        return None

    must: list[dict[str, Any]] = [
        {
            "key": "section",
            "match": {"value": "product_catalog"},
        }
    ]
    if product_type:
        must.append(
            {
                "key": "product_type",
                "match": {"value": product_type},
            }
        )
    return {"must": must}


def _resolve_retrieval_query(question: str, history: list[ChatMessage]) -> str:
    normalized = _normalize_query_key(question)
    if not _is_contextual_follow_up(normalized):
        return question

    subject = _latest_catalog_subject(history)
    if not subject:
        return question

    normalized_subject = _normalize_query_key(subject)
    if _has_phrase(normalized, normalized_subject):
        return question
    return f"{question} {subject}"


def _is_contextual_follow_up(normalized_query: str) -> bool:
    return any(_has_phrase(normalized_query, marker) for marker in CONTEXTUAL_FOLLOW_UP_MARKERS)


def _latest_catalog_subject(history: list[ChatMessage]) -> str | None:
    for message in reversed(history[-8:]):
        subject = _extract_catalog_subject(message.content)
        if subject:
            return subject
    return None


def _extract_catalog_subject(text: str) -> str | None:
    normalized_text = _normalize_query_key(text)
    for normalized_alias, title in _catalog_subject_aliases():
        if _has_phrase(normalized_text, normalized_alias):
            return title

    for pattern in FALLBACK_SUBJECT_PATTERNS:
        match = pattern.search(_normalize_query_key(text))
        if match:
            return _clean_fallback_subject(match.group(1))
    return None


def _clean_fallback_subject(subject: str) -> str:
    words = subject.split()
    return " ".join(words[:8]).strip()


@lru_cache(maxsize=1)
def _catalog_subject_aliases() -> tuple[tuple[str, str], ...]:
    aliases: dict[str, str] = {}
    if not CATALOG_CHUNK_PATH.exists():
        return ()

    with CATALOG_CHUNK_PATH.open("r", encoding="utf-8") as catalog_file:
        for line in catalog_file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                continue

            for title in _catalog_titles_from_metadata(metadata):
                for alias in _subject_aliases(title):
                    aliases.setdefault(alias, title)

    return tuple(
        sorted(
            aliases.items(),
            key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]),
        )
    )


def _catalog_titles_from_metadata(metadata: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for field_name in ("category_title", "parent_category_title"):
        value = metadata.get(field_name)
        if isinstance(value, str) and _is_specific_subject_title(value):
            titles.append(value.strip())

    items = metadata.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            if isinstance(title, str) and _is_specific_subject_title(title):
                titles.append(title.strip())
    return titles


def _subject_aliases(title: str) -> list[str]:
    normalized = _normalize_query_key(title)
    aliases = [normalized]
    for prefix in ("vietcombank ", "vcb ", "fwd "):
        if not normalized.startswith(prefix):
            continue
        tail = normalized.removeprefix(prefix).strip()
        if len(tail.split()) >= 2:
            aliases.append(tail)
    return aliases


def _is_specific_subject_title(title: str) -> bool:
    normalized = _normalize_query_key(title)
    return len(normalized.split()) >= 2


def _infer_product_type_from_query(normalized_query: str) -> str | None:
    padded_query = f" {normalized_query} "
    for product_type, markers in PRODUCT_TYPE_QUERY_MARKERS.items():
        for marker in markers:
            if f" {marker} " in padded_query or marker in normalized_query:
                return product_type
    return None


def _is_catalog_query(normalized_query: str, *, product_type: str | None) -> bool:
    if any(marker in normalized_query for marker in CATALOG_QUERY_MARKERS):
        return True
    return product_type is not None and _is_catalog_availability_query(normalized_query)


def _is_catalog_availability_query(normalized_query: str) -> bool:
    if any(
        _has_phrase(normalized_query, marker)
        for marker in ("co cho", "co dich vu", "co ho tro", "co cung cap")
    ):
        return True
    if not (_has_phrase(normalized_query, "co") and _has_phrase(normalized_query, "khong")):
        return False
    if _has_phrase(normalized_query, "co duoc"):
        return False
    return len(normalized_query.split()) <= 8


def _catalog_product_type_from_filter(filters: dict[str, Any]) -> str | None:
    for condition in filters.get("must", []):
        if not isinstance(condition, dict) or condition.get("key") != "product_type":
            continue
        match = condition.get("match")
        value = match.get("value") if isinstance(match, dict) else None
        if isinstance(value, str):
            return value
    return None


def _filter_catalog_chunks(
    chunks: list[RetrievedChunk],
    *,
    product_type: str | None,
    preferred_category_keys: list[str] | None = None,
) -> list[RetrievedChunk]:
    catalog_chunks = [
        chunk
        for chunk in chunks
        if chunk.section == "product_catalog"
        and (product_type is None or chunk.product_type == product_type)
    ]
    if not preferred_category_keys:
        return _prefer_parent_catalog_chunks(catalog_chunks)

    return (
        _catalog_chunks_matching_category_keys(
            catalog_chunks,
            product_type=product_type,
            preferred_category_keys=preferred_category_keys,
        )
        or catalog_chunks
    )


def _prefer_parent_catalog_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    parent_chunks = [
        chunk
        for chunk in chunks
        if not chunk.metadata.get("parent_category_title")
    ]
    return parent_chunks or chunks


def _catalog_chunks_matching_category_keys(
    chunks: list[RetrievedChunk],
    *,
    product_type: str | None,
    preferred_category_keys: list[str],
) -> list[RetrievedChunk]:
    catalog_chunks = [
        chunk
        for chunk in chunks
        if chunk.section == "product_catalog"
        and (product_type is None or chunk.product_type == product_type)
    ]
    preferred_index = {key: index for index, key in enumerate(preferred_category_keys)}
    preferred_chunks = [
        chunk
        for chunk in catalog_chunks
        if _catalog_category_key(chunk) in preferred_index
    ]
    return sorted(
        preferred_chunks,
        key=lambda chunk: (
            preferred_index[_catalog_category_key(chunk)],
            -(chunk.score or 0),
        ),
    )


def _missing_catalog_category_keys(
    chunks: list[RetrievedChunk],
    preferred_category_keys: list[str],
) -> list[str]:
    matched_category_keys = {
        _catalog_category_key(chunk)
        for chunk in chunks
        if chunk.section == "product_catalog"
    }
    return [key for key in preferred_category_keys if key not in matched_category_keys]


def _preferred_catalog_category_keys(query: str, product_type: str | None) -> list[str]:
    normalized = _normalize_query_key(query)
    if product_type == "transfer":
        return _preferred_transfer_category_keys(normalized)
    if product_type == "card":
        return _preferred_card_category_keys(normalized)
    if product_type == "insurance":
        return _preferred_insurance_category_keys(normalized)
    return []


def _preferred_transfer_category_keys(normalized_query: str) -> list[str]:
    preferred: list[str] = []
    mentions_foreign = "nuoc ngoai" in normalized_query or "quoc te" in normalized_query
    mentions_combined = _has_phrase(normalized_query, "chuyen va nhan") or (
        _has_phrase(normalized_query, "chuyen") and _has_phrase(normalized_query, "nhan")
    )
    mentions_receive = (
        _has_phrase(normalized_query, "nhan")
        or _has_phrase(normalized_query, "nhan tien")
        or "kieu hoi" in normalized_query
    )
    inbound_from_foreign = "tu nuoc ngoai" in normalized_query and (
        mentions_receive
        or _has_phrase(normalized_query, "ve")
        or "ve viet nam" in normalized_query
        or _has_phrase(normalized_query, "ve vn")
    )

    if "trong nuoc" in normalized_query:
        preferred.append("chuyen va nhan tien trong nuoc")
    if "kieu hoi" in normalized_query or (mentions_foreign and (mentions_receive or inbound_from_foreign)):
        preferred.append("nhan kieu hoi")
    if "ra nuoc ngoai" in normalized_query or (
        mentions_foreign
        and (_has_phrase(normalized_query, "chuyen") or _has_phrase(normalized_query, "chuyen tien"))
        and (mentions_combined or not inbound_from_foreign)
    ):
        preferred.append("chuyen tien ra nuoc ngoai")

    return _dedupe_preserving_order(preferred)


def _preferred_card_category_keys(normalized_query: str) -> list[str]:
    if "the thanh toan" in normalized_query or "the ghi no" in normalized_query or "debit" in normalized_query:
        return ["the thanh toan"]
    if "the tin dung" in normalized_query or "credit" in normalized_query:
        return ["the tin dung"]
    if "tra gop" in normalized_query:
        return ["tra gop"]
    return []


def _preferred_insurance_category_keys(normalized_query: str) -> list[str]:
    categories = (
        "bao hiem tiet kiem",
        "bao hiem bao ve",
        "bao hiem dau tu",
    )
    return [category for category in categories if category in normalized_query]


def _catalog_category_key(chunk: RetrievedChunk) -> str:
    category_title = chunk.metadata.get("category_title")
    if isinstance(category_title, str) and category_title.strip():
        return _normalize_query_key(category_title)
    return _normalize_query_key(chunk.title)


def _has_phrase(normalized_query: str, phrase: str) -> bool:
    return f" {phrase} " in f" {normalized_query} "


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _normalize_query_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(no_accents.casefold().split())
