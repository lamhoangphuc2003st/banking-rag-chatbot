from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage
from apps.api.app.rag.planner import QueryPlan
from apps.api.app.rag.retrieval.graph import (
    GraphRetrievalResult,
    GraphSubjectOption,
    ProductGraphRetriever,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MAX_SUBQUERIES = 8

COMPLEX_MARKERS = (
    " va ",
    " dong thoi ",
    " cung nhu ",
    " kem ",
    " hoac ",
    ",",
    ";",
    "/",
)

FIELD_MARKERS: dict[str, tuple[str, ...]] = {
    "benefits": (
        "dac diem",
        "hoan tien",
        "loi ich",
        "quyen loi",
        "tich diem",
        "uu dai",
    ),
    "condition": (
        "ai duoc",
        "dieu kien",
        "doi tuong",
        "yeu cau",
    ),
    "documents": (
        "can giay to",
        "chung tu",
        "giay to",
        "ho so",
    ),
    "fee": (
        "bieu phi",
        "chi phi",
        "mat phi",
        "phi",
    ),
    "interest_rate": ("lai suat",),
    "limit": (
        "han muc",
        "muc vay",
        "so tien vay",
    ),
    "procedure": (
        "cac buoc",
        "quy trinh",
        "thu tuc",
    ),
    "registration": (
        "cach dang ky",
        "dang ky",
        "mo the",
    ),
}

FIELD_LABELS = {
    "benefits": "Lợi ích và ưu đãi",
    "condition": "Điều kiện mở hoặc sử dụng",
    "documents": "Hồ sơ cần chuẩn bị",
    "fee": "Biểu phí",
    "interest_rate": "Lãi suất",
    "limit": "Hạn mức",
    "procedure": "Thủ tục",
    "registration": "Cách đăng ký",
}


@dataclass(frozen=True)
class QueryDecompositionResult:
    original_query: str
    subqueries: tuple[str, ...] = field(default_factory=tuple)
    route: str = "none"
    reason: str | None = None
    confidence: float | None = None

    @property
    def applied(self) -> bool:
        return len(self.subqueries) > 1


class QueryDecomposer:
    """Splits multi-part user questions into retrieval-focused subqueries.

    Facts are still retrieved from the indexed corpus. The optional LLM call only
    chooses search subqueries and is skipped when local rules can do that safely.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def decompose(
        self,
        *,
        question: str,
        history: list[ChatMessage],
        graph_result: GraphRetrievalResult,
        query_plan: QueryPlan,
        graph_retriever: ProductGraphRetriever,
    ) -> QueryDecompositionResult:
        local_result = self._local_decompose(
            question=question,
            graph_result=graph_result,
            query_plan=query_plan,
            graph_retriever=graph_retriever,
        )
        if local_result.applied or not self._should_call_llm(question, query_plan):
            return local_result

        try:
            payload = await self._call_llm_json(question=question, history=history)
        except Exception:
            return local_result

        return self._result_from_llm_payload(
            question=question,
            payload=payload,
            local_result=local_result,
        )

    def _local_decompose(
        self,
        *,
        question: str,
        graph_result: GraphRetrievalResult,
        query_plan: QueryPlan,
        graph_retriever: ProductGraphRetriever,
    ) -> QueryDecompositionResult:
        if _should_skip_decomposition(query_plan):
            return QueryDecompositionResult(
                original_query=question,
                route="skipped",
                reason="retrieval_plan_handles_scope",
            )

        normalized = _normalize_query_key(question)
        if not _has_complex_marker(normalized):
            return QueryDecompositionResult(
                original_query=question,
                route="none",
                reason="single_intent_query",
            )

        fields = _fields_from_query(normalized)
        subjects = _subjects_from_query(
            question,
            graph_result=graph_result,
            graph_retriever=graph_retriever,
        )
        if not fields or not subjects:
            return QueryDecompositionResult(
                original_query=question,
                route="none",
                reason="missing_fields_or_subjects",
            )

        if len(fields) == 1 and len(subjects) == 1:
            return QueryDecompositionResult(
                original_query=question,
                route="none",
                reason="single_field_single_subject",
            )

        subqueries = _compose_field_subject_subqueries(fields, subjects)
        if len(subqueries) <= 1:
            return QueryDecompositionResult(
                original_query=question,
                route="none",
                reason="no_valid_subqueries",
            )

        return QueryDecompositionResult(
            original_query=question,
            subqueries=tuple(subqueries),
            route="local_field_subject",
            reason="multi_field_or_multi_subject_query",
            confidence=0.88,
        )

    def _should_call_llm(self, question: str, query_plan: QueryPlan) -> bool:
        if _should_skip_decomposition(query_plan):
            return False
        if not _has_complex_marker(_normalize_query_key(question)):
            return False

        provider = self.settings.llm_provider.strip().lower()
        if provider in {"", "local", "none"}:
            return False
        return provider != "openai" or bool(
            self.settings.openai_api_key or self.settings.litellm_api_key
        )

    async def _call_llm_json(
        self,
        *,
        question: str,
        history: list[ChatMessage],
    ) -> dict[str, Any]:
        from litellm import acompletion

        compact_history = "\n".join(
            f"{message.role}: {message.content}" for message in history[-6:]
        )
        response = await acompletion(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You split complex Vietcombank retrieval queries. Return JSON only. "
                        "Do not answer the banking question or invent product facts."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_decompose_prompt(
                        question=question,
                        history=compact_history,
                    ),
                },
            ],
            temperature=0,
            api_key=self.settings.openai_api_key or self.settings.litellm_api_key,
        )
        content = str(response["choices"][0]["message"]["content"]).strip()
        return _extract_json_object(content)

    def _result_from_llm_payload(
        self,
        *,
        question: str,
        payload: dict[str, Any],
        local_result: QueryDecompositionResult,
    ) -> QueryDecompositionResult:
        raw_subqueries = payload.get("subqueries")
        if not isinstance(raw_subqueries, Sequence) or isinstance(raw_subqueries, str):
            return local_result

        subqueries = _dedupe_preserving_order(
            str(item).strip()
            for item in raw_subqueries
            if isinstance(item, str) and item.strip()
        )
        subqueries = [item for item in subqueries if _normalize_query_key(item) != _normalize_query_key(question)]
        if len(subqueries) <= 1:
            return local_result

        confidence = _coerce_confidence(payload.get("confidence"))
        if confidence is not None and confidence < 0.55:
            return local_result

        return QueryDecompositionResult(
            original_query=question,
            subqueries=tuple(subqueries[:MAX_SUBQUERIES]),
            route="llm_decompose",
            reason=str(payload.get("reason") or "llm_selected_subqueries").strip() or None,
            confidence=confidence,
        )


def _should_skip_decomposition(query_plan: QueryPlan) -> bool:
    return (
        query_plan.needs_clarification
        or query_plan.expands_product_details
        or query_plan.intent == "catalog_overview"
        or query_plan.route.startswith("planner_security_")
    )


def _fields_from_query(normalized_query: str) -> list[str]:
    fields: list[str] = []
    for field_name, markers in FIELD_MARKERS.items():
        if field_name == "registration" and "dieu kien" in normalized_query:
            continue
        if any(_has_phrase(normalized_query, marker) for marker in markers):
            fields.append(field_name)
    return fields


def _subjects_from_query(
    question: str,
    *,
    graph_result: GraphRetrievalResult,
    graph_retriever: ProductGraphRetriever,
) -> list[GraphSubjectOption]:
    options = [
        option
        for option in graph_retriever.match_subjects(question, limit=MAX_SUBQUERIES)
        if option.subject_type == "product"
    ]
    if options:
        return list(
            _sort_subject_options_by_query_order(
                question,
                _prefer_specific_subject_options(_dedupe_subject_options(options)),
            )
        )[:MAX_SUBQUERIES]

    fallback_options: list[GraphSubjectOption] = []
    for chunk in graph_result.chunks:
        if chunk.section != "product_detail":
            continue
        if not (
            chunk.chunk_id.startswith("graph:product:")
            or chunk.metadata.get("retrieval_source") == "graph"
        ):
            continue
        fallback_options.append(
            GraphSubjectOption(
                title=chunk.title,
                subject_type="product",
                url=chunk.source_url,
                product_type=chunk.product_type,
                category_title=str(chunk.metadata.get("category_title") or "") or None,
                parent_title=str(chunk.metadata.get("parent_category_title") or "") or None,
            )
        )
    return list(
        _sort_subject_options_by_query_order(
            question,
            _prefer_specific_subject_options(_dedupe_subject_options(fallback_options)),
        )
    )[:MAX_SUBQUERIES]


def _compose_field_subject_subqueries(
    fields: list[str],
    subjects: list[GraphSubjectOption],
) -> list[str]:
    subqueries: list[str] = []
    for subject in subjects:
        for field_name in fields:
            label = FIELD_LABELS.get(field_name)
            if not label:
                continue
            subqueries.append(f"{label} của {subject.title} là gì?")
            if len(subqueries) >= MAX_SUBQUERIES:
                return subqueries
    return subqueries


def _has_complex_marker(normalized_query: str) -> bool:
    padded = f" {normalized_query} "
    return any(marker in padded for marker in COMPLEX_MARKERS)


def _build_decompose_prompt(*, question: str, history: str) -> str:
    return f"""Split the current Vietnamese banking query into standalone retrieval subqueries.

Rules:
- Return a single JSON object, no markdown.
- Use this schema: {{"subqueries": ["..."], "confidence": 0.0, "reason": "short reason"}}.
- Keep subqueries concise and grounded in the product/service names from the user query or history.
- If the query has only one retrieval intent, return one subquery identical to the current query.
- Do not answer the question.

Conversation history:
{history or "none"}

Current query:
{question}
"""


def _extract_json_object(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("decomposition response must be a JSON object")
    return payload


def _dedupe_subject_options(
    options: list[GraphSubjectOption],
) -> tuple[GraphSubjectOption, ...]:
    deduped: list[GraphSubjectOption] = []
    seen: set[tuple[str, str]] = set()
    for option in options:
        key = (option.subject_type, option.url or _normalize_query_key(option.title))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return tuple(deduped)


def _prefer_specific_subject_options(
    options: tuple[GraphSubjectOption, ...],
) -> tuple[GraphSubjectOption, ...]:
    filtered: list[GraphSubjectOption] = []
    normalized_titles = [_normalize_query_key(option.title) for option in options]
    token_sets = [_distinctive_subject_tokens(option.title) for option in options]
    for index, option in enumerate(options):
        title_key = normalized_titles[index]
        tokens = token_sets[index]
        if not title_key or not tokens:
            filtered.append(option)
            continue

        is_subsumed = False
        for other_index, other_title_key in enumerate(normalized_titles):
            if index == other_index:
                continue
            other_tokens = token_sets[other_index]
            if len(other_tokens) <= len(tokens):
                continue
            if tokens < other_tokens and f" {title_key} " in f" {other_title_key} ":
                is_subsumed = True
                break
        if not is_subsumed:
            filtered.append(option)
    return tuple(filtered)


def _sort_subject_options_by_query_order(
    query: str,
    options: tuple[GraphSubjectOption, ...],
) -> tuple[GraphSubjectOption, ...]:
    normalized_query = _normalize_query_key(query)
    return tuple(
        sorted(
            options,
            key=lambda option: _subject_position(normalized_query, option.title),
        )
    )


def _subject_position(normalized_query: str, title: str) -> int:
    normalized_title = _normalize_query_key(title)
    position = normalized_query.find(normalized_title)
    return position if position >= 0 else len(normalized_query)


def _distinctive_subject_tokens(title: str) -> set[str]:
    generic_tokens = {
        "card",
        "the",
        "tin",
        "thanh",
        "toan",
        "tung",
        "vcb",
        "vietcombank",
    }
    return {
        token
        for token in TOKEN_PATTERN.findall(_normalize_query_key(title))
        if token not in generic_tokens
    }


def _dedupe_preserving_order(items: Sequence[str] | Any) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = _normalize_query_key(str(item))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(str(item))
    return deduped


def _coerce_confidence(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, number))


def _has_phrase(normalized_query: str, phrase: str) -> bool:
    return f" {phrase} " in f" {normalized_query} "


def _normalize_query_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(TOKEN_PATTERN.findall(no_accents.casefold()))
