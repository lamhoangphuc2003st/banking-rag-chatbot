from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from apps.api.app.models.chat import ChatMessage
from apps.api.app.rag.retrieval.graph import (
    GraphRetrievalResult,
    GraphSubjectOption,
    ProductGraphRetriever,
)
from apps.api.app.rag.security_intent import classify_security_intent
from packages.shared.schemas import RetrievedChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

PlanIntent = Literal[
    "direct_answer",
    "catalog_overview",
    "exhaustive_product_details",
]
RequestedField = Literal[
    "benefits",
    "condition",
    "documents",
    "fee",
    "interest_rate",
    "limit",
    "procedure",
    "registration",
]

ALLOWED_INTENTS = frozenset(
    {
        "direct_answer",
        "catalog_overview",
        "exhaustive_product_details",
    }
)
ALLOWED_PRODUCT_TYPES = frozenset(
    {
        "account",
        "card",
        "digital_banking",
        "insurance",
        "investment",
        "loan",
        "saving",
        "transfer",
    }
)
LLM_MIN_CONFIDENCE = 0.45

EXHAUSTIVE_DETAIL_MARKERS = (
    "cho toi biet toan bo",
    "day du thong tin",
    "thong tin day du",
    "tat ca thong tin",
    "toan bo thong tin",
    "tong hop day du",
    "tong hop toan bo",
)

CATALOG_MARKERS = (
    "cac dich vu",
    "cac goi do",
    "cac goi nay",
    "cac goi",
    "cac goi tren",
    "cac muc do",
    "cac muc nay",
    "cac san pham do",
    "cac san pham nay",
    "cac san pham",
    "co nhung",
    "danh sach",
    "goi nao",
    "goi tren",
    "hien co",
    "nhung dich vu",
    "nhung goi do",
    "nhung goi nay",
    "nhung goi",
    "nhung san pham do",
    "nhung san pham nay",
    "nhung san pham",
    "san pham nao",
    "tung goi",
    "tung goi tren",
    "tung san pham",
    "vua liet ke",
    "vua neu",
)

DETAIL_MARKERS = (
    "bieu phi",
    "cach dang ky",
    "can biet",
    "chi tiet",
    "dac diem",
    "dang ky",
    "dieu kien",
    "han muc",
    "ho so",
    "lai suat",
    "muc cho vay",
    "phi",
    "quyen loi",
    "thoi han",
    "thu tuc",
)

FIELD_MARKERS: dict[RequestedField, tuple[str, ...]] = {
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
        "muc cho vay",
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
    ),
    "benefits": (
        "dac diem",
        "loi ich",
        "quyen loi",
        "uu dai",
    ),
}

PRODUCT_TYPE_HINT_MARKERS = {
    "account": ("tai khoan",),
    "card": ("cac the", "loai the", "mo the", "the ghi no", "the thanh toan", "the tin dung"),
    "digital_banking": ("digibank", "ngan hang so", "sms banking"),
    "insurance": ("bao hiem", "fwd"),
    "investment": ("chung khoan", "dau tu", "quy"),
    "loan": ("goi vay", "khoan vay", "vay"),
    "saving": ("tien gui", "tiet kiem"),
    "transfer": ("chuyen khoan", "chuyen tien", "chuyen va nhan tien", "kieu hoi", "nhan tien"),
}

ADVISORY_QUERY_MARKERS = (
    "co goi nao",
    "co goi vay nao",
    "danh cho",
    "goi nao phu hop",
    "goi vay nao",
    "khoan vay nao",
    "nen chon",
    "nen dung",
    "nen su dung",
    "phu hop",
    "tu van",
)

LOAN_NEED_MARKERS = (
    "chi phi sinh hoat",
    "dong hoc phi",
    "hoc phi",
    "kho khan tai chinh",
    "khong co du tien",
    "phong tro",
    "sinh vien",
    "tien tro",
)

FIELD_ALIASES: dict[str, RequestedField | None] = {
    "benefit": "benefits",
    "benefits": "benefits",
    "condition": "condition",
    "conditions": "condition",
    "customer_condition": "condition",
    "document": "documents",
    "documents": "documents",
    "fee": "fee",
    "fees": "fee",
    "interest": "interest_rate",
    "interest_rate": "interest_rate",
    "limit": "limit",
    "limits": "limit",
    "overview": None,
    "procedure": "procedure",
    "procedures": "procedure",
    "registration": "registration",
}


@dataclass(frozen=True)
class QueryPlan:
    intent: PlanIntent
    route: str
    reason: str
    product_type: str | None = None
    max_subjects: int = 12
    detail_chunks_per_subject: int = 2
    context_top_k: int = 6
    confidence: float = 1.0
    requested_field: RequestedField | None = None
    compose_product_dossiers: bool = False
    max_chars_per_subject: int = 2800
    needs_clarification: bool = False
    clarification_question: str | None = None
    clarification_options: tuple[GraphSubjectOption, ...] = field(default_factory=tuple)

    @property
    def expands_product_details(self) -> bool:
        return self.intent == "exhaustive_product_details"


class QueryPlanner:
    """LLM-assisted strategy planner with a deterministic local fallback.

    The model is only allowed to choose the retrieval strategy. It does not
    provide product facts; the facts still come from graph/vector retrieval.
    """

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings

    async def plan(
        self,
        *,
        question: str,
        history: list[ChatMessage],
        graph_result: GraphRetrievalResult,
        graph_retriever: ProductGraphRetriever,
    ) -> QueryPlan:
        local_plan = self._local_plan(question=question, graph_result=graph_result)
        if not self._should_call_llm():
            return local_plan

        try:
            payload = await self._call_llm_json(
                question=question,
                history=history,
                graph_result=graph_result,
                candidate_options=graph_retriever.suggest_subjects(question, limit=8),
            )
        except Exception:
            return local_plan

        return self._plan_from_llm_payload(
            question=question,
            payload=payload,
            graph_result=graph_result,
            graph_retriever=graph_retriever,
            local_plan=local_plan,
        )

    def _local_plan(
        self,
        *,
        question: str,
        graph_result: GraphRetrievalResult,
    ) -> QueryPlan:
        normalized = _normalize_query_key(question)
        catalog_chunks = _catalog_chunks(graph_result)
        product_count = _catalog_product_count(catalog_chunks)
        product_type = _catalog_product_type(catalog_chunks) or _infer_product_type(normalized)
        requested_field = _requested_field(normalized)
        security_intent = classify_security_intent(question)

        if security_intent.is_security_related:
            return QueryPlan(
                intent="direct_answer",
                route=f"planner_security_{security_intent.kind}",
                reason="security_support_query_does_not_require_catalog_scope",
                product_type="digital_banking" if "digibank" in security_intent.normalized_query else None,
                context_top_k=8,
                confidence=0.9,
                requested_field=requested_field,
            )

        if _is_product_type_advisory_query(normalized, product_type):
            return QueryPlan(
                intent="exhaustive_product_details",
                route="planner_product_type_advisory",
                reason="product_type_scope_with_advisory_need",
                product_type=product_type,
                max_subjects=12,
                detail_chunks_per_subject=2,
                context_top_k=30,
                confidence=0.88,
                requested_field=None,
                compose_product_dossiers=True,
                max_chars_per_subject=3200,
            )

        if product_count > 1 and requested_field is not None:
            return QueryPlan(
                intent="exhaustive_product_details",
                route="planner_catalog_field_details",
                reason="catalog_subject_requests_field_details",
                product_type=product_type,
                max_subjects=12,
                detail_chunks_per_subject=2,
                context_top_k=max(18, min(40, product_count * 3)),
                confidence=0.9,
                requested_field=requested_field,
                compose_product_dossiers=True,
            )

        if (
            product_count > 1
            and _is_catalog_query(normalized)
            and _requests_exhaustive_details(normalized)
        ):
            return QueryPlan(
                intent="exhaustive_product_details",
                route="planner_exhaustive_product_details",
                reason="catalog_question_requests_full_details",
                product_type=product_type,
                max_subjects=12,
                detail_chunks_per_subject=2,
                context_top_k=max(18, min(40, product_count * 3)),
                confidence=0.9,
                requested_field=requested_field,
                compose_product_dossiers=True,
            )

        if product_count > 1 and _is_catalog_query(normalized):
            return QueryPlan(
                intent="catalog_overview",
                route="planner_catalog_overview",
                reason="catalog_question",
                product_type=product_type,
                context_top_k=8,
                confidence=0.85,
                requested_field=requested_field,
            )

        return QueryPlan(
            intent="direct_answer",
            route="planner_direct",
            reason="single_step_retrieval_is_sufficient",
            product_type=product_type,
            context_top_k=6,
            confidence=0.8,
            requested_field=requested_field,
        )

    def _should_call_llm(self) -> bool:
        if self.settings is None:
            return False

        provider = str(getattr(self.settings, "llm_provider", "") or "").strip().lower()
        if provider in {"", "local", "none"}:
            return False

        api_key = getattr(self.settings, "openai_api_key", None) or getattr(
            self.settings,
            "litellm_api_key",
            None,
        )
        return bool(api_key or provider != "openai")

    async def _call_llm_json(
        self,
        *,
        question: str,
        history: list[ChatMessage],
        graph_result: GraphRetrievalResult,
        candidate_options: tuple[GraphSubjectOption, ...],
    ) -> dict[str, Any]:
        if self.settings is None:
            raise RuntimeError("QueryPlanner settings are required for LLM planning.")

        from litellm import acompletion

        response = await acompletion(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval strategy planner for a Vietcombank RAG chatbot. "
                        "Return JSON only. Do not answer the banking question."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_planner_prompt(
                        question=question,
                        history=history,
                        graph_result=graph_result,
                        candidate_options=candidate_options,
                    ),
                },
            ],
            temperature=0,
            api_key=getattr(self.settings, "openai_api_key", None)
            or getattr(self.settings, "litellm_api_key", None),
        )
        content = str(response["choices"][0]["message"]["content"]).strip()
        return _extract_json_object(content)

    def _plan_from_llm_payload(
        self,
        *,
        question: str,
        payload: dict[str, Any],
        graph_result: GraphRetrievalResult,
        graph_retriever: ProductGraphRetriever,
        local_plan: QueryPlan,
    ) -> QueryPlan:
        action = str(payload.get("action") or "answer").strip().lower()
        confidence = _coerce_float(payload.get("confidence"), default=local_plan.confidence)
        reason = str(payload.get("reason") or "llm_strategy_planner").strip()
        if classify_security_intent(question).is_security_related and action == "clarify":
            return local_plan
        if local_plan.route == "planner_product_type_advisory":
            return local_plan

        if action == "clarify" and confidence >= LLM_MIN_CONFIDENCE:
            options = graph_retriever.suggest_subjects(question, limit=5)
            if not options:
                return local_plan
            return QueryPlan(
                intent="direct_answer",
                route="llm_planner_clarification",
                reason=reason or "llm_requested_clarification",
                confidence=confidence,
                requested_field=local_plan.requested_field,
                needs_clarification=True,
                clarification_question=_clarification_question(payload, options),
                clarification_options=options,
            )

        if action not in {"answer", "keep", "plan", ""}:
            return local_plan
        if confidence < LLM_MIN_CONFIDENCE:
            return local_plan

        raw_intent = str(payload.get("intent") or local_plan.intent).strip()
        intent = raw_intent if raw_intent in ALLOWED_INTENTS else local_plan.intent
        product_type = _validated_product_type(payload.get("product_type")) or local_plan.product_type
        requested_field = _requested_field_from_payload(payload) or local_plan.requested_field
        catalog_chunks = _catalog_chunks(graph_result)
        catalog_count = _catalog_product_count(catalog_chunks)
        normalized_question = _normalize_query_key(question)

        if (
            intent in {"catalog_overview", "exhaustive_product_details"}
            and _has_product_detail_signal(graph_result)
            and not _is_catalog_query(normalized_question)
            and not _is_multi_subject_detail_query(normalized_question)
        ):
            return local_plan

        if intent in {"catalog_overview", "exhaustive_product_details"} and (
            product_type is None and catalog_count == 0
        ):
            options = graph_retriever.suggest_subjects(question, limit=5)
            return QueryPlan(
                intent="direct_answer",
                route="llm_planner_clarification",
                reason=reason or "missing_catalog_scope",
                confidence=confidence,
                requested_field=requested_field,
                needs_clarification=True,
                clarification_question=_clarification_question(payload, options),
                clarification_options=options,
            )

        if intent == "catalog_overview":
            return QueryPlan(
                intent="catalog_overview",
                route="llm_planner",
                reason=reason or "llm_selected_catalog_overview",
                product_type=product_type,
                max_subjects=_coerce_int(payload.get("max_subjects"), default=local_plan.max_subjects),
                context_top_k=max(8, min(24, catalog_count or local_plan.context_top_k)),
                confidence=confidence,
                requested_field=requested_field,
            )

        if intent == "exhaustive_product_details":
            max_subjects = _coerce_int(payload.get("max_subjects"), default=local_plan.max_subjects)
            max_subjects = max(1, min(24, max_subjects))
            if catalog_count > 1:
                max_subjects = max(max_subjects, min(24, catalog_count))
            product_count = catalog_count or max_subjects
            compose_dossiers = bool(payload.get("compose_product_dossiers", True))
            if local_plan.compose_product_dossiers or (catalog_count > 1 and requested_field is not None):
                compose_dossiers = True
            return QueryPlan(
                intent="exhaustive_product_details",
                route="llm_planner",
                reason=reason or "llm_selected_query_composition",
                product_type=product_type,
                max_subjects=max_subjects,
                detail_chunks_per_subject=max(
                    1,
                    min(
                        4,
                        _coerce_int(
                            payload.get("detail_chunks_per_subject"),
                            default=local_plan.detail_chunks_per_subject,
                        ),
                    ),
                ),
                context_top_k=max(18, min(48, product_count * 3)),
                confidence=confidence,
                requested_field=requested_field,
                compose_product_dossiers=compose_dossiers,
                max_chars_per_subject=max(
                    1600,
                    min(
                        4200,
                        _coerce_int(
                            payload.get("max_chars_per_subject"),
                            default=local_plan.max_chars_per_subject,
                        ),
                    ),
                ),
            )

        return QueryPlan(
            intent="direct_answer",
            route="llm_planner",
            reason=reason or "llm_selected_direct_answer",
            product_type=product_type,
            context_top_k=max(4, min(12, local_plan.context_top_k)),
            confidence=confidence,
            requested_field=requested_field,
        )


def _build_planner_prompt(
    *,
    question: str,
    history: list[ChatMessage],
    graph_result: GraphRetrievalResult,
    candidate_options: tuple[GraphSubjectOption, ...],
) -> str:
    compact_history = "\n".join(
        f"{message.role}: {_truncate(message.content, 900)}" for message in history[-8:]
    )
    catalog_summary = _catalog_summary(graph_result)
    options = "\n".join(
        (
            f"- {option.title} [{option.subject_type}; "
            f"product_type={option.product_type or 'unknown'}; "
            f"parent={option.parent_title or option.category_title or 'none'}]"
        )
        for option in candidate_options[:8]
    )
    return f"""Decide how the chatbot should retrieve context before answering.

Rules:
- Do not provide the banking answer. Only choose a retrieval plan.
- Use conversation history only to resolve references like "cac goi do", "cac goi tren", "toan bo cac goi tren", "tung goi".
- If the current question names a concrete product or subject, prefer that current subject and ignore unrelated products from history.
- Choose intent="catalog_overview" when the user only asks what products/packages currently exist.
- Choose intent="exhaustive_product_details" when the user asks for full information, details, conditions, fees, rates, documents, or procedures for many/all products/packages.
- Choose intent="exhaustive_product_details" with product_type when the user asks which product/package fits their situation and the product type is clear, for example a loan package suitable for a student.
- For exhaustive multi-product questions, set compose_product_dossiers=true so the retriever builds one context dossier per product.
- Choose action="clarify" only if the product group/scope cannot be inferred from the current question, product_type hints, history, catalog summary, or candidates.
- Never invent product names. Use product_type instead of making up subjects.
- Return a single JSON object, no markdown.

Allowed product_type values: account, card, digital_banking, insurance, investment, loan, saving, transfer, null.
Allowed requested_fields values: overview, benefits, condition, documents, fee, interest_rate, limit, procedure, registration.

Schema:
{{
  "action": "answer|clarify",
  "intent": "direct_answer|catalog_overview|exhaustive_product_details",
  "product_type": "saving|null",
  "requested_fields": ["overview"],
  "compose_product_dossiers": true,
  "max_subjects": 12,
  "detail_chunks_per_subject": 2,
  "confidence": 0.0,
  "reason": "short reason"
}}

Conversation history:
{compact_history or "none"}

Current graph/catalog signal:
{catalog_summary or "none"}

Candidate subjects:
{options or "none"}

Current question:
{question}
"""


def _catalog_summary(graph_result: GraphRetrievalResult) -> str:
    lines: list[str] = []
    for chunk in _catalog_chunks(graph_result)[:6]:
        item_titles = _catalog_item_titles(chunk)
        lines.append(
            " | ".join(
                part
                for part in (
                    f"title={chunk.title}",
                    f"product_type={chunk.product_type or 'unknown'}",
                    f"item_count={len(item_titles)}",
                    f"items={', '.join(item_titles[:16])}",
                )
                if part
            )
        )
    return "\n".join(lines)


def _has_product_detail_signal(graph_result: GraphRetrievalResult) -> bool:
    return any(chunk.section == "product_detail" for chunk in graph_result.chunks)


def _catalog_item_titles(chunk: RetrievedChunk) -> list[str]:
    items = chunk.metadata.get("items")
    if not isinstance(items, list):
        return []

    titles: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


def _requests_exhaustive_details(normalized_query: str) -> bool:
    if any(marker in normalized_query for marker in EXHAUSTIVE_DETAIL_MARKERS):
        return True
    return ("toan bo" in normalized_query and any(
        marker in normalized_query for marker in DETAIL_MARKERS + CATALOG_MARKERS
    )) or (
        any(marker in normalized_query for marker in DETAIL_MARKERS)
        and _is_multi_subject_detail_query(normalized_query)
    )


def _is_product_type_advisory_query(normalized_query: str, product_type: str | None) -> bool:
    if product_type != "loan":
        return False
    if _contains_any(normalized_query, ADVISORY_QUERY_MARKERS):
        return True
    return _contains_any(normalized_query, LOAN_NEED_MARKERS) and _requested_detail_count(
        normalized_query
    ) >= 2


def _requested_detail_count(normalized_query: str) -> int:
    detail_markers = (
        "bieu phi",
        "dieu kien",
        "han muc",
        "ho so",
        "lai suat",
        "muc cho vay",
        "muc vay",
        "phi",
        "thoi han",
        "thu tuc",
    )
    return sum(1 for marker in detail_markers if marker in normalized_query)


def _is_catalog_query(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in CATALOG_MARKERS)


def _is_multi_subject_detail_query(normalized_query: str) -> bool:
    return any(
        marker in normalized_query
        for marker in (
            "cac goi do",
            "cac goi nay",
            "cac goi tren",
            "cac muc do",
            "cac muc nay",
            "cac san pham do",
            "cac san pham nay",
            "cac san pham tren",
            "goi tren",
            "moi goi",
            "moi san pham",
            "nhung goi do",
            "nhung goi nay",
            "nhung goi tren",
            "nhung san pham do",
            "nhung san pham nay",
            "nhung san pham tren",
            "san pham do",
            "san pham nay",
            "san pham tren",
            "tung goi",
            "tung san pham",
            "vua liet ke",
            "vua neu",
        )
    )


def _requested_field(normalized_query: str) -> RequestedField | None:
    for field_name, markers in FIELD_MARKERS.items():
        if any(marker in normalized_query for marker in markers):
            return field_name
    return None


def _requested_field_from_payload(payload: dict[str, Any]) -> RequestedField | None:
    raw_fields = payload.get("requested_fields")
    if raw_fields is None:
        raw_fields = payload.get("requested_field")
    if isinstance(raw_fields, str):
        values: Sequence[Any] = (raw_fields,)
    elif isinstance(raw_fields, Sequence):
        values = raw_fields
    else:
        values = ()

    for value in values:
        normalized = str(value or "").strip().lower()
        mapped = FIELD_ALIASES.get(normalized)
        if mapped:
            return mapped
    return None


def _catalog_chunks(graph_result: GraphRetrievalResult) -> list[RetrievedChunk]:
    return [chunk for chunk in graph_result.chunks if chunk.section == "product_catalog"]


def _catalog_product_count(chunks: Sequence[object]) -> int:
    seen_urls: set[str] = set()
    count = 0
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {})
        if not isinstance(metadata, dict):
            continue
        items = metadata.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            key = url or title
            if not key or key in seen_urls:
                continue
            seen_urls.add(key)
            count += 1
    return count


def _catalog_product_type(chunks: Sequence[object]) -> str | None:
    for chunk in chunks:
        product_type = getattr(chunk, "product_type", None)
        if isinstance(product_type, str) and product_type.strip():
            return product_type
    return None


def _validated_product_type(value: object) -> str | None:
    product_type = str(value or "").strip().lower()
    if product_type in {"", "none", "null"}:
        return None
    return product_type if product_type in ALLOWED_PRODUCT_TYPES else None


def _infer_product_type(normalized_query: str) -> str | None:
    padded_query = f" {normalized_query} "
    for product_type, markers in PRODUCT_TYPE_HINT_MARKERS.items():
        for marker in markers:
            if f" {marker} " in padded_query or marker in normalized_query:
                return product_type
    return None


def _contains_any(normalized_query: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized_query for marker in markers)


def _clarification_question(
    payload: dict[str, Any],
    options: tuple[GraphSubjectOption, ...],
) -> str:
    question = str(payload.get("clarification_question") or "").strip()
    if question:
        return question
    if options:
        return "Bạn muốn hỏi về nhóm sản phẩm hoặc dịch vụ nào của Vietcombank?"
    return "Bạn vui lòng nêu rõ sản phẩm, nhóm sản phẩm hoặc dịch vụ cần tra cứu."


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
        raise ValueError("planner response must be a JSON object")
    return payload


def _coerce_float(value: object, *, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return default
    try:
        return int(str(value))
    except ValueError:
        return default


def _truncate(text: str, max_chars: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars].rstrip()}..."


def _normalize_query_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(TOKEN_PATTERN.findall(no_accents.casefold()))
