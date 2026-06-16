from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage
from apps.api.app.rag.retrieval.graph import GraphSubjectOption, ProductGraphRetriever
from apps.api.app.rag.security_intent import classify_security_intent

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

CONTEXTUAL_MARKERS = (
    "ben tren",
    "cai do",
    "cai nay",
    "cai tren",
    "cac goi do",
    "cac goi nay",
    "cac goi tren",
    "cac muc do",
    "cac muc nay",
    "cac san pham do",
    "cac san pham nay",
    "cac san pham tren",
    "cua no",
    "dich vu do",
    "dich vu nay",
    "goi tren",
    "goi do",
    "goi nay",
    "mo the do",
    "nhom do",
    "nhom nay",
    "nhung goi do",
    "nhung goi nay",
    "nhung goi tren",
    "nhung san pham do",
    "nhung san pham nay",
    "nhung san pham tren",
    "san pham do",
    "san pham nay",
    "san pham tren",
    "the do",
    "the nay",
    "ve no",
    "vua liet ke",
    "vua neu",
    "vua noi",
)

DETAIL_MARKERS = (
    "bieu phi",
    "co gi khac",
    "dang ky",
    "dieu kien",
    "han muc",
    "ho so",
    "khac gi",
    "khac nhau",
    "lai suat",
    "mo the",
    "phi",
    "quyen loi",
    "so sanh",
    "thu tuc",
)

FIELD_MARKERS = {
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
    "procedure": (
        "cac buoc",
        "quy trinh",
        "thu tuc",
    ),
}

EXHAUSTIVE_DETAIL_MARKERS = (
    "cho toi biet toan bo",
    "day du thong tin",
    "thong tin day du",
    "tat ca thong tin",
    "toan bo thong tin",
    "tong hop day du",
    "tong hop toan bo",
)

LIST_MARKERS = (
    "bao gom",
    "cac goi do",
    "cac goi nay",
    "cac goi",
    "cac goi tren",
    "cac muc do",
    "cac muc nay",
    "cac san pham do",
    "cac san pham nay",
    "cac san pham",
    "cac dich vu",
    "co cac",
    "co nhung",
    "danh sach",
    "gom cac",
    "gom nhung",
    "liet ke",
    "goi tren",
    "hien co",
    "moi goi",
    "moi san pham",
    "nhung goi do",
    "nhung goi nay",
    "nhung goi",
    "nhung san pham",
    "nhung dich vu",
    "nhung goi tren",
    "nhung san pham do",
    "nhung san pham nay",
    "nhung san pham tren",
    "san pham nao",
    "san pham tren",
    "tung goi",
    "tung san pham",
    "vua liet ke",
    "vua neu",
)

PRODUCT_TYPE_HINTS = (
    "bao hiem",
    "chuyen tien",
    "chuyen va nhan tien",
    "digibank",
    "fwd",
    "kieu hoi",
    "mo the",
    "nhan tien",
    "ngan hang so",
    "tai khoan",
    "the thanh toan",
    "the tin dung",
    "tiet kiem",
    "vay",
)

PRODUCT_TYPE_HINT_MARKERS = {
    "insurance": ("bao hiem", "fwd"),
    "card": ("cac the", "loai the", "mo the", "the ghi no", "the thanh toan", "the tin dung"),
    "loan": ("khoan vay", "vay"),
    "saving": ("tien gui", "tiet kiem"),
    "transfer": ("chuyen khoan", "chuyen tien", "chuyen va nhan tien", "kieu hoi", "nhan tien"),
    "digital_banking": ("digibank", "ngan hang so", "sms banking"),
    "account": ("tai khoan",),
    "investment": ("chung khoan", "dau tu", "quy"),
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
MAX_RECENT_PRODUCT_CLARIFICATION_OPTIONS = 12


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    route: str = "none"
    needs_clarification: bool = False
    clarification_question: str | None = None
    clarification_options: tuple[GraphSubjectOption, ...] = field(default_factory=tuple)
    confidence: float | None = None
    reason: str | None = None

    @property
    def query_was_rewritten(self) -> bool:
        return self.rewritten_query.strip() != self.original_query.strip()


@dataclass(frozen=True)
class ClarificationSelection:
    previous_question: str
    selected_option: GraphSubjectOption
    product_options: tuple[GraphSubjectOption, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClarificationAllSelection:
    current_question: str
    intent_question: str
    product_options: tuple[GraphSubjectOption, ...]
    category_title: str | None = None


class QueryRewriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def rewrite(
        self,
        *,
        question: str,
        history: list[ChatMessage],
        graph_retriever: ProductGraphRetriever,
    ) -> QueryRewriteResult:
        local_result = self._local_rewrite(
            question=question,
            history=history,
            graph_retriever=graph_retriever,
        )

        if not self._should_call_llm(question, history, local_result):
            return local_result

        if not self._can_call_llm():
            return local_result

        try:
            payload = await self._call_llm_json(
                question=question,
                history=history,
                recent_subject=graph_retriever.latest_subject(history),
                candidate_options=graph_retriever.suggest_subjects(question, limit=8),
            )
        except Exception:
            return local_result

        return self._result_from_llm_payload(
            question=question,
            payload=payload,
            graph_retriever=graph_retriever,
            local_result=local_result,
        )

    def _local_rewrite(
        self,
        *,
        question: str,
        history: list[ChatMessage],
        graph_retriever: ProductGraphRetriever,
    ) -> QueryRewriteResult:
        normalized = _normalize_query_key(question)
        security_intent = classify_security_intent(question)
        if security_intent.is_security_related:
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route=f"local_security_{security_intent.kind}",
                confidence=0.95,
                reason="security_support_query_does_not_require_product_subject",
            )

        clarification_selection = _clarification_selection_from_history(
            question=question,
            history=history,
            graph_retriever=graph_retriever,
        )
        if clarification_selection:
            return _rewrite_from_clarification_selection(clarification_selection)

        all_selection = _all_options_selection_from_history(
            question=question,
            history=history,
            graph_retriever=graph_retriever,
        )
        if all_selection:
            return _rewrite_from_all_options_selection(all_selection)

        recent_subject = graph_retriever.latest_subject(history)
        explicit_subject = _latest_specific_user_subject(
            history=history,
            graph_retriever=graph_retriever,
        )
        exact_subjects = graph_retriever.match_subjects(question, limit=3)
        specific_exact_subjects = tuple(
            option
            for option in exact_subjects
            if option.subject_type == "product"
            or option.parent_title
            or len(_normalize_query_key(option.title).split()) > 2
        )
        candidate_options = _clarification_options_for_query(
            question,
            graph_retriever=graph_retriever,
            limit=5,
        )
        recent_product_options = _recent_product_options_from_history(
            history=history,
            graph_retriever=graph_retriever,
            product_type=_infer_product_type(normalized),
        )
        latest_user_product = _latest_specific_user_product(
            history=history,
            graph_retriever=graph_retriever,
        )
        ambiguous_detail_after_product_list = (
            _is_detail_query(normalized)
            and not specific_exact_subjects
            and latest_user_product is None
            and len(recent_product_options) > 1
            and not _requests_all_recent_products(normalized)
        )

        if specific_exact_subjects and _contains_any(normalized, DETAIL_MARKERS):
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route="local_exact_subject",
                confidence=1.0,
                reason="exact_subject_match",
            )

        if ambiguous_detail_after_product_list:
            clarification_options = recent_product_options[
                :MAX_RECENT_PRODUCT_CLARIFICATION_OPTIONS
            ]
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route="local_multi_product_detail_clarification",
                needs_clarification=True,
                clarification_question=_clarification_message(
                    clarification_options,
                    fallback=_detail_clarification_message(normalized),
                ),
                clarification_options=clarification_options,
                confidence=0.3,
                reason="ambiguous_detail_after_multi_product_context",
            )

        if explicit_subject and not specific_exact_subjects and _contains_any(normalized, DETAIL_MARKERS):
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=_append_subject(question, explicit_subject),
                route="local_context_detail",
                confidence=0.9,
                reason="resolved_detail_from_explicit_user_subject",
            )

        if _is_product_type_advisory_query(normalized):
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route="local_product_type_advisory",
                confidence=0.9,
                reason="product_type_scope_with_advisory_need",
            )

        if _is_under_specified_detail_query(normalized) and not specific_exact_subjects:
            if recent_subject:
                return QueryRewriteResult(
                    original_query=question,
                    rewritten_query=_append_subject(question, recent_subject),
                    route="local_context_detail",
                    confidence=0.82,
                    reason="resolved_detail_from_recent_single_subject",
                )
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route="local_clarification",
                needs_clarification=True,
                clarification_question=_clarification_message(candidate_options),
                clarification_options=candidate_options,
                confidence=0.25,
                reason="under_specified_detail_query",
            )

        if _is_multi_subject_follow_up(normalized):
            subject = _product_type_subject(_infer_product_type(normalized)) or _latest_product_type_subject(history)
            if subject:
                return QueryRewriteResult(
                    original_query=question,
                    rewritten_query=_append_subject(question, subject),
                    route="local_multi_subject_context",
                    confidence=0.88,
                    reason="resolved_multi_subject_follow_up",
                )
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route="local_clarification",
                needs_clarification=True,
                clarification_question=_clarification_message(candidate_options),
                clarification_options=candidate_options,
                confidence=0.0,
                reason="missing_multi_subject_scope",
            )

        if _is_contextual_follow_up(normalized):
            if recent_subject:
                return QueryRewriteResult(
                    original_query=question,
                    rewritten_query=_append_subject(question, recent_subject),
                    route="local_context",
                    confidence=0.85,
                    reason="resolved_from_recent_subject",
                )
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route="local_clarification",
                needs_clarification=True,
                clarification_question=_clarification_message(candidate_options),
                clarification_options=candidate_options,
                confidence=0.0,
                reason="missing_follow_up_subject",
            )

        if (
            _is_under_specified_exhaustive_query(normalized)
            and not recent_subject
            and not specific_exact_subjects
        ):
            options = candidate_options
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route="local_clarification",
                needs_clarification=True,
                clarification_question=_clarification_message(options),
                clarification_options=options,
                confidence=0.2,
                reason="under_specified_exhaustive_query",
            )

        if (
            _is_under_specified_detail_query(normalized)
            and not recent_subject
            and not specific_exact_subjects
        ):
            options = candidate_options
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=question,
                route="local_clarification",
                needs_clarification=True,
                clarification_question=_clarification_message(options),
                clarification_options=options,
                confidence=0.25,
                reason="under_specified_detail_query",
            )

        return QueryRewriteResult(
            original_query=question,
            rewritten_query=question,
            route="none",
            confidence=1.0,
        )

    def _should_call_llm(
        self,
        question: str,
        history: list[ChatMessage],
        local_result: QueryRewriteResult,
    ) -> bool:
        normalized = _normalize_query_key(question)
        if local_result.route in {
            "local_exact_subject",
            "local_choice",
            "local_choice_clarification",
            "local_context_detail",
            "local_multi_product_detail_clarification",
            "local_all_options_choice",
            "local_multi_subject_context",
            "local_security_public_info",
            "local_security_account_recovery",
        }:
            return False
        if local_result.needs_clarification:
            return True
        if local_result.query_was_rewritten:
            return False
        if _is_contextual_follow_up(normalized):
            return True
        if _looks_noisy_or_short(normalized):
            return True
        return _contains_any(normalized, DETAIL_MARKERS)

    def _can_call_llm(self) -> bool:
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
        recent_subject: str | None,
        candidate_options: tuple[GraphSubjectOption, ...],
    ) -> dict[str, Any]:
        from litellm import acompletion

        response = await acompletion(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": _build_rewrite_prompt(
                        question=question,
                        history=history,
                        recent_subject=recent_subject,
                        candidate_options=candidate_options,
                    ),
                }
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
        graph_retriever: ProductGraphRetriever,
        local_result: QueryRewriteResult,
    ) -> QueryRewriteResult:
        action = str(payload.get("action") or "").strip().lower()
        rewritten_query = str(payload.get("rewritten_question") or question).strip() or question
        confidence = _coerce_confidence(payload.get("confidence"))
        candidate_text = " ".join(
            str(item)
            for item in payload.get("candidate_subjects", [])
            if isinstance(item, str) and item.strip()
        )
        options = graph_retriever.suggest_subjects(
            " ".join(part for part in (question, rewritten_query, candidate_text) if part),
            limit=5,
        )
        options = _prefer_specific_category_options(
            _filter_options_for_query(question, options)
        )
        original_options = _clarification_options_for_query(
            question,
            graph_retriever=graph_retriever,
            limit=8,
        )
        if local_result.needs_clarification and original_options:
            options = original_options
        if not options:
            options = original_options

        if action == "clarify" or (confidence is not None and confidence < 0.55 and options):
            return QueryRewriteResult(
                original_query=question,
                rewritten_query=rewritten_query,
                route="llm_clarification",
                needs_clarification=True,
                clarification_question=_clarification_message(options),
                clarification_options=options,
                confidence=confidence,
                reason=str(payload.get("reason") or "llm_requested_clarification"),
            )

        if local_result.needs_clarification and not _has_validated_specific_rewrite(
            original_query=question,
            rewritten_query=rewritten_query,
            graph_retriever=graph_retriever,
            original_options=original_options,
        ):
            return _clarification_result_with_options(local_result, options or original_options)

        if action not in {"rewrite", "keep", ""}:
            return local_result

        if not rewritten_query:
            return local_result

        probe = graph_retriever.retrieve(rewritten_query, history=[], top_k=3)
        if action == "rewrite" and probe.route == "default" and local_result.needs_clarification:
            return local_result

        return QueryRewriteResult(
            original_query=question,
            rewritten_query=rewritten_query,
            route="llm_rewrite" if rewritten_query != question else "llm_keep",
            confidence=confidence,
            reason=str(payload.get("reason") or "").strip() or None,
        )


def _build_rewrite_prompt(
    *,
    question: str,
    history: list[ChatMessage],
    recent_subject: str | None,
    candidate_options: tuple[GraphSubjectOption, ...],
) -> str:
    compact_history = "\n".join(
        f"{message.role}: {message.content}" for message in history[-8:]
    )
    options = "\n".join(
        f"- {option.title} [{option.subject_type}]"
        for option in candidate_options[:8]
    )
    return f"""Bạn là bộ chuẩn hóa truy vấn cho chatbot Vietcombank RAG.

Nhiệm vụ:
- Sửa lỗi chính tả nhẹ, thiếu dấu tiếng Việt, thiếu dấu câu.
- Nếu câu hỏi là follow-up như "nó", "cái này", "thẻ đó", hãy viết lại thành câu hỏi đầy đủ dựa trên lịch sử.
- Nếu không chắc người dùng hỏi sản phẩm/nhóm/vấn đề nào, chọn action="clarify".
- Không bịa tên sản phẩm. Ưu tiên subject trong lịch sử hoặc trong danh sách ứng viên.
- Chỉ trả về JSON object, không markdown.

Schema JSON:
{{
  "action": "keep|rewrite|clarify",
  "rewritten_question": "câu hỏi đã chuẩn hóa hoặc câu gốc",
  "clarification_question": "câu hỏi lại ngắn gọn nếu action=clarify",
  "candidate_subjects": ["tên lựa chọn liên quan"],
  "confidence": 0.0,
  "reason": "lý do ngắn"
}}

Subject gần nhất trong lịch sử: {recent_subject or "không có"}

Ứng viên từ catalog/graph:
{options or "- không có"}

Lịch sử hội thoại:
{compact_history or "không có"}

Câu hỏi hiện tại:
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
        raise ValueError("query rewrite response must be a JSON object")
    return payload


def _clarification_message(
    options: tuple[GraphSubjectOption, ...],
    *,
    fallback: str | None = None,
) -> str:
    prefix = fallback or (
        "Bạn đang muốn hỏi về sản phẩm, nhóm sản phẩm hoặc dịch vụ nào của Vietcombank?"
    )
    if not options:
        return f"{prefix} Bạn vui lòng nêu rõ tên để tôi tra đúng nguồn."

    lines = [
        f"{prefix} Bạn có thể chọn một trong các mục liên quan sau:",
    ]
    for index, option in enumerate(options, start=1):
        label = "nhóm" if option.subject_type == "category" else "sản phẩm"
        lines.append(f"{index}. {option.title} ({label})")
    return "\n".join(lines)


def _clarification_result_with_options(
    result: QueryRewriteResult,
    options: tuple[GraphSubjectOption, ...],
) -> QueryRewriteResult:
    clarification_options = result.clarification_options or options
    return QueryRewriteResult(
        original_query=result.original_query,
        rewritten_query=result.rewritten_query,
        route=result.route,
        needs_clarification=True,
        clarification_question=_clarification_message(clarification_options),
        clarification_options=clarification_options,
        confidence=result.confidence,
        reason=result.reason,
    )


def _clarification_options_for_query(
    query: str,
    *,
    graph_retriever: ProductGraphRetriever,
    limit: int,
) -> tuple[GraphSubjectOption, ...]:
    options = graph_retriever.suggest_subjects(query, limit=max(limit, 8))
    filtered = _filter_options_for_query(query, options)
    preferred = _prefer_specific_category_options(filtered or options)
    return preferred[:limit]


def _filter_options_for_query(
    query: str,
    options: tuple[GraphSubjectOption, ...],
) -> tuple[GraphSubjectOption, ...]:
    product_type = _infer_product_type(_normalize_query_key(query))
    if product_type is None:
        return options
    return tuple(option for option in options if option.product_type == product_type)


def _prefer_specific_category_options(
    options: tuple[GraphSubjectOption, ...],
) -> tuple[GraphSubjectOption, ...]:
    products = [option for option in options if option.subject_type == "product"]
    child_categories = [
        option
        for option in options
        if option.subject_type == "category" and option.parent_title
    ]
    parent_categories = [
        option
        for option in options
        if option.subject_type == "category" and not option.parent_title
    ]
    if child_categories:
        return tuple(child_categories)
    return tuple([*products, *parent_categories])


def _recent_product_options_from_history(
    *,
    history: list[ChatMessage],
    graph_retriever: ProductGraphRetriever,
    product_type: str | None,
) -> tuple[GraphSubjectOption, ...]:
    for message in reversed(history[-6:]):
        if message.role != "assistant" or not message.content.strip():
            continue

        options = _numbered_product_options_from_text(
            message.content,
            graph_retriever=graph_retriever,
            product_type=product_type,
        )
        if options:
            return options

        normalized_text = _normalize_query_key(message.content)
        if not _looks_like_multi_subject_answer(normalized_text):
            continue

        matched_options = tuple(
            option
            for option in graph_retriever.match_subjects(message.content, limit=16)
            if option.subject_type == "product"
            and (product_type is None or option.product_type == product_type)
        )
        matched_options = _sort_options_by_text_order(
            _dedupe_subject_options(matched_options),
            normalized_text,
        )
        if len(matched_options) >= 2:
            return matched_options
    return ()


def _numbered_product_options_from_text(
    text: str,
    *,
    graph_retriever: ProductGraphRetriever,
    product_type: str | None,
) -> tuple[GraphSubjectOption, ...]:
    option_titles = _parse_numbered_options(text)
    if len(option_titles) < 2:
        return ()

    options = tuple(
        option
        for title in option_titles.values()
        if (option := _resolve_subject_option(title, graph_retriever=graph_retriever))
        is not None
        and option.subject_type == "product"
        and (product_type is None or option.product_type == product_type)
    )
    return _dedupe_subject_options(options)


def _looks_like_multi_subject_answer(normalized_text: str) -> bool:
    return any(
        marker in normalized_text
        for marker in (
            "cac goi",
            "cac san pham",
            "cac dich vu",
            "danh sach",
            "gom",
            "hien co",
            "nhu",
            "phuc vu",
            "sau",
        )
    )


def _sort_options_by_text_order(
    options: tuple[GraphSubjectOption, ...],
    normalized_text: str,
) -> tuple[GraphSubjectOption, ...]:
    return tuple(
        sorted(
            options,
            key=lambda option: _subject_position(normalized_text, option.title),
        )
    )


def _subject_position(normalized_text: str, title: str) -> int:
    position = normalized_text.find(_normalize_query_key(title))
    return position if position >= 0 else len(normalized_text)


def _detail_clarification_message(normalized_query: str) -> str:
    if any(marker in normalized_query for marker in FIELD_MARKERS["documents"]):
        return "Bạn muốn hỏi hồ sơ cần chuẩn bị của sản phẩm/gói nào?"
    if any(marker in normalized_query for marker in FIELD_MARKERS["condition"]):
        return "Bạn muốn hỏi điều kiện của sản phẩm/gói nào?"
    if any(marker in normalized_query for marker in FIELD_MARKERS["fee"]):
        return "Bạn muốn hỏi biểu phí của sản phẩm/gói nào?"
    if any(marker in normalized_query for marker in FIELD_MARKERS["interest_rate"]):
        return "Bạn muốn hỏi lãi suất của sản phẩm/gói nào?"
    if any(marker in normalized_query for marker in FIELD_MARKERS["procedure"]):
        return "Bạn muốn hỏi thủ tục/quy trình của sản phẩm/gói nào?"
    return "Bạn muốn hỏi thông tin của sản phẩm/gói nào?"


def _dedupe_subject_options(
    options: tuple[GraphSubjectOption, ...],
) -> tuple[GraphSubjectOption, ...]:
    deduped: list[GraphSubjectOption] = []
    seen: set[tuple[str, str]] = set()
    for option in options:
        key = _subject_option_key(option)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return tuple(deduped)


def _clarification_selection_from_history(
    *,
    question: str,
    history: list[ChatMessage],
    graph_retriever: ProductGraphRetriever,
) -> ClarificationSelection | None:
    selected_index = _parse_numeric_choice(question)
    selected_text_key = _parse_text_choice(question) if selected_index is None else None
    if selected_index is None and not selected_text_key:
        return None

    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message.role != "assistant":
            continue
        option_titles = _parse_numbered_options(message.content)
        selected_title = (
            option_titles.get(selected_index)
            if selected_index is not None
            else _matched_text_option_title(selected_text_key or "", option_titles)
        )
        if not selected_title:
            continue

        selected_option = _resolve_subject_option(selected_title, graph_retriever=graph_retriever)
        if selected_option is None:
            continue

        previous_question = _previous_user_question(history, before_index=index)
        if not previous_question:
            continue

        return ClarificationSelection(
            previous_question=previous_question,
            selected_option=selected_option,
            product_options=graph_retriever.product_options_for_category(selected_option),
        )
    return None


def _all_options_selection_from_history(
    *,
    question: str,
    history: list[ChatMessage],
    graph_retriever: ProductGraphRetriever,
) -> ClarificationAllSelection | None:
    normalized = _normalize_query_key(question)
    if not _is_all_options_selection(normalized):
        return None
    if _has_explicit_current_subject(question, graph_retriever=graph_retriever):
        return None

    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message.role != "assistant":
            continue

        option_titles = _parse_numbered_options(message.content)
        if len(option_titles) < 2:
            continue

        resolved_options = tuple(
            option
            for title in option_titles.values()
            if (option := _resolve_subject_option(title, graph_retriever=graph_retriever))
            is not None
        )
        product_options = tuple(
            option for option in resolved_options if option.subject_type == "product"
        )
        if len(product_options) < 2:
            continue

        intent_question = (
            _previous_detail_question(history, before_index=index)
            or _previous_user_question(history, before_index=index)
            or question
        )
        return ClarificationAllSelection(
            current_question=question,
            intent_question=intent_question,
            product_options=product_options,
            category_title=_common_category_title(product_options),
        )
    return None


def _rewrite_from_clarification_selection(selection: ClarificationSelection) -> QueryRewriteResult:
    previous_normalized = _normalize_query_key(selection.previous_question)
    selected_option = selection.selected_option
    if (
        selected_option.subject_type == "category"
        and _contains_any(previous_normalized, DETAIL_MARKERS)
        and len(selection.product_options) > 1
    ):
        clarification_options = selection.product_options[
            :MAX_RECENT_PRODUCT_CLARIFICATION_OPTIONS
        ]
        return QueryRewriteResult(
            original_query=selection.previous_question,
            rewritten_query=selection.previous_question,
            route="local_choice_clarification",
            needs_clarification=True,
            clarification_question=_clarification_message(
                clarification_options,
                fallback=f"Bạn đã chọn {selected_option.title}. Bạn muốn hỏi sản phẩm nào?",
            ),
            clarification_options=clarification_options,
            confidence=0.9,
            reason="selected_group_requires_product",
        )

    subject_title = selected_option.title
    if selected_option.subject_type == "category" and len(selection.product_options) == 1:
        subject_title = selection.product_options[0].title

    return QueryRewriteResult(
        original_query=selection.previous_question,
        rewritten_query=_append_subject(selection.previous_question, subject_title),
        route="local_choice",
        confidence=0.95,
        reason="resolved_from_clarification_choice",
    )


def _rewrite_from_all_options_selection(selection: ClarificationAllSelection) -> QueryRewriteResult:
    product_titles = "; ".join(option.title for option in selection.product_options)
    if selection.category_title:
        scope = f"trong nhom {selection.category_title}"
    else:
        scope = "da duoc liet ke"

    rewritten_query = (
        f"{selection.intent_question}. Hay tra loi cho tung san pham {scope}: "
        f"{product_titles}"
    )
    return QueryRewriteResult(
        original_query=selection.current_question,
        rewritten_query=rewritten_query,
        route="local_all_options_choice",
        confidence=0.94,
        reason="resolved_all_clarification_options_with_previous_intent",
    )


def _parse_numeric_choice(question: str) -> int | None:
    normalized = _normalize_query_key(question)
    match = re.fullmatch(r"(?:chon|lua chon|so)?\s*(\d{1,2})", normalized)
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _parse_text_choice(question: str) -> str | None:
    normalized = _normalize_query_key(question)
    for prefix in (
        "chon ",
        "lua chon ",
        "toi chon ",
        "minh chon ",
        "toi muon chon ",
        "toi muon hoi ",
    ):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix).strip()
            break
    normalized = _choice_label_key(normalized)
    return normalized if len(normalized.split()) >= 2 else None


def _is_all_options_selection(normalized_query: str) -> bool:
    if not normalized_query:
        return False
    token_count = len(TOKEN_PATTERN.findall(normalized_query))
    if normalized_query in {"tat ca", "toan bo", "het", "ca tat ca"}:
        return True
    if re.fullmatch(
        r"ca\s+\d{1,2}(?:\s+(?:muc|goi|san pham|lua chon))?(?:\s+tren)?",
        normalized_query,
    ):
        return True
    return any(
        marker in normalized_query
        for marker in (
            "ca bon",
            "cac lua chon tren",
            "cac muc tren",
            "cac san pham tren",
            "cac goi tren",
            "nhung san pham tren",
            "nhung goi tren",
        )
    ) or (
        token_count <= 5
        and any(marker in normalized_query for marker in ("tat ca", "toan bo", "het"))
        and any(marker in normalized_query for marker in ("tren", "lua chon", "muc", "goi", "san pham"))
    )


def _has_explicit_current_subject(
    question: str,
    *,
    graph_retriever: ProductGraphRetriever,
) -> bool:
    normalized_question = _normalize_query_key(question)
    query_tokens = set(TOKEN_PATTERN.findall(normalized_question))
    matches = graph_retriever.match_subjects(question, limit=5)
    for option in _specific_subject_options(matches):
        subject_tokens = _distinctive_subject_tokens(option.title)
        overlap = query_tokens & subject_tokens
        if not overlap:
            continue
        if _has_phrase(normalized_question, _normalize_query_key(option.title)):
            return True
        if len(overlap) >= min(2, len(subject_tokens)):
            return True
        if any(len(token) >= 4 for token in overlap):
            return True
    return False


def _distinctive_subject_tokens(title: str) -> set[str]:
    generic_tokens = {
        "bao",
        "cac",
        "cho",
        "dich",
        "goi",
        "hiem",
        "mo",
        "nhom",
        "pham",
        "san",
        "the",
        "tin",
        "vay",
        "vcb",
        "vietcombank",
    }
    return set(TOKEN_PATTERN.findall(_normalize_query_key(title))) - generic_tokens


def _parse_numbered_options(text: str) -> dict[int, str]:
    options: dict[int, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,2})[\.)]\s+(.+?)(?:\s+\([^()]+\))?\s*$", line)
        if not match:
            continue
        options[int(match.group(1))] = match.group(2).strip()
    return options


def _matched_text_option_title(
    selected_text_key: str,
    option_titles: dict[int, str],
) -> str | None:
    for title in option_titles.values():
        option_key = _choice_label_key(_normalize_query_key(title))
        if selected_text_key == option_key:
            return title
    return None


def _choice_label_key(normalized_text: str) -> str:
    words = normalized_text.split()
    while words and words[-1] in {"nhom", "san", "pham", "dich", "vu", "lua", "chon"}:
        words.pop()
    return " ".join(words)


def _resolve_subject_option(
    title: str,
    *,
    graph_retriever: ProductGraphRetriever,
) -> GraphSubjectOption | None:
    normalized_title = _normalize_query_key(title)
    matches = graph_retriever.match_subjects(title, limit=8)
    for option in matches:
        if _normalize_query_key(option.title) == normalized_title:
            return option
    return matches[0] if matches else None


def _previous_user_question(history: list[ChatMessage], *, before_index: int) -> str | None:
    for message in reversed(history[:before_index]):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return None


def _previous_detail_question(history: list[ChatMessage], *, before_index: int) -> str | None:
    for message in reversed(history[:before_index]):
        if message.role != "user" or not message.content.strip():
            continue
        normalized = _normalize_query_key(message.content)
        if _contains_any(normalized, DETAIL_MARKERS + EXHAUSTIVE_DETAIL_MARKERS):
            return message.content.strip()
    return None


def _latest_specific_user_subject(
    *,
    history: list[ChatMessage],
    graph_retriever: ProductGraphRetriever,
) -> str | None:
    for message in reversed(history[-8:]):
        if message.role != "user" or not message.content.strip():
            continue

        specific_matches = _specific_subject_options(
            graph_retriever.match_subjects(message.content, limit=8)
        )
        if specific_matches:
            return specific_matches[0].title
    return None


def _latest_specific_user_product(
    *,
    history: list[ChatMessage],
    graph_retriever: ProductGraphRetriever,
) -> GraphSubjectOption | None:
    for message in reversed(history[-8:]):
        if message.role != "user" or not message.content.strip():
            continue

        for option in _specific_subject_options(
            graph_retriever.match_subjects(message.content, limit=8)
        ):
            if option.subject_type == "product":
                return option
    return None


def _common_category_title(options: tuple[GraphSubjectOption, ...]) -> str | None:
    titles = {
        option.category_title
        for option in options
        if option.category_title and option.category_title.strip()
    }
    return next(iter(titles)) if len(titles) == 1 else None


def _has_validated_specific_rewrite(
    *,
    original_query: str,
    rewritten_query: str,
    graph_retriever: ProductGraphRetriever,
    original_options: tuple[GraphSubjectOption, ...],
) -> bool:
    if _normalize_query_key(rewritten_query) == _normalize_query_key(original_query):
        return False

    rewritten_subjects = _specific_subject_options(
        graph_retriever.match_subjects(rewritten_query, limit=8)
    )
    if not rewritten_subjects:
        return False

    original_option_keys = {_subject_option_key(option) for option in original_options}
    for subject in rewritten_subjects:
        if _subject_option_key(subject) not in original_option_keys:
            continue
        if subject.subject_type == "product":
            return True
        if _subject_overlap(original_query, subject.title) >= 1:
            return True
    return False


def _specific_subject_options(
    options: tuple[GraphSubjectOption, ...],
) -> tuple[GraphSubjectOption, ...]:
    return tuple(
        option
        for option in options
        if option.subject_type == "product"
        or option.parent_title
        or len(_normalize_query_key(option.title).split()) > 2
    )


def _subject_option_key(option: GraphSubjectOption) -> tuple[str, str]:
    if option.subject_type == "category":
        return (
            option.subject_type,
            "|".join(
                [
                    _normalize_query_key(option.product_type or ""),
                    _normalize_query_key(option.parent_title or ""),
                    _normalize_query_key(option.title),
                ]
            ),
        )
    return (option.subject_type, option.url or _normalize_query_key(option.title))


def _subject_overlap(query: str, subject: str) -> int:
    query_tokens = set(TOKEN_PATTERN.findall(_normalize_query_key(query)))
    subject_tokens = set(TOKEN_PATTERN.findall(_normalize_query_key(subject)))
    generic_tokens = {"bao", "hiem", "mo", "the", "vietcombank", "vcb"}
    return len((query_tokens & subject_tokens) - generic_tokens)


def _append_subject(question: str, subject: str) -> str:
    normalized_question = _normalize_query_key(question)
    normalized_subject = _normalize_query_key(subject)
    if normalized_subject and _has_phrase(normalized_question, normalized_subject):
        return question
    return f"{question} {subject}"


def _is_contextual_follow_up(normalized_query: str) -> bool:
    return any(_has_phrase(normalized_query, marker) for marker in CONTEXTUAL_MARKERS)


def _is_multi_subject_follow_up(normalized_query: str) -> bool:
    return any(
        marker in normalized_query
        for marker in (
            "ca cac goi tren",
            "ca cac san pham tren",
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
            "tat ca goi tren",
            "tat ca san pham tren",
            "toan bo goi tren",
            "toan bo san pham tren",
            "tung goi",
            "tung san pham",
            "vua liet ke",
            "vua neu",
        )
    )


def _latest_product_type_subject(history: list[ChatMessage]) -> str | None:
    for message in reversed(history[-8:]):
        product_type = _infer_product_type(_normalize_query_key(message.content))
        subject = _product_type_subject(product_type)
        if subject:
            return subject
    return None


def _product_type_subject(product_type: str | None) -> str | None:
    return {
        "account": "tài khoản",
        "card": "thẻ",
        "digital_banking": "ngân hàng số",
        "insurance": "bảo hiểm",
        "investment": "đầu tư",
        "loan": "vay",
        "saving": "tiết kiệm",
        "transfer": "chuyển và nhận tiền",
    }.get(product_type or "")


def _is_under_specified_detail_query(normalized_query: str) -> bool:
    if not normalized_query:
        return False
    if _contains_any(normalized_query, LIST_MARKERS):
        return False
    if _contains_any(normalized_query, CONTEXTUAL_MARKERS):
        return True
    has_detail = _contains_any(normalized_query, DETAIL_MARKERS)
    has_product_type = _contains_any(normalized_query, PRODUCT_TYPE_HINTS)
    token_count = len(TOKEN_PATTERN.findall(normalized_query))
    return has_detail and (token_count <= 4 or has_product_type)


def _is_product_type_advisory_query(normalized_query: str) -> bool:
    if _infer_product_type(normalized_query) != "loan":
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


def _is_detail_query(normalized_query: str) -> bool:
    return bool(normalized_query and _contains_any(normalized_query, DETAIL_MARKERS))


def _requests_all_recent_products(normalized_query: str) -> bool:
    if not normalized_query:
        return False
    if _is_multi_subject_follow_up(normalized_query):
        return True
    return any(
        marker in normalized_query
        for marker in (
            "cac lua chon tren",
            "cac muc tren",
            "cac san pham tren",
            "cac goi tren",
            "moi san pham",
            "moi goi",
            "nhung san pham tren",
            "nhung goi tren",
            "tat ca cac san pham",
            "tat ca cac goi",
            "tat ca lua chon",
            "tat ca san pham",
            "tat ca goi",
            "toan bo cac san pham",
            "toan bo cac goi",
            "toan bo san pham",
            "toan bo goi",
            "tung san pham",
            "tung goi",
        )
    )


def _is_under_specified_exhaustive_query(normalized_query: str) -> bool:
    if not normalized_query:
        return False
    has_exhaustive_marker = _contains_any(normalized_query, EXHAUSTIVE_DETAIL_MARKERS) or (
        "toan bo" in normalized_query and "thong tin" in normalized_query
    )
    if not has_exhaustive_marker:
        return False
    return not _contains_any(normalized_query, PRODUCT_TYPE_HINTS)


def _looks_noisy_or_short(normalized_query: str) -> bool:
    tokens = TOKEN_PATTERN.findall(normalized_query)
    if len(tokens) <= 3:
        return True
    long_tokens = [token for token in tokens if len(token) >= 9]
    return any(_has_repeated_character(token) for token in long_tokens)


def _contains_any(normalized_query: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized_query for marker in markers)


def _infer_product_type(normalized_query: str) -> str | None:
    padded_query = f" {normalized_query} "
    for product_type, markers in PRODUCT_TYPE_HINT_MARKERS.items():
        for marker in markers:
            if f" {marker} " in padded_query or marker in normalized_query:
                return product_type
    return None


def _has_phrase(normalized_text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {normalized_text} "


def _has_repeated_character(token: str) -> bool:
    return bool(re.search(r"([a-z0-9])\1{2,}", token))


def _coerce_confidence(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, number))


def _normalize_query_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(TOKEN_PATTERN.findall(no_accents.casefold()))
