from __future__ import annotations

from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage, ChatRequest
from apps.api.app.rag.pipeline import RagPipeline
from apps.api.app.rag.query_rewrite import QueryRewriter
from apps.api.app.rag.retrieval.graph import GraphSubjectOption, ProductGraphRetriever


class FakeLLMQueryRewriter(QueryRewriter):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(Settings(_env_file=None, openai_api_key="test-key"))
        self.payload = payload

    async def _call_llm_json(
        self,
        *,
        question: str,
        history: list[ChatMessage],
        recent_subject: str | None,
        candidate_options: tuple[GraphSubjectOption, ...],
    ) -> dict[str, Any]:
        return self.payload


async def test_llm_query_rewrite_corrects_typo_and_preserves_known_subject() -> None:
    rewriter = FakeLLMQueryRewriter(
        {
            "action": "rewrite",
            "rewritten_question": "Điều kiện mở thẻ Vietcombank Chợ Rẫy Connect24",
            "confidence": 0.92,
            "reason": "corrected typo",
        }
    )

    result = await rewriter.rewrite(
        question="Cho Ray Conect24 dieu kien",
        history=[],
        graph_retriever=ProductGraphRetriever(),
    )

    assert result.route == "llm_rewrite"
    assert result.rewritten_query == "Điều kiện mở thẻ Vietcombank Chợ Rẫy Connect24"
    assert not result.needs_clarification


async def test_llm_query_rewrite_can_return_clarification_options() -> None:
    rewriter = FakeLLMQueryRewriter(
        {
            "action": "clarify",
            "rewritten_question": "Phí bảo hiểm Vietcombank",
            "clarification_question": "Bạn muốn hỏi phí của nhóm bảo hiểm nào?",
            "candidate_subjects": ["Bảo hiểm bảo vệ", "Bảo hiểm đầu tư"],
            "confidence": 0.35,
            "reason": "missing product or group",
        }
    )

    result = await rewriter.rewrite(
        question="phi bao hiem sao",
        history=[],
        graph_retriever=ProductGraphRetriever(),
    )

    assert result.route == "llm_clarification"
    assert result.needs_clarification
    assert "Bạn có thể chọn" in (result.clarification_question or "")
    assert result.clarification_options
    assert "Bảo hiểm" not in {option.title for option in result.clarification_options}


async def test_llm_keep_cannot_override_missing_card_subject_clarification() -> None:
    rewriter = FakeLLMQueryRewriter(
        {
            "action": "keep",
            "rewritten_question": "Dieu kien mo the",
            "confidence": 0.9,
            "reason": "generic card question",
        }
    )

    result = await rewriter.rewrite(
        question="Dieu kien mo the",
        history=[],
        graph_retriever=ProductGraphRetriever(),
    )

    assert result.route == "local_clarification"
    assert result.needs_clarification
    assert any("Thẻ" in option.title for option in result.clarification_options)


async def test_llm_cannot_override_resolved_text_clarification_choice() -> None:
    rewriter = FakeLLMQueryRewriter(
        {
            "action": "clarify",
            "rewritten_question": "Thẻ thanh toán là gì?",
            "candidate_subjects": ["Thẻ thanh toán"],
            "confidence": 0.9,
            "reason": "would override local choice",
        }
    )

    result = await rewriter.rewrite(
        question="Thẻ thanh toán (nhóm)",
        history=[
            ChatMessage(role="user", content="Dieu kien mo the"),
            ChatMessage(
                role="assistant",
                content=(
                    "Ban muon hoi nhom nao?\n"
                    "1. The tin dung (nhom)\n"
                    "2. The thanh toan (nhom)\n"
                    "3. Dich vu the (nhom)"
                ),
            ),
        ],
        graph_retriever=ProductGraphRetriever(),
    )

    assert result.route == "local_choice_clarification"
    assert result.needs_clarification
    assert "Vietcombank Connect24" in (result.clarification_question or "")


async def test_pipeline_asks_clarification_for_low_information_query() -> None:
    pipeline = RagPipeline(Settings(_env_file=None, rag_cache_backend="memory"))

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content="Phi sao")])
    )

    assert response.metadata["clarification_required"] is True
    assert response.metadata["query_rewrite_route"] == "local_clarification"
    assert not response.sources


async def test_pipeline_asks_clarification_for_generic_card_condition_query() -> None:
    pipeline = RagPipeline(Settings(_env_file=None, rag_cache_backend="memory"))

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content="Dieu kien mo the")])
    )

    assert response.metadata["clarification_required"] is True
    assert "Bạn có thể chọn" in response.answer
    assert any(
        option["title"] in {"Thẻ tín dụng", "Thẻ thanh toán", "Trả góp"}
        for option in response.metadata["clarification_options"]
    )


async def test_explicit_product_query_after_numbered_history_does_not_select_all_options() -> None:
    rewriter = QueryRewriter(Settings(_env_file=None, llm_provider="local"))

    result = await rewriter.rewrite(
        question="Cho tôi biết toàn bộ thông tin về thẻ Vietcombank MeGO",
        history=[
            ChatMessage(role="user", content="Nếu tôi cần mua nhà thì VCB có hỗ trợ cho vay không?"),
            ChatMessage(
                role="assistant",
                content=(
                    "Vietcombank có các gói vay hỗ trợ mua nhà như sau:\n"
                    "1. Vay mua nhà dự án\n"
                    "2. Vay mua nhà ở, đất ở\n"
                    "3. Vay xây sửa nhà ở\n"
                    "4. Nhà Mới Thành Đạt"
                ),
            ),
        ],
        graph_retriever=ProductGraphRetriever(),
    )

    assert result.route != "local_all_options_choice"
    assert result.rewritten_query == "Cho tôi biết toàn bộ thông tin về thẻ Vietcombank MeGO"
    assert "Vay mua nhà" not in result.rewritten_query


async def test_all_products_detail_follow_up_after_numbered_history_is_allowed() -> None:
    rewriter = QueryRewriter(Settings(_env_file=None, llm_provider="local"))

    result = await rewriter.rewrite(
        question="Lợi ích và điều kiện của các sản phẩm trên là gì",
        history=[
            ChatMessage(role="user", content="VCB hiện có các sản phẩm bảo hiểm nào"),
            ChatMessage(
                role="assistant",
                content=(
                    "Vietcombank hiện có các sản phẩm bảo hiểm sau:\n"
                    "1. FWD Con vươn xa 2.0\n"
                    "2. FWD Vững ước mơ\n"
                    "3. FWD Cả nhà vui khỏe"
                ),
            ),
        ],
        graph_retriever=ProductGraphRetriever(),
    )

    assert result.route == "local_all_options_choice"
    assert not result.needs_clarification
    assert "FWD Con vươn xa 2.0" in result.rewritten_query
    assert "FWD Vững ước mơ" in result.rewritten_query


async def test_current_explicit_product_is_not_augmented_with_previous_subject() -> None:
    rewriter = QueryRewriter(Settings(_env_file=None, llm_provider="local"))

    result = await rewriter.rewrite(
        question="Lợi ích và điều kiện mở thẻ của thẻ Vietcombank Mastercard® Debit",
        history=[
            ChatMessage(
                role="user",
                content="Lợi ích và điều kiện mở thẻ của thẻ Vietcombank Vibe Platinum",
            ),
            ChatMessage(
                role="assistant",
                content="Vietcombank Vibe Platinum có các ưu đãi và điều kiện mở thẻ sau.",
            ),
        ],
        graph_retriever=ProductGraphRetriever(),
    )

    assert result.route == "local_exact_subject"
    assert result.rewritten_query == "Lợi ích và điều kiện mở thẻ của thẻ Vietcombank Mastercard® Debit"
    assert "Vibe Platinum" not in result.rewritten_query


async def test_remittance_benefit_query_matches_category_instead_of_clarifying() -> None:
    rewriter = QueryRewriter(Settings(_env_file=None, llm_provider="local"))

    result = await rewriter.rewrite(
        question="nhan tien kieu hoi tai Vietcombank thi khac gi voi nhung cho khac toi se duoc ich loi gi",
        history=[],
        graph_retriever=ProductGraphRetriever(),
    )

    assert result.route == "local_exact_subject"
    assert not result.needs_clarification
