from __future__ import annotations

import json
from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage, ChatRequest
from apps.api.app.rag.pipeline import RagPipeline, _answer_question_for_plan
from apps.api.app.rag.planner import QueryPlan, QueryPlanner
from apps.api.app.rag.query_rewrite import QueryRewriteResult
from apps.api.app.rag.retrieval.graph import GraphRetrievalResult, ProductGraphRetriever
from packages.shared.schemas import RetrievedChunk


def _test_settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="local",
        rag_cache_backend="memory",
        openai_api_key=None,
        litellm_api_key=None,
    )


class EmptyRetriever:
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 12,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        _ = query, top_k, filters
        return []

    async def scroll_by_filter(
        self,
        *,
        filters: dict[str, Any],
        limit: int = 1000,
    ) -> list[Any]:
        _ = filters, limit
        return []


class ClarifyingPlanner(QueryPlanner):
    async def plan(self, **kwargs: Any) -> QueryPlan:
        _ = kwargs
        return QueryPlan(
            intent="direct_answer",
            route="llm_planner_clarification",
            reason="test_planner_requested_clarification",
            needs_clarification=True,
            clarification_question="Bạn muốn hỏi về nhóm sản phẩm nào?",
            confidence=0.8,
        )


class SingleChunkRetriever(EmptyRetriever):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 12,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        _ = query, top_k, filters
        return [
            RetrievedChunk(
                chunk_id="faq-otp-methods",
                document_id="faq-digibank",
                title="Phân biệt các phương thức nhận OTP?",
                source_url="https://www.vietcombank.com.vn/faq",
                text=(
                    "SMS OTP là phương thức xác thực trong đó mã OTP được gửi qua tin nhắn. "
                    "VCB Smart OTP là phương thức xác thực trong đó mã OTP được tạo ra bởi ứng dụng."
                ),
                score=0.95,
                section="faq",
                product_type="digital_banking",
            )
        ]


class CookiesFaqRetriever(EmptyRetriever):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 12,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        _ = query, top_k, filters
        return [
            RetrievedChunk(
                chunk_id="faq-cookies",
                document_id="faq-digibank",
                title="Tôi có nhất thiết phải sử dụng Cookies cho trình duyệt hay không?",
                source_url="https://www.vietcombank.com.vn/vi-VN/KHCN/Lien-he-va-Ho-tro/Danh-sach-cau-hoi-theo-chu-de-Ngan-hang-so",
                text=(
                    "Quý khách phải sử dụng Cookies để có thể duy trì phiên giao dịch "
                    "sau khi đăng nhập vào chương trình."
                ),
                score=0.95,
                section="faq",
                product_type="digital_banking",
            )
        ]


class IrrelevantVietcombankRetriever(EmptyRetriever):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 12,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        _ = query, top_k, filters
        return [
            RetrievedChunk(
                chunk_id="saving-product",
                document_id="saving-doc",
                title="Tiết kiệm tự động",
                source_url="https://www.vietcombank.com.vn/example",
                text="Thông tin sản phẩm tiết kiệm của Vietcombank.",
                score=0.8,
                section="product_detail",
                product_type="saving",
            )
        ]


class FakeLLM:
    async def generate_answer(self, **_: Any) -> str:
        return "stub answer"

    async def stream_answer(self, **_: Any) -> Any:
        yield "stub answer"


class RaisingLLM:
    """Simulates an LLM provider failure (quota/rate-limit/timeout/outage)."""

    async def stream_answer(self, **_: Any) -> Any:
        raise RuntimeError("simulated LLM provider outage")
        yield ""  # pragma: no cover - unreachable; makes this an async generator


class StubPlanner(QueryPlanner):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(_test_settings())
        self.payload = payload

    def _should_call_llm(self) -> bool:
        return True

    async def _call_llm_json(self, **_: Any) -> dict[str, Any]:
        return self.payload


async def test_llm_planner_selects_query_composition_for_plural_follow_up() -> None:
    graph_retriever = ProductGraphRetriever()
    graph_result = graph_retriever.retrieve(
        "VCB hien co cac goi tiet kiem nao",
        history=[],
        top_k=12,
    )
    planner = StubPlanner(
        {
            "action": "answer",
            "intent": "exhaustive_product_details",
            "product_type": "saving",
            "requested_fields": ["overview"],
            "compose_product_dossiers": True,
            "max_subjects": 12,
            "confidence": 0.92,
            "reason": "resolved_plural_follow_up_from_catalog_history",
        }
    )

    plan = await planner.plan(
        question="Cho toi biet thong tin chi tiet cua cac goi do luon",
        history=[
            ChatMessage(role="user", content="VCB hien co cac goi tiet kiem nao"),
            ChatMessage(
                role="assistant",
                content=(
                    "Vietcombank hien co cac goi tiet kiem sau: "
                    "1. Tien gui An Vui 2. Tiet kiem tu dong"
                ),
            ),
        ],
        graph_result=graph_result,
        graph_retriever=graph_retriever,
    )

    assert plan.route == "llm_planner"
    assert plan.intent == "exhaustive_product_details"
    assert plan.product_type == "saving"
    assert plan.compose_product_dossiers is True
    assert plan.context_top_k >= 18


async def test_llm_planner_cannot_disable_catalog_field_dossiers() -> None:
    graph_retriever = ProductGraphRetriever()
    question = "VCB có cho vay phục vụ nhu cầu bất động sản không? Và điều kiện vay là gì?"
    graph_result = graph_retriever.retrieve(question, history=[], top_k=12)
    planner = StubPlanner(
        {
            "action": "answer",
            "intent": "exhaustive_product_details",
            "product_type": "loan",
            "requested_fields": ["condition"],
            "compose_product_dossiers": False,
            "max_subjects": 1,
            "confidence": 0.9,
            "reason": "llm_attempted_under_expansion",
        }
    )

    plan = await planner.plan(
        question=question,
        history=[],
        graph_result=graph_result,
        graph_retriever=graph_retriever,
    )

    assert plan.route == "llm_planner"
    assert plan.intent == "exhaustive_product_details"
    assert plan.requested_field == "condition"
    assert plan.compose_product_dossiers is True
    assert plan.max_subjects >= 4


async def test_llm_planner_low_confidence_falls_back_to_local_plan() -> None:
    graph_retriever = ProductGraphRetriever()
    planner = StubPlanner(
        {
            "action": "answer",
            "intent": "exhaustive_product_details",
            "product_type": "saving",
            "confidence": 0.2,
            "reason": "not_sure",
        }
    )

    plan = await planner.plan(
        question="Lai suat the nao",
        history=[],
        graph_result=GraphRetrievalResult(chunks=[], route="default"),
        graph_retriever=graph_retriever,
    )

    assert plan.route == "planner_direct"
    assert plan.intent == "direct_answer"


async def test_local_planner_routes_student_loan_advisory_to_catalog_dossiers() -> None:
    graph_retriever = ProductGraphRetriever()
    planner = QueryPlanner(_test_settings())
    question = (
        "Toi la sinh vien, khong du tien dong hoc phi va tien phong tro. "
        "VCB co goi vay nao phu hop danh cho sinh vien khong? "
        "Lai suat, muc cho vay va thoi han nhu the nao?"
    )

    plan = await planner.plan(
        question=question,
        history=[],
        graph_result=GraphRetrievalResult(chunks=[], route="default"),
        graph_retriever=graph_retriever,
    )

    assert plan.route == "planner_product_type_advisory"
    assert plan.intent == "exhaustive_product_details"
    assert plan.product_type == "loan"
    assert plan.compose_product_dossiers is True
    assert plan.requested_field is None
    assert plan.needs_clarification is False


async def test_llm_planner_cannot_clarify_student_loan_advisory_scope() -> None:
    graph_retriever = ProductGraphRetriever()
    planner = StubPlanner(
        {
            "action": "clarify",
            "confidence": 0.9,
            "reason": "model_wants_specific_loan_group",
        }
    )
    question = (
        "Toi la sinh vien, khong du tien dong hoc phi va tien phong tro. "
        "VCB co goi vay nao phu hop danh cho sinh vien khong? "
        "Lai suat, muc cho vay va thoi han nhu the nao?"
    )

    plan = await planner.plan(
        question=question,
        history=[],
        graph_result=GraphRetrievalResult(chunks=[], route="default"),
        graph_retriever=graph_retriever,
    )

    assert plan.route == "planner_product_type_advisory"
    assert plan.needs_clarification is False
    assert plan.product_type == "loan"


async def test_llm_planner_cannot_narrow_student_loan_advisory_scope() -> None:
    graph_retriever = ProductGraphRetriever()
    planner = StubPlanner(
        {
            "action": "answer",
            "intent": "exhaustive_product_details",
            "product_type": "loan",
            "max_subjects": 1,
            "compose_product_dossiers": False,
            "confidence": 0.9,
            "reason": "model_picked_one_group",
        }
    )
    question = (
        "Toi la sinh vien, khong du tien dong hoc phi va tien phong tro. "
        "VCB co goi vay nao phu hop danh cho sinh vien khong? "
        "Lai suat, muc cho vay va thoi han nhu the nao?"
    )

    plan = await planner.plan(
        question=question,
        history=[],
        graph_result=GraphRetrievalResult(chunks=[], route="default"),
        graph_retriever=graph_retriever,
    )

    assert plan.route == "planner_product_type_advisory"
    assert plan.max_subjects == 12
    assert plan.compose_product_dossiers is True
    assert plan.product_type == "loan"


async def test_llm_planner_ignores_clarification_without_graph_options() -> None:
    graph_retriever = ProductGraphRetriever()
    planner = StubPlanner(
        {
            "action": "clarify",
            "confidence": 0.9,
            "reason": "model_thinks_scope_is_missing",
        }
    )

    plan = await planner.plan(
        question="Tôi có nhất thiết phải sử dụng Cookies cho trình duyệt hay không?",
        history=[],
        graph_result=GraphRetrievalResult(chunks=[], route="default"),
        graph_retriever=graph_retriever,
    )

    assert plan.route == "planner_direct"
    assert plan.needs_clarification is False


async def test_pipeline_seeds_catalog_when_llm_plan_has_product_type_without_graph_catalog() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    retrieval = await pipeline._retrieve_for_plan(
        "Cho toi biet thong tin chi tiet cua cac goi do luon",
        GraphRetrievalResult(chunks=[], route="default"),
        QueryPlan(
            intent="exhaustive_product_details",
            route="llm_planner",
            reason="llm_selected_query_composition",
            product_type="saving",
            max_subjects=12,
            context_top_k=36,
            compose_product_dossiers=True,
        ),
    )

    assert "planner_catalog_seed" in retrieval.route
    assert retrieval.expanded_product_count >= 10
    assert retrieval.expanded_chunk_count > retrieval.expanded_product_count


async def test_pipeline_answers_student_loan_advisory_without_clarification() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Toi la sinh vien, khong du tien dong hoc phi va tien phong tro. "
                        "VCB co goi vay nao phu hop danh cho sinh vien khong? "
                        "Lai suat, muc cho vay va thoi han nhu the nao?"
                    ),
                )
            ]
        )
    )

    assert response.metadata.get("clarification_required") is not True
    assert response.metadata["retrieval_plan_route"] == "planner_product_type_advisory"
    assert response.metadata["retrieval_plan_product_type"] == "loan"
    assert response.metadata["retrieval_plan_composes_product_dossiers"] is True
    assert response.metadata["expanded_product_count"] >= 10
    assert response.answer == "stub answer"
    assert response.sources


async def test_pipeline_expands_exhaustive_catalog_question_to_product_details() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Cho tôi biết toàn bộ thông tin về các gói gửi tiết kiệm hiện có",
                )
            ]
        )
    )

    assert response.metadata["retrieval_plan_intent"] == "exhaustive_product_details"
    assert response.metadata["retrieval_plan_route"] == "planner_exhaustive_product_details"
    assert response.metadata["expanded_product_count"] >= 10
    assert response.metadata["expanded_chunk_count"] > response.metadata["expanded_product_count"]
    assert response.metadata["reranked_count"] > 6
    assert response.metadata["retrieval_route"].startswith("planner_exhaustive_product_details+")
    assert any(source.section == "product_detail" for source in response.sources)
    assert response.metadata["retrieval_plan_composes_product_dossiers"] is True


async def test_pipeline_expands_catalog_question_with_detail_field_in_same_turn() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "VCB có cho vay phục vụ nhu cầu bất động sản không? "
                        "Và điều kiện vay là gì?"
                    ),
                )
            ]
        )
    )

    assert response.metadata["query_rewrite_route"] == "local_exact_subject"
    assert response.metadata["retrieval_plan_intent"] == "exhaustive_product_details"
    assert response.metadata["retrieval_plan_route"] == "planner_catalog_field_details"
    assert response.metadata["retrieval_plan_reason"] == "catalog_subject_requests_field_details"
    assert response.metadata["retrieval_plan_requested_field"] == "condition"
    assert response.metadata["retrieval_plan_composes_product_dossiers"] is True
    assert response.metadata["expanded_product_count"] == 4
    assert response.metadata["expanded_chunk_count"] >= 12
    assert {
        "Vay xây sửa nhà ở",
        "Nhà Mới Thành Đạt",
        "Vay mua nhà dự án",
        "Vay mua nhà ở, đất ở",
    } <= {source.title for source in response.sources}
    assert "Công dân Việt Nam từ 20 tuổi" in response.answer
    assert "Có nhu cầu vay vốn mua nhà ở" in response.answer
    assert "Có thu nhập ổn định, đủ khả năng trả nợ" in response.answer
    assert "Mức vay" not in response.answer


async def test_pipeline_asks_clarification_for_exhaustive_question_without_subject() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Cho tôi biết toàn bộ thông tin về các gói hiện có",
                )
            ]
        )
    )

    assert response.metadata["clarification_required"] is True
    assert response.metadata["query_rewrite_route"] == "local_clarification"
    assert response.metadata["query_rewrite_reason"] == "under_specified_exhaustive_query"
    assert not response.sources


async def test_pipeline_routes_account_recovery_question_to_retrieval() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Quên mật khẩu rút tiền, làm sao để lấy lại",
                )
            ]
        )
    )

    assert response.metadata["retrieval_plan_route"] == "planner_security_account_recovery"
    assert response.metadata["retrieval_route"].endswith("default")
    assert "Bạn vui lòng nêu rõ" not in response.answer
    assert "Không gửi mật khẩu" not in response.answer
    assert response.refusal is False


async def test_pipeline_routes_otp_comparison_faq_to_retrieval() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = SingleChunkRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Phân biệt các phương thức nhận OTP?",
                )
            ]
        )
    )

    assert response.metadata["query_rewrite_route"] == "local_security_public_info"
    assert response.metadata["retrieval_plan_route"] == "planner_security_public_info"
    assert response.metadata["retrieved_count"] == 1
    assert response.answer == "stub answer"
    assert response.refusal is False


async def test_pipeline_degrades_gracefully_when_generation_fails() -> None:
    # Regression for the live-deploy finding: an LLM provider error must degrade to
    # a graceful fallback (HTTP 200), never propagate as an unhandled HTTP 500.
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = SingleChunkRetriever()  # type: ignore[assignment]
    pipeline.llm = RaisingLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content="Phân biệt các phương thức nhận OTP?")
            ]
        )
    )

    assert response.refusal is False
    assert response.metadata["generation_failed"] is True
    assert "thử lại sau" in response.answer
    # A fallback message is not grounded in the retrieved chunks, so cite nothing.
    assert response.sources == []


async def test_pipeline_allows_indexed_faq_without_bank_keyword() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = CookiesFaqRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Tôi có nhất thiết phải sử dụng Cookies cho trình duyệt hay không?",
                )
            ]
        )
    )

    assert response.metadata["retrieved_count"] == 1
    assert response.answer == "stub answer"
    assert response.refusal is False


async def test_pipeline_refuses_out_of_scope_when_no_evidence() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Dự báo thời tiết hôm nay",
                )
            ]
        )
    )

    assert response.refusal is True
    assert response.metadata["guardrail_reason"] == "out_of_scope"


async def test_pipeline_refuses_out_of_scope_with_only_weak_vietcombank_overlap() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = IrrelevantVietcombankRetriever()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Dự báo thời tiết hôm nay",
                )
            ]
        )
    )

    assert response.refusal is True
    assert response.metadata["guardrail_reason"] == "out_of_scope"


async def test_pipeline_routes_public_password_policy_question_to_retrieval() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Mật khẩu sử dụng dịch vụ VCB Digibank nên được đặt như thế nào "
                        "và sau bao lâu tôi phải đổi mật khẩu?"
                    ),
                )
            ]
        )
    )

    assert response.metadata["retrieval_plan_route"] == "planner_security_public_info"
    assert response.metadata["retrieval_route"].endswith("default")
    assert response.metadata["retrieved_count"] == 0
    assert "Không gửi mật khẩu" not in response.answer
    assert response.refusal is False


async def test_pipeline_answers_exact_faq_from_local_chunks_before_planner(tmp_path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    title = (
        "Trong thời gian xây dựng, chưa có thu nhập từ tiền kinh doanh cơ sở lưu trú, "
        "Ngân hàng có cho phép tôi được ân hạn gốc vay?"
    )
    payload = {
        "chunk_id": "faq-loan-grace-period",
        "document_id": "faq-loan",
        "title": title,
        "source_url": "https://www.vietcombank.com.vn/faq#faq=loan-grace-period",
        "text": (
            "[FAQ] Topic: Vay Question: "
            f"{title} Answer: Khách hàng được ân hạn trả nợ gốc tối đa 24 tháng."
        ),
        "product_type": "loan",
        "section": "faq",
        "chunk_index": 0,
        "metadata": {"document_type": "faq"},
    }
    with (chunks_dir / "vietcombank_faq_chunks.jsonl").open("w", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    settings = _test_settings().model_copy(update={"rag_data_root": str(tmp_path)})
    pipeline = RagPipeline(settings)
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.query_planner = ClarifyingPlanner(settings)

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content=title)])
    )

    assert response.metadata["retrieval_plan_route"] == "planner_exact_faq"
    assert response.metadata["retrieval_route"].startswith("exact_faq+")
    assert "ân hạn trả nợ gốc tối đa 24 tháng" in response.answer
    assert [source.chunk_id for source in response.sources] == ["faq-loan-grace-period"]


async def test_pipeline_prefers_exact_faq_over_catalog_filter(tmp_path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    title = (
        "Trong thời gian xây dựng, chưa có thu nhập từ tiền kinh doanh cơ sở lưu trú, "
        "Ngân hàng có cho phép tôi được ân hạn gốc vay?"
    )
    payload = {
        "chunk_id": "faq-loan-grace-period",
        "document_id": "faq-loan",
        "title": title,
        "source_url": "https://www.vietcombank.com.vn/faq#faq=loan-grace-period",
        "text": (
            "[FAQ] Topic: Vay Question: "
            f"{title} Answer: Khách hàng được ân hạn trả nợ gốc tối đa 24 tháng."
        ),
        "product_type": "loan",
        "section": "faq",
        "chunk_index": 0,
        "metadata": {"document_type": "faq"},
    }
    with (chunks_dir / "vietcombank_faq_chunks.jsonl").open("w", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    settings = _test_settings().model_copy(update={"rag_data_root": str(tmp_path)})
    pipeline = RagPipeline(settings)
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content=title)])
    )

    assert response.metadata["retrieval_plan_route"] == "planner_exact_faq"
    assert response.metadata["retrieval_route"].startswith("exact_faq+")
    assert "ân hạn trả nợ gốc tối đa 24 tháng" in response.answer


async def test_pipeline_does_not_clarify_generic_password_policy_question() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Sau bao lâu thì phải đổi mật khẩu một lần và mật khẩu nên đặt như thế nào "
                        "để an toàn"
                    ),
                )
            ]
        )
    )

    assert response.metadata["retrieval_plan_route"] == "planner_security_public_info"
    assert "Bạn vui lòng nêu rõ" not in response.answer
    assert "Không gửi mật khẩu" not in response.answer
    assert response.refusal is False


async def test_pipeline_expands_multi_product_follow_up_from_catalog_context() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="cho tôi biết thông tin về các gói vay hiện có",
                ),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Vietcombank hiện có các gói vay chính sau:\n"
                        "1. Vay tín chấp theo lương\n"
                        "2. Vay mua ô tô\n"
                        "3. Vay mua nhà ở, đất ở"
                    ),
                ),
                ChatMessage(
                    role="user",
                    content="Cho tôi biết điều kiện vay của từng gói vay trên",
                ),
            ]
        )
    )

    assert response.metadata["query_rewrite_route"] == "local_multi_subject_context"
    assert response.metadata["retrieval_plan_intent"] == "exhaustive_product_details"
    assert response.metadata["retrieval_plan_requested_field"] == "condition"
    assert response.metadata["expanded_product_count"] >= 10
    assert response.metadata["expanded_chunk_count"] > response.metadata["expanded_product_count"]
    assert response.metadata["reranked_count"] > 6
    assert any(source.section == "product_detail" for source in response.sources)


async def test_pipeline_resolves_all_products_after_product_clarification() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content="Dieu kien cho vay cua VCB"),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Ban muon hoi nhom nao?\n"
                        "1. Vay tieu dung (nhom)\n"
                        "2. Vay mua o to (nhom)\n"
                        "3. Vay san xuat kinh doanh (nhom)\n"
                        "4. Vay nhu cau bat dong san (nhom)"
                    ),
                ),
                ChatMessage(role="user", content="Vay nhu cau bat dong san"),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Ban da chon Vay nhu cau bat dong san. Ban muon hoi san pham nao?\n"
                        "1. Vay xay sua nha o (san pham)\n"
                        "2. Nha Moi Thanh Dat (san pham)\n"
                        "3. Vay mua nha du an (san pham)\n"
                        "4. Vay mua nha o, dat o (san pham)"
                    ),
                ),
                ChatMessage(role="user", content="Ca 4 san pham tren"),
            ]
        )
    )

    assert response.metadata["query_rewrite_route"] == "local_all_options_choice"
    assert response.metadata["query_rewrite_reason"] == (
        "resolved_all_clarification_options_with_previous_intent"
    )
    assert response.metadata["retrieval_plan_intent"] == "exhaustive_product_details"
    assert response.metadata["retrieval_plan_requested_field"] == "condition"
    assert response.metadata["expanded_product_count"] == 4
    assert response.metadata["expanded_chunk_count"] >= 12
    assert {
        "Vay xây sửa nhà ở",
        "Nhà Mới Thành Đạt",
        "Vay mua nhà dự án",
        "Vay mua nhà ở, đất ở",
    } <= {source.title for source in response.sources}


async def test_pipeline_keeps_new_explicit_product_after_numbered_history() -> None:
    pipeline = RagPipeline(_test_settings())
    pipeline.retriever = EmptyRetriever()  # type: ignore[assignment]
    pipeline.llm = FakeLLM()  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Nếu tôi cần mua nhà thì VCB có hỗ trợ cho vay không?",
                ),
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
                ChatMessage(
                    role="user",
                    content="Cho tôi biết toàn bộ thông tin về thẻ Vietcombank MeGO",
                ),
            ]
        )
    )

    assert response.metadata["query_rewrite_route"] != "local_all_options_choice"
    assert "Vietcombank MeGO" in response.metadata["retrieval_query"]
    assert "Vay mua nhà" not in response.metadata["retrieval_query"]


def test_condition_plan_prompt_does_not_allow_documents_as_substitute() -> None:
    question = _answer_question_for_plan(
        "Cho tôi biết điều kiện vay của từng gói vay trên",
        "Cho tôi biết điều kiện vay của từng gói vay trên vay",
        QueryRewriteResult(
            original_query="Cho tôi biết điều kiện vay của từng gói vay trên",
            rewritten_query="Cho tôi biết điều kiện vay của từng gói vay trên vay",
        ),
        QueryPlan(
            intent="exhaustive_product_details",
            route="planner_exhaustive_product_details",
            reason="catalog_question_requests_full_details",
            requested_field="condition",
        ),
    )

    assert "Không dùng mục Hồ sơ" in question
    assert "Chưa tìm thấy điều kiện cụ thể" not in question
    assert "bắt buộc nêu riêng đoạn đó" in question
    assert "không thay bằng Mức vay" in question


def test_condition_field_prioritizes_customer_condition_chunks() -> None:
    graph_retriever = ProductGraphRetriever()
    graph_result = graph_retriever.retrieve(
        "Cho tôi biết điều kiện vay của từng gói vay trên vay",
        history=[],
        top_k=12,
    )

    chunks = graph_retriever.product_detail_chunks_for_catalog_chunks(
        graph_result.chunks,
        query="Cho tôi biết điều kiện vay của từng gói vay trên vay",
        max_products=12,
        top_k_per_product=1,
        requested_field="condition",
    )

    auto_loan_detail = next(
        chunk
        for chunk in chunks
        if chunk.title == "Vay mua ô tô"
        and chunk.section == "product_detail"
        and not chunk.chunk_id.startswith("graph:product:")
    )
    assert "Đối tượng khách hàng" in auto_loan_detail.text


def test_product_dossier_preserves_saving_detail_sections() -> None:
    graph_retriever = ProductGraphRetriever()
    graph_result = graph_retriever.retrieve(
        "Cho tôi biết thông tin của toàn bộ các gói tiết kiệm hiện có",
        history=[],
        top_k=12,
    )

    dossiers = graph_retriever.product_dossier_chunks_for_catalog_chunks(
        graph_result.chunks,
        query="Cho tôi biết thông tin của toàn bộ các gói tiết kiệm hiện có",
        max_products=12,
        max_chars_per_product=2800,
    )

    automatic_saving = next(chunk for chunk in dossiers if chunk.title == "Tiết kiệm tự động")
    assert automatic_saving.metadata["retrieval_source"] == "query_composition"
    assert automatic_saving.metadata["source_chunk_count"] >= 2
    assert "Tiền gửi tối thiểu 01 triệu VND hoặc 100 USD" in automatic_saving.text
    assert "Kỳ hạn Tối đa 60 tháng" in automatic_saving.text
    assert "Đối tượng tham gia" in automatic_saving.text
