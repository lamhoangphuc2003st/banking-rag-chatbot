from apps.api.app.core.config import Settings
from apps.api.app.rag.generation.llm import LLMClient, sanitize_answer_text
from packages.shared.schemas import RetrievedChunk


class FakeStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        if not hasattr(self, "_items"):
            self._items = iter(
                [
                    {"choices": [{"delta": {"content": "Xin "}}]},
                    {"choices": [{"delta": {"content": "chào"}}]},
                ]
            )
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def test_sanitize_answer_text_removes_detail_markdown_links_and_urls() -> None:
    answer = (
        "Vietcombank có hỗ trợ vay mua nhà qua Vay mua nhà dự án.\n"
        "[Chi tiết](https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Vay/Vay-mua-nha-du-an)\n\n"
        "Tham khảo thêm: https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Vay"
    )

    sanitized = sanitize_answer_text(answer)

    assert "https://" not in sanitized
    assert "[Chi tiết]" not in sanitized
    assert sanitized == "Vietcombank có hỗ trợ vay mua nhà qua Vay mua nhà dự án."


def test_sanitize_answer_text_keeps_markdown_link_label_without_url() -> None:
    answer = "1. [Vay mua nhà dự án](https://example.com/vay-mua-nha-du-an)"

    assert sanitize_answer_text(answer) == "1. Vay mua nhà dự án"


def test_sanitize_answer_text_removes_missing_condition_lines() -> None:
    answer = (
        "1. Vay mua nhà dự án\n"
        "- Mức vay: Lên tới 100% giá trị ngôi nhà\n"
        "- Chưa tìm thấy điều kiện cụ thể trong nguồn hiện có."
    )

    sanitized = sanitize_answer_text(answer)

    assert "Chưa tìm thấy điều kiện cụ thể" not in sanitized
    assert "Mức vay" in sanitized


def test_sanitize_answer_text_removes_general_missing_field_disclaimers() -> None:
    answer = (
        "Vietcombank có các gói vay hỗ trợ mua nhà như sau:\n\n"
        "1. Vay mua nhà dự án: mức vay lên tới 100% giá trị ngôi nhà.\n"
        "2. Nhà Mới Thành Đạt: mức cho vay lên tới 70% giá trị căn nhà.\n\n"
        "Nguồn dữ liệu hiện có chưa cung cấp thông tin chi tiết về điều kiện, "
        "đối tượng hay yêu cầu sử dụng cụ thể cho các gói vay này. "
        "Bạn cần cung cấp thêm thông tin hoặc kiểm tra kênh chính thức để biết chi tiết hơn."
    )

    sanitized = sanitize_answer_text(answer)

    assert "Nguồn dữ liệu hiện có chưa cung cấp" not in sanitized
    assert "Bạn cần cung cấp thêm" not in sanitized
    assert "Vay mua nhà dự án" in sanitized
    assert "Nhà Mới Thành Đạt" in sanitized


def test_format_context_does_not_expose_source_urls_to_prompt() -> None:
    client = LLMClient(Settings(_env_file=None))
    context = client._format_context(
        [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                title="Vay mua nhà dự án",
                source_url="https://example.com/vay-mua-nha-du-an",
                text=(
                    "Product: Vay mua nhà dự án\n"
                    "URL: https://example.com/vay-mua-nha-du-an\n"
                    "Mức vay lên tới 100% giá trị ngôi nhà."
                ),
                score=0.9,
                section="product_detail",
                product_type="loan",
                metadata={},
            )
        ]
    )

    assert "https://example.com" not in context
    assert "Mức vay lên tới 100% giá trị ngôi nhà." in context


def test_format_context_hides_internal_missing_detail_marker() -> None:
    client = LLMClient(Settings(_env_file=None))
    context = client._format_context(
        [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                title="Nhà Mới Thành Đạt",
                source_url="https://example.com/nha-moi-thanh-dat",
                text="Composed detail context: no product detail chunks were found.",
                score=0.9,
                section="product_detail",
                product_type="loan",
                metadata={},
            )
        ]
    )

    assert "no product detail chunks were found" not in context


async def test_stream_answer_yields_provider_chunks_incrementally(monkeypatch) -> None:
    async def fake_acompletion(**kwargs):
        assert kwargs["stream"] is True
        return FakeStream()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    client = LLMClient(Settings(_env_file=None, llm_provider="openai", openai_api_key="test"))

    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            title="Vietcombank Vibe Platinum",
            source_url="https://example.com/vibe",
            text="Vietcombank Vibe Platinum có ưu đãi VCB Rewards.",
            score=0.9,
            section="product_detail",
            product_type="card",
        )
    ]

    parts = [
        part
        async for part in client.stream_answer(
            question="Lợi ích của Vietcombank Vibe Platinum",
            history="",
            chunks=chunks,
        )
    ]

    assert parts == ["Xin ", "chào"]
