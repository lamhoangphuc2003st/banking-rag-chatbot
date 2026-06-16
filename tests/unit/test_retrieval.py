import json
from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage
from apps.api.app.rag.pipeline import (
    _catalog_retrieval_filter,
    _exact_faq_chunks_for_query,
    _expand_exact_faq_context,
    _filter_catalog_chunks,
    _preferred_catalog_category_keys,
    _resolve_retrieval_query,
)
from apps.api.app.rag.retrieval.hybrid import HybridRetriever, _lexical_match_score, _tokenize
from packages.shared.schemas import RetrievedChunk


class CountingRetriever(HybridRetriever):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                _env_file=None,
                rag_cache_enabled=True,
                rag_cache_backend="memory",
                rag_cache_ttl_seconds=60,
                rag_cache_max_entries=16,
            )
        )
        self.vector_calls = 0
        self.lexical_calls = 0

    async def _vector_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        _ = query, top_k, filters
        self.vector_calls += 1
        return [
            _chunk(
                chunk_id="vector-hit",
                title="Vietcombank Mastercard Debit",
                text="Hoan tien chi tieu 0,4%",
                score=0.8,
            )
        ]

    async def _lexical_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        _ = query, top_k, filters
        self.lexical_calls += 1
        return []


async def test_hybrid_retriever_caches_repeated_retrieval() -> None:
    retriever = CountingRetriever()

    first = await retriever.retrieve("Loi ich Mastercard Debit", top_k=5)
    second = await retriever.retrieve("Loi ich Mastercard Debit", top_k=5)

    assert [chunk.chunk_id for chunk in first] == ["vector-hit"]
    assert [chunk.chunk_id for chunk in second] == ["vector-hit"]
    assert retriever.vector_calls == 1
    assert retriever.lexical_calls == 1


def test_retriever_reranks_title_match_over_higher_vector_score() -> None:
    retriever = HybridRetriever(Settings(_env_file=None, rag_cache_backend="memory"))
    unrelated = _chunk(
        chunk_id="unrelated",
        title="Sản phẩm bảo hiểm liên kết chung FWD Bảo vệ gia tăng",
        text="Thông tin về bảo hiểm và quyền lợi bảo vệ.",
        score=0.72,
    )
    title_match = _chunk(
        chunk_id="title-match",
        title="Vay tín chấp theo lương",
        text="Hồ sơ vay tín chấp theo lương tại Vietcombank.",
        score=0.62,
    )

    results = retriever._merge_hits(
        "Vay tín chấp theo lương Vietcombank cần hồ sơ gì?",
        [unrelated, title_match],
        [],
    )

    assert [result.chunk_id for result in results] == ["title-match", "unrelated"]


def test_lexical_score_prefers_exact_product_title() -> None:
    query_tokens = _tokenize("Điều kiện vay mua ô tô là gì?")

    auto_loan = _chunk(
        chunk_id="auto",
        title="Vay mua ô tô",
        text="Điều kiện vay mua ô tô tại Vietcombank.",
        score=0.0,
    )
    home_loan = _chunk(
        chunk_id="home",
        title="Vay mua nhà ở, đất ở",
        text="Điều kiện vay mua nhà ở, đất ở tại Vietcombank.",
        score=0.0,
    )

    assert _lexical_match_score(query_tokens, auto_loan) > _lexical_match_score(
        query_tokens,
        home_loan,
    )


def test_catalog_query_filter_targets_insurance_catalog() -> None:
    filters = _catalog_retrieval_filter("Vietcombank có các gói bảo hiểm nào?")

    assert filters == {
        "must": [
            {"key": "section", "match": {"value": "product_catalog"}},
            {"key": "product_type", "match": {"value": "insurance"}},
        ]
    }


def test_catalog_availability_query_targets_transfer_catalog() -> None:
    filters = _catalog_retrieval_filter("VCB co cho chuyen va nhan tien tu nuoc ngoai khong?")

    assert filters == {
        "must": [
            {"key": "section", "match": {"value": "product_catalog"}},
            {"key": "product_type", "match": {"value": "transfer"}},
        ]
    }


def test_catalog_filter_does_not_hijack_long_refund_faq() -> None:
    filters = _catalog_retrieval_filter(
        "Khách hàng rút tiền tại ATM của Vietcombank, giao dịch không thành công "
        "nhưng tài khoản bị trừ tiền thì sau bao lâu được hoàn trả tiền. "
        "Nếu khách hàng không làm tra soát thì có được hoàn trả tiền không?"
    )

    assert filters is None


def test_tokenize_normalizes_vietnamese_d() -> None:
    tokens = _tokenize("đăng ký được hoàn trả")

    assert "dang" in tokens
    assert "ang" not in tokens
    assert "uoc" not in tokens


def test_exact_faq_context_is_expanded_in_source_order(tmp_path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    source_url = "https://www.vietcombank.com.vn/faq-atm-refund"
    title = "ATM refund failed but account debited?"
    payloads = [
        {
            "chunk_id": "faq-0",
            "document_id": "faq-atm",
            "title": title,
            "source_url": source_url,
            "text": "Answer part 1",
            "content_hash": "hash",
            "language": "vi",
            "product_type": "transfer",
            "section": "faq",
            "chunk_index": 0,
            "metadata": {"document_type": "faq"},
        },
        {
            "chunk_id": "faq-1",
            "document_id": "faq-atm",
            "title": title,
            "source_url": source_url,
            "text": "Answer part 2",
            "content_hash": "hash",
            "language": "vi",
            "product_type": "transfer",
            "section": "faq",
            "chunk_index": 1,
            "metadata": {"document_type": "faq"},
        },
        {
            "chunk_id": "faq-2",
            "document_id": "faq-atm",
            "title": title,
            "source_url": source_url,
            "text": "Answer part 3",
            "content_hash": "hash",
            "language": "vi",
            "product_type": "transfer",
            "section": "faq",
            "chunk_index": 2,
            "metadata": {"document_type": "faq"},
        },
    ]
    with (chunks_dir / "vietcombank_chunks.jsonl").open("w", encoding="utf-8") as file:
        for payload in payloads:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    reranked = [
        RetrievedChunk(
            chunk_id="faq-2",
            document_id="faq-atm",
            title=title,
            source_url=source_url,
            text="Answer part 3",
            score=1.0,
            section="faq",
            product_type="transfer",
        ),
        _chunk(chunk_id="other", title="Other FAQ", text="Other answer", score=0.8),
    ]

    expanded = _expand_exact_faq_context(
        "ATM refund failed but account debited?",
        reranked,
        data_root=tmp_path,
    )

    assert [chunk.chunk_id for chunk in expanded] == ["faq-0", "faq-1", "faq-2", "other"]


def test_exact_faq_lookup_reads_dedicated_faq_chunks_file(tmp_path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    title = (
        "Trong thời gian xây dựng, chưa có thu nhập từ tiền kinh doanh cơ sở lưu trú, "
        "Ngân hàng có cho phép tôi được ân hạn gốc vay?"
    )
    payload = {
        "chunk_id": "faq-grace-period",
        "document_id": "faq-loan",
        "title": title,
        "source_url": "https://www.vietcombank.com.vn/faq#faq=loan-grace-period",
        "text": (
            "[FAQ] Question: "
            f"{title} Answer: Khách hàng được ân hạn trả nợ gốc tối đa 24 tháng."
        ),
        "content_hash": "hash",
        "language": "vi",
        "product_type": "loan",
        "section": "faq",
        "chunk_index": 0,
        "metadata": {"document_type": "faq"},
    }
    with (chunks_dir / "vietcombank_faq_chunks.jsonl").open("w", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    chunks = _exact_faq_chunks_for_query(title, data_root=tmp_path)

    assert [chunk.chunk_id for chunk in chunks] == ["faq-grace-period"]
    assert chunks[0].product_type == "loan"


def test_exact_faq_lookup_prioritizes_matching_category_when_titles_duplicate(tmp_path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    title = (
        "Trong thời gian xây dựng, chưa có thu nhập từ tiền kinh doanh cơ sở lưu trú, "
        "Ngân hàng có cho phép tôi được ân hạn gốc vay?"
    )
    payloads = [
        {
            "chunk_id": "faq-nang-cap",
            "document_id": "faq-loan-upgrade",
            "title": title,
            "source_url": "https://www.vietcombank.com.vn/faq#faq=Vay---Nang-cap-co-so-LTDL---Q5",
            "text": f"[FAQ] Question: {title} Answer: Tối đa 06 tháng.",
            "product_type": "loan",
            "section": "faq",
            "chunk_index": 0,
            "metadata": {"document_type": "faq", "category": "Nang cap co so luu tru du lich"},
        },
        {
            "chunk_id": "faq-xay-moi",
            "document_id": "faq-loan-new-build",
            "title": title,
            "source_url": "https://www.vietcombank.com.vn/faq#faq=Vay---Xay-moi-co-so-luu-tru---Q5",
            "text": f"[FAQ] Question: {title} Answer: Tối đa 24 tháng.",
            "product_type": "loan",
            "section": "faq",
            "chunk_index": 0,
            "metadata": {"document_type": "faq", "category": "Xay moi co so luu tru"},
        },
    ]
    with (chunks_dir / "vietcombank_faq_chunks.jsonl").open("w", encoding="utf-8") as file:
        for payload in payloads:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    chunks = _exact_faq_chunks_for_query(title, data_root=tmp_path)

    assert [chunk.chunk_id for chunk in chunks[:2]] == ["faq-xay-moi", "faq-nang-cap"]


def test_follow_up_query_resolves_recent_product_subject() -> None:
    resolved = _resolve_retrieval_query(
        "Điều kiện để mở thẻ đó",
        [
            ChatMessage(
                role="user",
                content="Vietcombank Chợ Rẫy Connect24 cho tôi thông tin về nó",
            ),
            ChatMessage(
                role="assistant",
                content="Thẻ Vietcombank Chợ Rẫy Connect24 là thẻ thanh toán chuyên dùng.",
            ),
        ],
    )

    assert resolved == "Điều kiện để mở thẻ đó Vietcombank Chợ Rẫy Connect24"


def test_standalone_query_is_not_resolved_from_history() -> None:
    resolved = _resolve_retrieval_query(
        "VCB có các thẻ thanh toán nào?",
        [
            ChatMessage(
                role="user",
                content="Vietcombank Chợ Rẫy Connect24 cho tôi thông tin về nó",
            )
        ],
    )

    assert resolved == "VCB có các thẻ thanh toán nào?"


def test_transfer_catalog_query_prefers_receive_and_outbound_foreign_categories() -> None:
    preferred_categories = _preferred_catalog_category_keys(
        "VCB co cho chuyen va nhan tien tu nuoc ngoai khong?",
        "transfer",
    )

    assert preferred_categories == ["nhan kieu hoi", "chuyen tien ra nuoc ngoai"]


def test_insurance_catalog_query_prefers_named_group_only_when_specific() -> None:
    assert _preferred_catalog_category_keys(
        "VCB co cac goi bao hiem nao?",
        "insurance",
    ) == []
    assert _preferred_catalog_category_keys(
        "Bao hiem bao ve co san pham nao?",
        "insurance",
    ) == ["bao hiem bao ve"]


def test_catalog_fallback_filter_keeps_matching_catalog_chunks() -> None:
    insurance_catalog = _chunk(
        chunk_id="insurance-catalog",
        title="Danh sách gói Bảo hiểm Vietcombank",
        text="FWD Con vươn xa 2.0; FWD Vững ước mơ",
        score=0.9,
        section="product_catalog",
        product_type="insurance",
    )
    card_catalog = _chunk(
        chunk_id="card-catalog",
        title="Danh sách gói Thẻ Vietcombank",
        text="Thẻ tín dụng",
        score=0.8,
        section="product_catalog",
        product_type="card",
    )
    detail = _chunk(
        chunk_id="detail",
        title="FWD Bảo vệ gia tăng",
        text="Chi tiết sản phẩm",
        score=0.7,
        section="product_detail",
        product_type="insurance",
    )

    assert _filter_catalog_chunks(
        [insurance_catalog, card_catalog, detail],
        product_type="insurance",
    ) == [insurance_catalog]


def test_catalog_filter_prefers_parent_catalog_for_general_list_query() -> None:
    parent_catalog = _chunk(
        chunk_id="insurance-parent",
        title="Danh sach goi Bao hiem Vietcombank",
        text="Bao hiem tiet kiem; Bao hiem bao ve; Bao hiem dau tu",
        score=0.9,
        section="product_catalog",
        product_type="insurance",
        metadata={"category_title": "Bao hiem"},
    )
    protection_catalog = _chunk(
        chunk_id="insurance-protection",
        title="Danh sach goi Bao hiem bao ve Vietcombank",
        text="FWD Vung uoc mo; FWD Ca nha vui khoe",
        score=0.95,
        section="product_catalog",
        product_type="insurance",
        metadata={"category_title": "Bao hiem bao ve", "parent_category_title": "Bao hiem"},
    )

    assert _filter_catalog_chunks(
        [protection_catalog, parent_catalog],
        product_type="insurance",
    ) == [parent_catalog]


def test_catalog_filter_keeps_specific_insurance_subcategory() -> None:
    parent_catalog = _chunk(
        chunk_id="insurance-parent",
        title="Danh sach goi Bao hiem Vietcombank",
        text="Bao hiem tiet kiem; Bao hiem bao ve; Bao hiem dau tu",
        score=0.9,
        section="product_catalog",
        product_type="insurance",
        metadata={"category_title": "Bao hiem"},
    )
    protection_catalog = _chunk(
        chunk_id="insurance-protection",
        title="Danh sach goi Bao hiem bao ve Vietcombank",
        text="FWD Vung uoc mo; FWD Ca nha vui khoe",
        score=0.95,
        section="product_catalog",
        product_type="insurance",
        metadata={"category_title": "Bao hiem bao ve", "parent_category_title": "Bao hiem"},
    )

    assert _filter_catalog_chunks(
        [parent_catalog, protection_catalog],
        product_type="insurance",
        preferred_category_keys=["bao hiem bao ve"],
    ) == [protection_catalog]


def test_catalog_filter_prefers_matching_transfer_subcategories() -> None:
    parent_catalog = _chunk(
        chunk_id="transfer-parent",
        title="Danh sach goi Chuyen va nhan tien Vietcombank",
        text="Chuyen va nhan tien trong nuoc; Nhan kieu hoi tai Viet Nam; Chuyen tien ra nuoc ngoai",
        score=0.95,
        section="product_catalog",
        product_type="transfer",
        metadata={"category_title": "Chuyen va nhan tien"},
    )
    outbound_catalog = _chunk(
        chunk_id="outbound-transfer",
        title="Danh sach goi Chuyen tien ra nuoc ngoai Vietcombank",
        text="Chuyen tien ra nuoc ngoai",
        score=0.9,
        section="product_catalog",
        product_type="transfer",
        metadata={"category_title": "Chuyen tien ra nuoc ngoai"},
    )
    remittance_catalog = _chunk(
        chunk_id="remittance",
        title="Danh sach goi Nhan kieu hoi Vietcombank",
        text="Nhan kieu hoi tai Viet Nam",
        score=0.7,
        section="product_catalog",
        product_type="transfer",
        metadata={"category_title": "Nhan kieu hoi"},
    )
    domestic_catalog = _chunk(
        chunk_id="domestic-transfer",
        title="Danh sach goi Chuyen va nhan tien trong nuoc Vietcombank",
        text="Chuyen va nhan tien trong nuoc",
        score=0.8,
        section="product_catalog",
        product_type="transfer",
        metadata={"category_title": "Chuyen va nhan tien trong nuoc"},
    )

    filtered = _filter_catalog_chunks(
        [parent_catalog, outbound_catalog, remittance_catalog, domestic_catalog],
        product_type="transfer",
        preferred_category_keys=["nhan kieu hoi", "chuyen tien ra nuoc ngoai"],
    )

    assert filtered == [remittance_catalog, outbound_catalog]


def _chunk(
    *,
    chunk_id: str,
    title: str,
    text: str,
    score: float,
    section: str | None = None,
    product_type: str | None = None,
    metadata: dict[str, object] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=title,
        source_url="https://www.vietcombank.com.vn/example",
        text=text,
        score=score,
        section=section,
        product_type=product_type,
        metadata=metadata or {},
    )
