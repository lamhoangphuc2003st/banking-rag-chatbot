from urllib.parse import unquote, urldefrag

from apps.api.app.rag.citations import build_citations
from packages.shared.schemas import RetrievedChunk


def test_build_citations_deduplicates_by_source_url() -> None:
    citations = build_citations(
        [
            _chunk("chunk-1", "Vay mua ô tô", "https://example.com/vay-mua-o-to", 0.9),
            _chunk("chunk-2", "Vay mua ô tô", "https://example.com/vay-mua-o-to", 0.8),
        ],
        query="Điều kiện vay mua ô tô là gì?",
    )

    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-1"


def test_build_citations_keeps_exact_product_title_for_specific_query() -> None:
    citations = build_citations(
        [
            _chunk("auto", "Vay mua ô tô", "https://example.com/vay-mua-o-to", 0.9),
            _chunk("home", "Vay mua nhà ở, đất ở", "https://example.com/vay-mua-nha", 0.88),
            _chunk("generic", "Vay tiêu dùng có tài sản bảo đảm", "https://example.com/vay-tieu-dung", 0.87),
        ],
        query="Điều kiện vay mua ô tô là gì?",
    )

    assert [citation.chunk_id for citation in citations] == ["auto"]


def test_build_citations_keeps_multiple_products_for_broad_availability_query() -> None:
    citations = build_citations(
        [
            _chunk("home-land", "Vay mua nhà ở, đất ở", "https://example.com/vay-mua-nha-dat", 0.9),
            _chunk("home-project", "Vay mua nhà dự án", "https://example.com/vay-mua-nha-du-an", 0.88),
        ],
        query="VCB có cho vay để mua nhà, mua đất không?",
    )

    assert [citation.chunk_id for citation in citations] == ["home-land", "home-project"]


def test_build_citations_keeps_specific_fwd_product_only() -> None:
    citations = build_citations(
        [
            _chunk("child", "FWD Con vươn xa 2.0", "https://example.com/fwd-con-vuon-xa", 0.9),
            _chunk("dream", "FWD Vững ước mơ", "https://example.com/fwd-vung-uoc-mo", 0.8),
            _chunk("family", "FWD Cả nhà vui khỏe", "https://example.com/fwd-ca-nha-vui-khoe", 0.7),
        ],
        query="Cho tôi biết toàn bộ thông tin về sản phẩm FWD Con vươn xa 2.0",
    )

    assert [citation.chunk_id for citation in citations] == ["child"]


def test_build_citations_keeps_duplicate_versions_for_comparison_query() -> None:
    citations = build_citations(
        [
            _chunk(
                "khdn",
                "FWD Bảo hiểm tai nạn trực tuyến (KHDN)",
                "https://example.com/FWD-Bao-hiem-tai-nan-truc-tuyen-KHDN",
                0.9,
            ),
            _chunk(
                "khcn",
                "FWD Bảo hiểm tai nạn trực tuyến (KHCN)",
                "https://example.com/KHCN---FWD-Bao-hiem-tai-nan-truc-tuyen",
                0.88,
            ),
        ],
        query="FWD Bao hiem tai nan truc tuyen co gi khac nhau?",
    )

    assert [citation.chunk_id for citation in citations] == ["khdn", "khcn"]


def test_build_citations_expands_product_catalog_items() -> None:
    citations = build_citations(
        [
            _chunk(
                "transfer-catalog",
                "Danh sách gói Chuyển và nhận tiền Vietcombank",
                "https://example.com/chuyen-va-nhan-tien",
                0.9,
                section="product_catalog",
                metadata={
                    "items": [
                        {
                            "title": "Chuyển và nhận tiền trong nước",
                            "url": "https://example.com/chuyen-nhan-trong-nuoc",
                            "category": "Chuyển và nhận tiền trong nước",
                            "summary": "Đa kênh",
                        },
                        {
                            "title": "Nhận kiều hối tại Việt Nam",
                            "url": "https://example.com/nhan-kieu-hoi",
                            "category": "Nhận kiều hối",
                            "summary": "Nhanh chóng",
                        },
                        {
                            "title": "Chuyển tiền ra nước ngoài",
                            "url": "https://example.com/chuyen-tien-ra-nuoc-ngoai",
                            "category": "Chuyển tiền ra nước ngoài",
                            "summary": "Hơn 190 loại tiền",
                        },
                    ]
                },
            )
        ],
        query="Vietcombank có các dịch vụ Chuyển và nhận tiền nào?",
    )

    assert [citation.title for citation in citations] == [
        "Chuyển và nhận tiền trong nước",
        "Nhận kiều hối tại Việt Nam",
        "Chuyển tiền ra nước ngoài",
    ]
    assert citations[0].source_url == "https://example.com/chuyen-nhan-trong-nuoc"
    assert all("Danh sách" not in citation.title for citation in citations)
    assert len({citation.chunk_id for citation in citations}) == 3


def test_build_citations_links_catalog_groups_for_multi_item_groups() -> None:
    citations = build_citations(
        [
            _chunk(
                "insurance-catalog",
                "Danh sách gói Bảo hiểm Vietcombank",
                "https://example.com/bao-hiem",
                0.9,
                product_type="insurance",
                section="product_catalog",
                metadata={
                    "category_url": "https://example.com/bao-hiem",
                    "items": [
                        {
                            "title": "FWD Con vươn xa 2.0",
                            "url": "https://example.com/fwd-con-vuon-xa",
                            "category": "Bảo hiểm tiết kiệm",
                            "summary": "Quỹ học vấn",
                        },
                        {
                            "title": "FWD Vững ước mơ",
                            "url": "https://example.com/fwd-vung-uoc-mo",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Thanh toán khoản vay",
                        },
                        {
                            "title": "FWD Cả nhà vui khỏe",
                            "url": "https://example.com/fwd-ca-nha-vui-khoe",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Bảo vệ sức khỏe gia đình",
                        },
                        {
                            "title": "FWD Đầu tư đón đầu",
                            "url": "https://example.com/fwd-dau-tu-don-dau",
                            "category": "Bảo hiểm đầu tư",
                            "summary": "Gia tăng tích lũy",
                        },
                    ],
                },
            )
        ],
        query="VCB có các gói bảo hiểm nào?",
    )

    assert [citation.title for citation in citations] == [
        "Bảo hiểm tiết kiệm",
        "Bảo hiểm bảo vệ",
        "Bảo hiểm đầu tư",
    ]
    assert citations[1].source_url == (
        "https://example.com/bao-hiem#subcategory=B%E1%BA%A3o%20hi%E1%BB%83m%20b%E1%BA%A3o%20v%E1%BB%87"
        "&loan-list_type=B%E1%BA%A3o%20hi%E1%BB%83m%20b%E1%BA%A3o%20v%E1%BB%87&e=0"
    )
    assert all(":group:" in citation.chunk_id for citation in citations)


def test_build_citations_can_include_product_links_for_grouped_catalogs() -> None:
    citations = build_citations(
        [
            _chunk(
                "insurance-catalog",
                "Danh sách gói Bảo hiểm Vietcombank",
                "https://example.com/bao-hiem",
                0.9,
                product_type="insurance",
                section="product_catalog",
                metadata={
                    "category_url": "https://example.com/bao-hiem",
                    "items": [
                        {
                            "title": "FWD Con vươn xa 2.0",
                            "url": "https://example.com/fwd-con-vuon-xa",
                            "category": "Bảo hiểm tiết kiệm",
                            "summary": "Quỹ học vấn",
                        },
                        {
                            "title": "FWD Vững ước mơ",
                            "url": "https://example.com/fwd-vung-uoc-mo",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Thanh toán khoản vay",
                        },
                        {
                            "title": "FWD Cả nhà vui khỏe",
                            "url": "https://example.com/fwd-ca-nha-vui-khoe",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Bảo vệ sức khỏe gia đình",
                        },
                    ],
                },
            )
        ],
        query="VCB có các gói bảo hiểm nào?",
        include_grouped_catalog_items=True,
    )

    assert [citation.title for citation in citations[:3]] == [
        "FWD Con vươn xa 2.0",
        "FWD Vững ước mơ",
        "FWD Cả nhà vui khỏe",
    ]
    assert [citation.source_url for citation in citations[:3]] == [
        "https://example.com/fwd-con-vuon-xa",
        "https://example.com/fwd-vung-uoc-mo",
        "https://example.com/fwd-ca-nha-vui-khoe",
    ]
    assert all(":item:" in citation.chunk_id for citation in citations[:3])
    assert any(":group:" in citation.chunk_id for citation in citations[3:])


def test_build_citations_prefers_exact_catalog_items_over_groups() -> None:
    citations = build_citations(
        [
            _chunk(
                "insurance-catalog",
                "Danh sách gói Bảo hiểm Vietcombank",
                "https://example.com/bao-hiem",
                0.9,
                product_type="insurance",
                section="product_catalog",
                metadata={
                    "category_url": "https://example.com/bao-hiem",
                    "items": [
                        {
                            "title": "FWD Vững ước mơ",
                            "url": "https://example.com/fwd-vung-uoc-mo",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Thanh toán khoản vay",
                        },
                        {
                            "title": "FWD Bảo hiểm tai nạn trực tuyến",
                            "url": "https://example.com/KHCN---FWD-Bao-hiem-tai-nan-truc-tuyen",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Linh hoạt theo nhu cầu",
                        },
                        {
                            "title": "FWD Bảo hiểm tai nạn trực tuyến",
                            "url": "https://example.com/FWD-Bao-hiem-tai-nan-truc-tuyen-KHDN",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Cho khách hàng doanh nghiệp",
                        },
                        {
                            "title": "FWD Đầu tư đón đầu",
                            "url": "https://example.com/fwd-dau-tu-don-dau",
                            "category": "Bảo hiểm đầu tư",
                            "summary": "Gia tăng tích lũy",
                        },
                    ],
                },
            )
        ],
        query="Tại sao FWD Bảo hiểm tai nạn trực tuyến có 2 phiên bản?",
    )

    assert [citation.title for citation in citations] == [
        "FWD Bảo hiểm tai nạn trực tuyến (KHCN)",
        "FWD Bảo hiểm tai nạn trực tuyến (KHDN)",
    ]
    assert [citation.source_url for citation in citations] == [
        "https://example.com/KHCN---FWD-Bao-hiem-tai-nan-truc-tuyen",
        "https://example.com/FWD-Bao-hiem-tai-nan-truc-tuyen-KHDN",
    ]
    assert all(":item:" in citation.chunk_id for citation in citations)


def test_build_citations_links_group_then_all_items_when_query_targets_one_catalog_group() -> None:
    citations = build_citations(
        [
            _chunk(
                "insurance-catalog",
                "Danh sách gói Bảo hiểm Vietcombank",
                "https://example.com/bao-hiem",
                0.9,
                product_type="insurance",
                section="product_catalog",
                metadata={
                    "category_url": "https://example.com/bao-hiem",
                    "items": [
                        {
                            "title": "FWD Con vươn xa 2.0",
                            "url": "https://example.com/fwd-con-vuon-xa",
                            "category": "Bảo hiểm tiết kiệm",
                            "summary": "Quỹ học vấn",
                        },
                        {
                            "title": "FWD Vững ước mơ",
                            "url": "https://example.com/fwd-vung-uoc-mo",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Thanh toán khoản vay",
                        },
                        {
                            "title": "FWD Cả nhà vui khỏe",
                            "url": "https://example.com/fwd-ca-nha-vui-khoe",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Bảo vệ sức khỏe gia đình",
                        },
                    ],
                },
            )
        ],
        query="Bảo hiểm bảo vệ có sản phẩm nào?",
    )

    assert [citation.title for citation in citations] == [
        "Bảo hiểm bảo vệ",
        "FWD Vững ước mơ",
        "FWD Cả nhà vui khỏe",
    ]
    assert citations[0].source_url == (
        "https://example.com/bao-hiem#subcategory=B%E1%BA%A3o%20hi%E1%BB%83m%20b%E1%BA%A3o%20v%E1%BB%87"
        "&loan-list_type=B%E1%BA%A3o%20hi%E1%BB%83m%20b%E1%BA%A3o%20v%E1%BB%87&e=0"
    )
    assert ":group:" in citations[0].chunk_id
    assert all(":item:" in citation.chunk_id for citation in citations[1:])


def test_build_citations_links_subcatalog_group_then_all_items() -> None:
    citations = build_citations(
        [
            _chunk(
                "insurance-protection-catalog",
                "Danh sách gói Bảo hiểm bảo vệ Vietcombank",
                "https://example.com/bao-hiem#subcategory=Bao%20hiem%20bao%20ve&loan-list_type=Bao%20hiem%20bao%20ve&e=0",
                0.9,
                product_type="insurance",
                section="product_catalog",
                metadata={
                    "category_url": "https://example.com/bao-hiem",
                    "category_title": "Bảo hiểm bảo vệ",
                    "parent_category_title": "Bảo hiểm",
                    "items": [
                        {
                            "title": "FWD Vững ước mơ",
                            "url": "https://example.com/fwd-vung-uoc-mo",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Thanh toán khoản vay",
                        },
                        {
                            "title": "FWD Cả nhà vui khỏe",
                            "url": "https://example.com/fwd-ca-nha-vui-khoe",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Bảo vệ sức khỏe gia đình",
                        },
                        {
                            "title": "FWD Vững ước mơ - Đóng phí 1 lần",
                            "url": "https://example.com/fwd-vung-uoc-mo-dong-phi-1-lan",
                            "category": "Bảo hiểm bảo vệ",
                            "summary": "Chỉ đóng 1 lần",
                        },
                    ],
                },
            )
        ],
        query="Nhóm Bảo hiểm bảo vệ có các gói nào?",
    )

    assert [citation.title for citation in citations] == [
        "Bảo hiểm bảo vệ",
        "FWD Vững ước mơ",
        "FWD Cả nhà vui khỏe",
        "FWD Vững ước mơ - Đóng phí 1 lần",
    ]
    assert ":group:" in citations[0].chunk_id
    assert all(":item:" in citation.chunk_id for citation in citations[1:])


def test_build_citations_prioritizes_matching_catalog_item_category() -> None:
    citations = build_citations(
        [
            _chunk(
                "card-catalog",
                "Danh sách gói Thẻ Vietcombank",
                "https://example.com/the",
                0.9,
                product_type="card",
                section="product_catalog",
                metadata={
                    "category_url": "https://example.com/the",
                    "items": [
                        {
                            "title": "Vietcombank Vibe Platinum",
                            "url": "https://example.com/credit-vibe",
                            "category": "Thẻ tín dụng",
                            "summary": "Tích điểm",
                        },
                        {
                            "title": "Vietcombank Takashimaya Visa",
                            "url": "https://example.com/debit-takashimaya",
                            "category": "Thẻ thanh toán",
                            "summary": "Miễn phí phát hành",
                        },
                        {
                            "title": "Vietcombank Connect24",
                            "url": "https://example.com/debit-connect24",
                            "category": "Thẻ thanh toán",
                            "summary": "Thanh toán không tiếp xúc",
                        },
                    ]
                },
            )
        ],
        query="VCB có các thẻ thanh toán nào?",
    )

    assert [citation.title for citation in citations] == [
        "Thẻ thanh toán",
        "Vietcombank Takashimaya Visa",
        "Vietcombank Connect24",
    ]
    assert "card-list_type=Th%E1%BA%BB%20thanh%20to%C3%A1n" in citations[0].source_url


def test_build_citations_filters_catalog_items_to_answered_products() -> None:
    citations = build_citations(
        [
            _chunk(
                "loan-catalog",
                "Danh sach goi vay Vietcombank",
                "https://example.com/vay",
                0.9,
                product_type="loan",
                section="product_catalog",
                metadata={
                    "category_url": "https://example.com/vay",
                    "items": [
                        {
                            "title": "Vay tin chap theo luong",
                            "url": "https://example.com/vay-tin-chap-theo-luong",
                            "category": "Vay tieu dung",
                            "summary": "Muc vay linh hoat",
                        },
                        {
                            "title": "Vay mua o to",
                            "url": "https://example.com/vay-mua-o-to",
                            "category": "Vay mua o to",
                            "summary": "Muc vay len toi 100% gia tri xe",
                        },
                        {
                            "title": "An tam kinh doanh",
                            "url": "https://example.com/an-tam-kinh-doanh",
                            "category": "Vay san xuat kinh doanh",
                            "summary": "Muc vay len toi 70% phuong an vay",
                        },
                        {
                            "title": "Vay nang cap co so luu tru du lich",
                            "url": "https://example.com/vay-nang-cap-co-so-luu-tru-du-lich",
                            "category": "Vay san xuat kinh doanh",
                            "summary": "Muc vay len toi 60% phuong an vay",
                        },
                        {
                            "title": "Kinh doanh tai loc",
                            "url": "https://example.com/kinh-doanh-tai-loc",
                            "category": "Vay san xuat kinh doanh",
                            "summary": "Muc vay len toi 85% phuong an kinh doanh",
                        },
                    ],
                },
            )
        ],
        query="VCB co cac goi vay phuc vu nhu cau kinh doanh nao?",
        answer=(
            "Vietcombank co cac goi vay phuc vu nhu cau kinh doanh gom: "
            "1. An tam kinh doanh. "
            "2. Vay nang cap co so luu tru du lich. "
            "3. Kinh doanh tai loc."
        ),
        include_grouped_catalog_items=True,
    )

    assert [citation.title for citation in citations] == [
        "An tam kinh doanh",
        "Vay nang cap co so luu tru du lich",
        "Kinh doanh tai loc",
    ]


def test_build_citations_filters_expanded_detail_chunks_to_answered_products() -> None:
    citations = build_citations(
        [
            _chunk(
                "loan-auto-catalog",
                "Danh sach goi Vay mua o to Vietcombank",
                "https://example.com/vay#subcategory=vay-mua-o-to",
                2.7,
                product_type="loan",
                section="product_catalog",
                metadata={
                    "category_title": "Vay mua o to",
                    "parent_category_title": "Vay",
                    "items": [
                        {
                            "title": "Vay mua o to",
                            "url": "https://example.com/vay-mua-o-to",
                            "category": "Vay mua o to",
                            "summary": "Muc vay len toi 100% gia tri xe",
                        },
                    ],
                },
            ),
            _chunk(
                "composed:product:vay-tin-chap",
                "Vay tin chap theo luong",
                "https://example.com/vay-tin-chap-theo-luong",
                2.6,
                section="product_detail",
                metadata={"retrieval_source": "query_composition"},
            ),
            _chunk(
                "composed:product:nha-moi-thanh-dat",
                "Nha Moi Thanh Dat",
                "https://example.com/nha-moi-thanh-dat",
                2.5,
                section="product_detail",
                metadata={"retrieval_source": "query_composition"},
            ),
            _chunk(
                "composed:product:vay-mua-nha-du-an",
                "Vay mua nha du an",
                "https://example.com/vay-mua-nha-du-an",
                2.4,
                section="product_detail",
                metadata={"retrieval_source": "query_composition"},
            ),
            _chunk(
                "graph:product:vay-mua-o-to",
                "Vay mua o to",
                "https://example.com/vay-mua-o-to",
                2.3,
                section="product_detail",
                metadata={"retrieval_source": "graph"},
            ),
        ],
        query="VCB co ho tro vay mua nha khong?",
        answer="Vietcombank co ho tro mua nha qua Nha Moi Thanh Dat va Vay mua nha du an.",
        include_grouped_catalog_items=True,
    )

    assert [citation.title for citation in citations] == [
        "Nha Moi Thanh Dat",
        "Vay mua nha du an",
    ]


def test_build_citations_rewrites_faq_questiontype_slug_to_display_label() -> None:
    citations = build_citations(
        [
            _chunk(
                "faq-xay-moi-q5",
                "Trong thời gian xây dựng, chưa có thu nhập từ tiền kinh doanh cơ sở lưu trú, Ngân hàng có cho phép tôi được ân hạn gốc vay?",
                "https://www.vietcombank.com.vn/vi-VN/KHCN/Lien-he-va-Ho-tro/Danh-sach-cau-hoi-theo-chu-de-Vay#p=10&e=0&questiontype=Xay%20moi%20co%20so%20luu%20tru&comp=faq_ls_tp&faq=Vay---Xay-moi-co-so-luu-tru---Q5",
                3.0,
                product_type="loan",
                section="faq",
                metadata={
                    "document_type": "faq",
                    "category": "Xay moi co so luu tru",
                    "category_slug": "Xay moi co so luu tru",
                },
            )
        ],
        query="ân hạn gốc vay cơ sở lưu trú",
    )

    fragment = unquote(urldefrag(citations[0].source_url).fragment)

    assert citations[0].title.startswith("Vay xây mới cơ sở lưu trú du lịch:")
    assert "questiontype=Vay xây mới cơ sở lưu trú du lịch" in fragment
    assert "faq=Vay---Xay-moi-co-so-luu-tru---Q5" in fragment


def _chunk(
    chunk_id: str,
    title: str,
    source_url: str,
    score: float,
    *,
    product_type: str | None = None,
    section: str | None = None,
    metadata: dict[str, object] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=title,
        source_url=source_url,
        text=f"{title} là sản phẩm công khai của Vietcombank.",
        score=score,
        product_type=product_type,
        section=section,
        metadata=metadata or {},
    )
