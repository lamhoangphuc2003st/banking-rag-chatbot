from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage, ChatRequest
from apps.api.app.rag.pipeline import RagPipeline


def _test_settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="local",
        rag_cache_backend="memory",
        openai_api_key=None,
        litellm_api_key=None,
    )


async def test_pipeline_asks_product_when_detail_follow_up_after_multi_product_answer() -> None:
    pipeline = RagPipeline(_test_settings())

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content="Vietcombank co cho vay kinh doanh khong"),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Vietcombank co cac goi vay phuc vu muc dich kinh doanh nhu "
                        "An tam kinh doanh, Vay nang cap co so luu tru du lich, "
                        "Vay xay moi co so luu tru du lich va Kinh doanh tai loc."
                    ),
                ),
                ChatMessage(role="user", content="Ho so vay can chuan bi la gi"),
            ]
        )
    )

    option_titles = [option["title"] for option in response.metadata["clarification_options"]]
    assert response.metadata["clarification_required"] is True
    assert response.metadata["query_rewrite_route"] == "local_multi_product_detail_clarification"
    assert response.metadata["query_rewrite_reason"] == "ambiguous_detail_after_multi_product_context"
    assert option_titles == [
        "An tâm kinh doanh",
        "Vay nâng cấp cơ sở lưu trú du lịch",
        "Vay xây mới cơ sở lưu trú du lịch",
        "Kinh doanh tài lộc",
    ]
    assert not response.sources


async def test_pipeline_asks_card_choice_for_ambiguous_detail_after_card_catalog() -> None:
    pipeline = RagPipeline(_test_settings())

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content="VCB hiện có các thẻ thanh toán nào"),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Vietcombank hiện có các thẻ thanh toán sau:\n\n"
                        "1. Vietcombank Takashimaya Visa\n"
                        "2. Vietcombank Connect24\n"
                        "3. Vietcombank Visa Platinum Debit\n"
                        "4. Vietcombank Đại học Quốc gia Hồ Chí Minh Visa\n"
                        "5. Vietcombank Tekmedi Thống Nhất Connect24\n"
                        "6. Vietcombank eVer-link\n"
                        "7. VCB DigiCard\n"
                        "8. Vietcombank Chợ Rẫy Connect24\n"
                        "9. Vietcombank MeGO\n"
                        "10. Vietcombank Mastercard® Debit\n"
                        "11. Thẻ ngừng phát hành"
                    ),
                ),
                ChatMessage(role="user", content="Lợi ích và điều kiện mở thẻ"),
            ]
        )
    )

    option_titles = [option["title"] for option in response.metadata["clarification_options"]]
    assert response.metadata["clarification_required"] is True
    assert response.metadata["query_rewrite_route"] == "local_multi_product_detail_clarification"
    assert response.metadata["query_rewrite_reason"] == "ambiguous_detail_after_multi_product_context"
    assert "Bạn muốn hỏi điều kiện của sản phẩm/gói nào?" in response.answer
    assert "Vietcombank Takashimaya Visa" in option_titles
    assert "Vietcombank Mastercard® Debit" in option_titles
    assert len(option_titles) == 11
    assert not response.sources


async def test_pipeline_asks_product_for_ambiguous_detail_after_insurance_catalog() -> None:
    pipeline = RagPipeline(_test_settings())

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content="VCB hiện có các sản phẩm bảo hiểm nào"),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Vietcombank hiện có các sản phẩm bảo hiểm sau:\n\n"
                        "1. FWD Con vươn xa 2.0\n"
                        "2. FWD Vững ước mơ\n"
                        "3. FWD Cả nhà vui khỏe\n"
                        "4. FWD Vững ước mơ - Đóng phí 1 lần\n"
                        "5. FWD Bảo vệ gia tăng - Phiên bản trực tuyến bán phần\n"
                        "6. Sản phẩm bảo hiểm liên kết chung FWD Bảo vệ gia tăng\n"
                        "7. FWD Bảo hiểm sức khỏe trực tuyến\n"
                        "8. FWD Bảo hiểm tai nạn trực tuyến (dành cho cá nhân)\n"
                        "9. FWD Bảo hiểm tai nạn trực tuyến (dành cho doanh nghiệp)\n"
                        "10. Sản phẩm bảo hiểm liên kết đơn vị FWD Đầu tư đón đầu"
                    ),
                ),
                ChatMessage(role="user", content="Có lợi ích và điều kiện sử dụng là gì"),
            ]
        )
    )

    option_titles = [option["title"] for option in response.metadata["clarification_options"]]
    assert response.metadata["clarification_required"] is True
    assert response.metadata["query_rewrite_route"] == "local_multi_product_detail_clarification"
    assert response.metadata["query_rewrite_reason"] == "ambiguous_detail_after_multi_product_context"
    assert "Bạn muốn hỏi điều kiện của sản phẩm/gói nào?" in response.answer
    assert "FWD Con vươn xa 2.0" in option_titles
    assert any("FWD Bảo hiểm sức khỏe trực tuyến" in title for title in option_titles)
    assert len(option_titles) >= 8
    assert not response.sources


async def test_pipeline_streams_clarification_options_in_metadata() -> None:
    pipeline = RagPipeline(_test_settings())

    events = [
        event
        async for event in pipeline.stream_events(
            ChatRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content="Dieu kien de mua goi bao hiem",
                    )
                ]
            )
        )
    ]
    metadata_event = next(event for event in events if event["type"] == "metadata")
    option_titles = [
        option["title"]
        for option in metadata_event["metadata"]["clarification_options"]
    ]

    assert metadata_event["metadata"]["clarification_required"] is True
    assert option_titles == [
        "Bảo hiểm tiết kiệm",
        "Bảo hiểm bảo vệ",
        "Bảo hiểm đầu tư",
    ]


async def test_pipeline_keeps_single_selected_product_for_detail_follow_up() -> None:
    pipeline = RagPipeline(_test_settings())

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content="Vietcombank co cho vay kinh doanh khong"),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Vietcombank co cac goi vay phuc vu muc dich kinh doanh nhu "
                        "An tam kinh doanh, Vay nang cap co so luu tru du lich, "
                        "Vay xay moi co so luu tru du lich va Kinh doanh tai loc."
                    ),
                ),
                ChatMessage(role="user", content="Vay xay moi co so luu tru du lich"),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Duoi day la dieu kien vay cua Vay xay moi co so luu tru du lich: "
                        "Cong dan Viet Nam tu 18 den 65 tuoi; co nhu cau vay trung dai han; "
                        "co tai san bao dam la bat dong san, o to, giay to co gia."
                    ),
                ),
                ChatMessage(role="user", content="Ho so can chuan bi la gi"),
            ]
        )
    )

    assert response.metadata["query_rewrite_route"] == "local_context_detail"
    assert all("nang cap" not in source.title.lower() for source in response.sources)
    assert response.sources


async def test_pipeline_asks_clarification_for_generic_insurance_condition_query() -> None:
    pipeline = RagPipeline(_test_settings())

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content="Dieu kien de mua goi bao hiem")])
    )

    option_titles = [option["title"] for option in response.metadata["clarification_options"]]
    assert response.metadata["clarification_required"] is True
    assert option_titles[:3] == [
        "Bảo hiểm tiết kiệm",
        "Bảo hiểm bảo vệ",
        "Bảo hiểm đầu tư",
    ]
    assert option_titles == [
        "Bảo hiểm tiết kiệm",
        "Bảo hiểm bảo vệ",
        "Bảo hiểm đầu tư",
    ]
    assert "Tiết kiệm" not in option_titles


async def test_generic_product_type_clarification_does_not_include_parent_category() -> None:
    pipeline = RagPipeline(_test_settings())
    examples = {
        "Dieu kien mo the": {"Thẻ tín dụng", "Thẻ thanh toán", "Dịch vụ thẻ"},
        "Dieu kien vay": {
            "Vay tiêu dùng",
            "Vay mua ô tô",
            "Vay sản xuất kinh doanh",
            "Vay nhu cầu bất động sản",
        },
        "Lai suat tiet kiem": {
            "Tiền gửi tiết kiệm",
            "Tiết kiệm tích lũy",
            "Tiết kiệm trực tuyến",
        },
        "Thu tuc chuyen tien": {
            "Chuyển và nhận tiền trong nước",
            "Nhận kiều hối",
            "Chuyển tiền ra nước ngoài",
        },
        "Dieu kien dau tu": {
            "Chứng khoán",
            "Quỹ mở",
            "Ủy thác quản lý tài khoản",
            "Chứng chỉ tiền gửi",
            "Hỗ trợ tài chính",
        },
    }
    parent_titles = {"Thẻ", "Vay", "Tiết kiệm", "Chuyển và nhận tiền", "Đầu tư"}

    for query, expected_titles in examples.items():
        response = await pipeline.answer(
            ChatRequest(messages=[ChatMessage(role="user", content=query)])
        )
        option_titles = {option["title"] for option in response.metadata["clarification_options"]}

        assert response.metadata["clarification_required"] is True
        assert option_titles <= expected_titles
        assert option_titles
        assert option_titles.isdisjoint(parent_titles)


async def test_pipeline_resolves_numeric_choice_to_single_product_in_group() -> None:
    pipeline = RagPipeline(_test_settings())
    previous_question = "Dieu kien de mua goi bao hiem"
    clarification = "\n".join(
        [
            "Ban muon hoi nhom nao?",
            "1. Bao hiem tiet kiem (nhom)",
            "2. Bao hiem bao ve (nhom)",
            "3. Bao hiem dau tu (nhom)",
        ]
    )

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content=previous_question),
                ChatMessage(role="assistant", content=clarification),
                ChatMessage(role="user", content="1"),
            ]
        )
    )

    assert response.metadata["query_rewrite_route"] == "local_choice"
    assert response.metadata["query_rewrite_applied"] is True
    assert "FWD Con" in response.metadata["retrieval_query"]
    assert response.sources


async def test_pipeline_asks_product_choice_when_numeric_choice_is_multi_product_group() -> None:
    pipeline = RagPipeline(_test_settings())
    previous_question = "Dieu kien de mua goi bao hiem"
    clarification = "\n".join(
        [
            "Ban muon hoi nhom nao?",
            "1. Bao hiem tiet kiem (nhom)",
            "2. Bao hiem bao ve (nhom)",
            "3. Bao hiem dau tu (nhom)",
        ]
    )

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content=previous_question),
                ChatMessage(role="assistant", content=clarification),
                ChatMessage(role="user", content="2"),
            ]
        )
    )

    assert response.metadata["clarification_required"] is True
    assert response.metadata["query_rewrite_route"] == "local_choice_clarification"
    assert "FWD Vững ước mơ" in response.answer
    assert not response.sources


async def test_pipeline_resolves_text_choice_to_selected_group() -> None:
    pipeline = RagPipeline(_test_settings())
    previous_question = "Dieu kien mo the"
    clarification = "\n".join(
        [
            "Ban muon hoi nhom nao?",
            "1. The tin dung (nhom)",
            "2. The thanh toan (nhom)",
            "3. Dich vu the (nhom)",
        ]
    )

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(role="user", content=previous_question),
                ChatMessage(role="assistant", content=clarification),
                ChatMessage(role="user", content="Thẻ thanh toán (nhóm)"),
            ]
        )
    )

    assert response.metadata["clarification_required"] is True
    assert response.metadata["query_rewrite_route"] == "local_choice_clarification"
    assert "Bạn đã chọn Thẻ thanh toán" in response.answer
    assert "Vietcombank Connect24" in response.answer
    assert "Vietcombank Đại học Quốc gia Hồ Chí Minh Visa" in response.answer
    assert "Phí thường niên ít nhất" not in response.answer
    assert not response.sources


async def test_pipeline_answers_card_type_query_with_groups_only() -> None:
    pipeline = RagPipeline(_test_settings())

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content="Vietcombank hiện có các loại thẻ nào")])
    )

    assert response.metadata["retrieval_route"] == "graph:type_only_catalog"
    assert "Thẻ tín dụng" in response.answer
    assert "Thẻ thanh toán" in response.answer
    assert "Dịch vụ thẻ" in response.answer
    assert "Vietcombank Vibe Platinum" not in response.answer
    assert [source.title for source in response.sources] == [
        "Thẻ tín dụng",
        "Thẻ thanh toán",
        "Dịch vụ thẻ",
    ]
