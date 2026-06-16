from apps.api.app.models.chat import ChatMessage
from apps.api.app.rag.retrieval.graph import ProductGraphRetriever, _normalize_query_key


def test_graph_loads_catalog_products_and_detail_chunks() -> None:
    retriever = ProductGraphRetriever()

    assert len(retriever.graph.categories_by_key) >= 30
    assert len(retriever.graph.products_by_url) >= 70
    assert len(retriever.graph.detail_chunks_by_url) >= 70


def test_graph_retrieves_complete_parent_insurance_catalog() -> None:
    result = ProductGraphRetriever().retrieve(
        "Vietcombank co cac goi bao hiem nao?",
        history=[],
    )

    assert result.route == "graph"
    assert result.chunks[0].title == "Danh sách gói Bảo hiểm Vietcombank"
    assert result.chunks[0].metadata["item_count"] == 10


def test_graph_retrieves_specific_insurance_subcategory_catalog() -> None:
    result = ProductGraphRetriever().retrieve(
        "Nhom Bao hiem bao ve co cac goi nao?",
        history=[],
    )

    assert result.route == "graph"
    assert [chunk.title for chunk in result.chunks] == ["Danh sách gói Bảo hiểm bảo vệ Vietcombank"]
    assert result.chunks[0].metadata["item_count"] == 6


def test_graph_resolves_follow_up_to_recent_product_and_detail_chunks() -> None:
    result = ProductGraphRetriever().retrieve(
        "Dieu kien de mo the do",
        history=[
            ChatMessage(
                role="user",
                content="Vietcombank Cho Ray Connect24 cho toi thong tin ve no",
            )
        ],
    )

    assert result.route == "graph"
    assert result.resolved_query == "Dieu kien de mo the do Vietcombank Chợ Rẫy Connect24"
    assert result.chunks[0].title == "Vietcombank Chợ Rẫy Connect24"
    assert result.chunks[0].source_url.endswith("/Debit_VCB-Cho-Ray-24h")
    assert len(result.chunks) > 1


def test_graph_prioritizes_specific_product_over_parent_category() -> None:
    result = ProductGraphRetriever().retrieve(
        "Dieu kien de mua goi bao hiem FWD Con vuon xa 2.0",
        history=[],
    )

    assert result.route == "graph"
    assert "FWD Con" in result.chunks[0].title
    assert result.chunks[0].section == "product_detail"


def test_graph_asks_clarification_for_unresolved_follow_up() -> None:
    result = ProductGraphRetriever().retrieve(
        "Dieu kien cua no?",
        history=[],
    )

    assert result.route == "graph_clarification"
    assert result.clarification is not None
    assert result.chunks == []


def test_graph_suggests_product_for_typo_subject() -> None:
    options = ProductGraphRetriever().suggest_subjects("Cho Ray Conect24 dieu kien", limit=3)

    assert any(option.url.endswith("/Debit_VCB-Cho-Ray-24h") for option in options)


def test_graph_suggests_category_for_typo_group() -> None:
    options = ProductGraphRetriever().suggest_subjects("bao hime bao ve co goi nao", limit=3)

    assert any(option.subject_type == "category" and "Bảo hiểm bảo vệ" in option.title for option in options)


def test_graph_matches_category_when_query_inserts_token_between_alias_words() -> None:
    options = ProductGraphRetriever().match_subjects(
        "nhan tien kieu hoi tai Vietcombank khac gi voi nhung cho khac",
        limit=5,
    )

    assert any(
        option.subject_type == "category"
        and _normalize_query_key(option.title) == "nhan kieu hoi"
        for option in options
    )


def test_graph_does_not_match_category_when_alias_tokens_are_far_apart() -> None:
    result = ProductGraphRetriever().retrieve(
        "mat khau su dung dich vu VCB Digibank nen duoc dat nhu the nao",
        history=[],
    )

    assert result.route == "default"
    assert result.chunks == []


def test_graph_prefers_mastercard_debit_over_shorter_credit_alias() -> None:
    result = ProductGraphRetriever().retrieve(
        "Loi ich va dieu kien mo the Vietcombank Mastercard Debit",
        history=[],
        top_k=12,
    )

    assert result.route == "graph"
    assert len(result.chunks) >= 5
    assert all(chunk.source_url.endswith("/Debit_VCB-Mastercard") for chunk in result.chunks)
    assert all("Debit" in chunk.title for chunk in result.chunks)


def test_graph_prefers_vibe_platinum_over_shorter_vibe_alias() -> None:
    result = ProductGraphRetriever().retrieve(
        "Loi ich cua Vietcombank Vibe Platinum",
        history=[],
        top_k=12,
    )

    assert result.route == "graph"
    assert len(result.chunks) >= 5
    assert all(chunk.source_url.endswith("/Credit_VCB-Vibe-Platinum") for chunk in result.chunks)


def test_graph_keeps_standalone_and_longer_product_in_comparison() -> None:
    result = ProductGraphRetriever().retrieve(
        "So sanh Vietcombank Vibe va Vietcombank Vibe Platinum",
        history=[],
        top_k=12,
    )

    source_urls = {chunk.source_url for chunk in result.chunks}
    assert any(url.endswith("/Credit_VCB-Vibe") for url in source_urls)
    assert any(url.endswith("/Credit_VCB-Vibe-Platinum") for url in source_urls)


def test_graph_returns_duplicate_product_versions_for_comparison() -> None:
    result = ProductGraphRetriever().retrieve(
        "FWD Bao hiem tai nan truc tuyen co gi khac nhau?",
        history=[],
    )

    source_urls = [chunk.source_url for chunk in result.chunks[:2]]
    assert any("KHDN" in source_url for source_url in source_urls)
    assert any("KHCN" in source_url for source_url in source_urls)
    assert result.chunks[0].title.endswith("(KHDN)")
    assert result.chunks[1].title.endswith("(KHCN)")
