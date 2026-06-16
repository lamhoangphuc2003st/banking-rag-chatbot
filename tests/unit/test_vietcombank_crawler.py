from packages.data_pipeline.crawler.vietcombank import (
    SitecoreSearchConfig,
    canonicalize_product_url,
    extract_faq_category_labels_from_facet_payload,
    extract_faq_topics_from_search_payload,
    extract_linked_resources_from_raw_document,
    extract_product_catalog_items_from_search_payload,
    extract_product_search_configs,
    extract_product_urls_from_search_payload,
    faq_category_from_url,
    faq_facet_config_from_search_config,
    infer_linked_resource_type,
    infer_product_catalog_item_category,
    normalize_faq_category_label,
    product_catalog_subcategory_source_url,
    raw_document_from_faq_result,
    raw_document_from_product_catalog_results,
    raw_documents_from_product_catalog_results,
)
from packages.shared.schemas import RawDocument

BASE_URL = "https://www.vietcombank.com.vn"


def test_extract_product_search_configs_reads_sitecore_properties() -> None:
    html = """
    <div
      class="component search-results cards-list"
      data-properties='{
        "endpoint":"/vi-VN/sxa/searchapi/customresults/",
        "v":"{view-id}",
        "s":"{scope-id}",
        "l":"vi-VN",
        "p":6,
        "sig":"card-list",
        "itemid":"{item-id}",
        "autoFireSearch":true
      }'>
    </div>
    """

    configs = extract_product_search_configs(html, f"{BASE_URL}/vi-VN/KHCN/SPDV/The")

    assert len(configs) == 1
    assert configs[0].endpoint == f"{BASE_URL}/vi-VN/sxa/searchapi/customresults/"
    assert configs[0].params["sig"] == "card-list"
    assert configs[0].params["p"] == "6"
    assert "endpoint" not in configs[0].params
    assert "autoFireSearch" not in configs[0].params


def test_extract_product_urls_from_search_payload_normalizes_and_dedupes() -> None:
    payload = {
        "Results": [
            {
                "Url": "/vi-VN/KHCN/SPDV/The/Credit_VCB-Vibe-Platinum#dataIndex=0",
                "Html": """
                    <a href="/vi-VN/KHCN/SPDV/The/Credit_VCB-Vibe-Platinum?x=1">
                        Duplicate
                    </a>
                    <a href="/vi-VN/KHCN/SPDV/The">Category page</a>
                    <a href="https://example.com/not-vietcombank">External</a>
                """,
            },
            {"Url": "/vi-VN/KHCN/SPDV/Vay/Vay-mua-o-to"},
        ]
    }

    urls = extract_product_urls_from_search_payload(payload, BASE_URL)

    assert urls == [
        f"{BASE_URL}/vi-VN/KHCN/SPDV/The/Credit_VCB-Vibe-Platinum",
        f"{BASE_URL}/vi-VN/KHCN/SPDV/Vay/Vay-mua-o-to",
    ]


def test_canonicalize_product_url_rejects_category_and_external_urls() -> None:
    assert canonicalize_product_url("/vi-VN/KHCN/SPDV/The", BASE_URL) is None
    assert canonicalize_product_url("https://example.com/vi-VN/KHCN/SPDV/The/Card", BASE_URL) is None


def test_extract_product_catalog_items_from_search_payload_reads_product_cards() -> None:
    payload = {
        "Results": [
            {
                "Html": """
                <article class="product-card">
                  <a href="/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Con-vuon-xa-2">
                    <h3>FWD Con vươn xa 2.0</h3>
                  </a>
                  <span class="tag">Bảo hiểm tiết kiệm</span>
                  <strong>300% số tiền bảo hiểm</strong>
                  <button>Xem chi tiết</button>
                </article>
                """
            },
            {
                "Url": "/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Vung-uoc-mo",
                "Name": "FWD Vững ước mơ",
                "Html": "<a href='/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Vung-uoc-mo'>Xem chi tiết</a>",
            },
        ]
    }

    items = extract_product_catalog_items_from_search_payload(payload, BASE_URL)

    assert [item.title for item in items] == ["FWD Con vươn xa 2.0", "FWD Vững ước mơ"]
    assert items[0].url == f"{BASE_URL}/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Con-vuon-xa-2"
    assert items[0].category == "Bảo hiểm tiết kiệm"
    assert "300% số tiền bảo hiểm" in (items[0].summary or "")
    assert "Xem chi tiết" not in (items[0].summary or "")


def test_raw_document_from_product_catalog_results_marks_catalog_metadata() -> None:
    category_raw = RawDocument(
        source_url=f"{BASE_URL}/vi-VN/KHCN/SPDV/Bao-hiem",
        html="<html><body><h1>Danh sách sản phẩm</h1></body></html>",
        status_code=200,
        content_hash="abc",
    )
    results = [
        {
            "Url": "/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Con-vuon-xa-2",
            "Name": "FWD Con vươn xa 2.0",
        },
        {
            "Url": "/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Vung-uoc-mo",
            "Name": "FWD Vững ước mơ",
        },
    ]

    raw = raw_document_from_product_catalog_results(category_raw, results, BASE_URL)

    assert raw is not None
    assert raw.source_url == category_raw.source_url
    assert raw.metadata["document_type"] == "product_catalog"
    assert raw.metadata["category_title"] == "Bảo hiểm"
    assert raw.metadata["product_type"] == "insurance"
    assert raw.metadata["item_count"] == 2
    assert raw.metadata["items"][0]["title"] == "FWD Con vươn xa 2.0"


def test_raw_documents_from_product_catalog_results_adds_subcategory_catalogs() -> None:
    category_raw = RawDocument(
        source_url=f"{BASE_URL}/vi-VN/KHCN/SPDV/The",
        html="<html><body><h1>Danh sách sản phẩm</h1></body></html>",
        status_code=200,
        content_hash="abc",
    )
    results = [
        {
            "Url": "/vi-VN/KHCN/SPDV/The/Credit_VCB-Vibe",
            "Name": "Vietcombank Vibe",
            "Html": "<a href='/vi-VN/KHCN/SPDV/The/Credit_VCB-Vibe'><h3>Vietcombank Vibe</h3></a>",
        },
        {
            "Url": "/vi-VN/KHCN/SPDV/The/Debit_VCB-Connect24",
            "Name": "Vietcombank Connect24",
            "Html": "<a href='/vi-VN/KHCN/SPDV/The/Debit_VCB-Connect24'><h3>Vietcombank Connect24</h3></a>",
        },
        {
            "Url": "/vi-VN/KHCN/SPDV/The/Tra-gop-linh-hoat",
            "Name": "Dịch vụ trả góp linh hoạt",
            "Html": "<a href='/vi-VN/KHCN/SPDV/The/Tra-gop-linh-hoat'><h3>Dịch vụ trả góp linh hoạt</h3></a>",
        },
    ]

    raws = raw_documents_from_product_catalog_results(category_raw, results, BASE_URL)

    assert [raw.metadata["category_title"] for raw in raws] == [
        "Thẻ",
        "Thẻ tín dụng",
        "Thẻ thanh toán",
        "Dịch vụ thẻ",
    ]
    assert raws[1].metadata["parent_category_title"] == "Thẻ"
    assert (
        raws[2].source_url
        == f"{BASE_URL}/vi-VN/KHCN/SPDV/The#subcategory=Th%E1%BA%BB%20thanh%20to%C3%A1n&card-list_type=Th%E1%BA%BB%20thanh%20to%C3%A1n&e=0"
    )
    assert raws[1].metadata["item_count"] == 1


def test_product_catalog_subcategory_source_url_keeps_site_tab_state() -> None:
    url = product_catalog_subcategory_source_url(
        f"{BASE_URL}/vi-VN/KHCN/SPDV/The",
        "Thẻ thanh toán",
        list_sig="card-list",
    )

    assert url == (
        f"{BASE_URL}/vi-VN/KHCN/SPDV/The#subcategory=Th%E1%BA%BB%20thanh%20to%C3%A1n"
        "&card-list_type=Th%E1%BA%BB%20thanh%20to%C3%A1n&e=0"
    )


def test_infer_product_catalog_item_category_maps_menu_branches() -> None:
    assert (
        infer_product_catalog_item_category(
            "loan",
            f"{BASE_URL}/vi-VN/KHCN/SPDV/Vay/Vay-mua--oto",
            "Vay mua ô tô",
        )
        == "Vay mua ô tô"
    )
    assert (
        infer_product_catalog_item_category(
            "insurance",
            f"{BASE_URL}/vi-VN/KHCN/SPDV/Bao-hiem/Bảo-hiểm-đầu-tư/KHCN---FWD-Dau-tu-don-dau",
            "FWD Đầu tư đón đầu",
        )
        == "Bảo hiểm đầu tư"
    )


def test_extract_linked_resources_from_raw_document_keeps_parent_context() -> None:
    raw = RawDocument(
        source_url=f"{BASE_URL}/vi-VN/KHCN/SPDV/Tiet-kiem/Test",
        html="""
        <html>
          <body>
            <h1>Saving Product</h1>
            <div class="component information-detail-component">
              <h2 class="title">Product information</h2>
              <div class="select-item-wrapper">
                <div class="select-item active">Reference documents</div>
              </div>
              <div class="content-wrapper active">
                <div class="content-item">
                  <p class="name">Interest rate reference</p>
                  <div class="documents">
                    <a href="/-/media/rate.pdf?ts=1">Xem t&#7841;i &#273;&#226;y</a>
                  </div>
                </div>
              </div>
            </div>
          </body>
        </html>
        """,
        status_code=200,
        content_hash="abc",
    )

    resources = extract_linked_resources_from_raw_document(raw, BASE_URL)

    assert len(resources) == 1
    assert resources[0].url == f"{BASE_URL}/-/media/rate.pdf?ts=1"
    assert resources[0].parent_title == "Saving Product"
    assert resources[0].link_label == "Interest rate reference"
    assert resources[0].section_title == "Interest rate reference"
    assert resources[0].tab_title == "Reference documents"


def test_infer_linked_resource_type_prefers_file_extension() -> None:
    assert infer_linked_resource_type(f"{BASE_URL}/-/media/rate.pdf?ts=1", "text/html") == "pdf"
    assert infer_linked_resource_type(f"{BASE_URL}/-/media/rate", "application/pdf") == "pdf"
    assert infer_linked_resource_type(f"{BASE_URL}/vi-VN/KHCN/Cong-cu-Tien-ich/Lai-suat", "text/html") == "html"


def test_extract_faq_topics_from_search_payload_reads_topic_card_links() -> None:
    payload = {
        "Results": [
            {
                "Html": """
                <div class="question-category">
                  <div class="question-category__title"><strong>Ngân hàng số</strong></div>
                  <a class="question-category__button"
                     href="/vi-VN/KHCN/Lien-he-va-Ho-tro/Danh-sach-cau-hoi-theo-chu-de-Ngan-hang-so---Content">
                    Xem chi tiết
                  </a>
                </div>
                """
            }
        ]
    }

    topics = extract_faq_topics_from_search_payload(payload, BASE_URL)

    assert len(topics) == 1
    assert topics[0].title == "Ngân hàng số"
    assert topics[0].url == (
        f"{BASE_URL}/vi-VN/KHCN/Lien-he-va-Ho-tro/"
        "Danh-sach-cau-hoi-theo-chu-de-Ngan-hang-so---Content"
    )

def test_extract_faq_category_labels_from_facet_payload_normalizes_keys() -> None:
    payload = {
        "Facets": [
            {
                "Name": "QuestionType",
                "Values": [
                    {"Name": "Chuy\u1ec3n ti\u1ec1n t\u1eeb Vi\u1ec7t Nam ra n\u01b0\u1edbc ngo\u00e0i"},
                ],
            }
        ]
    }

    labels = extract_faq_category_labels_from_facet_payload(payload)

    assert labels["chuyen tien tu viet nam ra nuoc ngoai"] == (
        "Chuy\u1ec3n ti\u1ec1n t\u1eeb Vi\u1ec7t Nam ra n\u01b0\u1edbc ngo\u00e0i"
    )


def test_faq_facet_config_from_search_config_uses_topic_search_params() -> None:
    search_config = SitecoreSearchConfig(
        endpoint=f"{BASE_URL}/sxa/searchapi/customresults/",
        params={
            "s": "{scope-id}",
            "l": "vi-VN",
            "itemid": "{item-id}",
            "sig": "",
        },
        source_url=f"{BASE_URL}/faq-topic",
    )

    facet_config = faq_facet_config_from_search_config(search_config, BASE_URL)

    assert facet_config is not None
    assert facet_config.endpoint == f"{BASE_URL}/sxa/searchapi/customfacets/"
    assert facet_config.params["f"] == "questiontype"
    assert facet_config.params["s"] == "{scope-id}"
    assert facet_config.params["itemid"] == "{item-id}"


def test_raw_document_from_faq_result_marks_faq_metadata() -> None:
    topic = extract_faq_topics_from_search_payload(
        {
            "Results": [
                {
                    "Html": """
                    <a href="/vi-VN/KHCN/Lien-he-va-Ho-tro/Danh-sach-cau-hoi-theo-chu-de-Ngan-hang-so---Content">
                      Ngân hàng số
                    </a>
                    """
                }
            ]
        },
        BASE_URL,
    )[0]
    result = {
        "Url": "/vi-VN/Data/Question-List/KHCN/Ngan-hang-so---Content/VCB-Digibank/Q1",
        "Html": "<div class='field-heading'>Question</div><div class='field-content'>Answer</div>",
        "Path": "/sitecore/content/faq",
    }

    raw = raw_document_from_faq_result(result, BASE_URL, topic, category_position=11)

    assert raw is not None
    assert raw.metadata["document_type"] == "faq"
    assert raw.metadata["topic"] == "Ngân hàng số"
    assert raw.metadata["category"] == "VCB Digibank"
    assert raw.metadata["category_slug"] == "VCB Digibank"
    assert "/Data/Question-List/" not in raw.source_url
    assert "Danh-sach-cau-hoi-theo-chu-de-Ngan-hang-so---Content#" in raw.source_url
    assert "p=10&e=10" in raw.source_url
    assert "questiontype=VCB%20Digibank" in raw.source_url
    assert "faq=Q1" in raw.source_url
    assert raw.metadata["category_position"] == 11
    assert raw.metadata["category_page_offset"] == 10
    assert raw.metadata["sitecore_item_url"].endswith("/VCB-Digibank/Q1")
    assert faq_category_from_url(raw.metadata["sitecore_item_url"]) == "VCB Digibank"


def test_raw_document_from_faq_result_normalizes_loan_category_slug_for_browser_hash() -> None:
    topic = extract_faq_topics_from_search_payload(
        {
            "Results": [
                {
                    "Html": """
                    <a href="/vi-VN/KHCN/Lien-he-va-Ho-tro/Danh-sach-cau-hoi-theo-chu-de-Vay">
                      Vay
                    </a>
                    """
                }
            ]
        },
        BASE_URL,
    )[0]
    result = {
        "Url": "/vi-VN/Data/Question-List/KHCN/Vay---Content/Xay-moi-co-so-luu-tru/Vay---Xay-moi-co-so-luu-tru---Q5",
        "Html": "<div class='field-heading'>Question</div><div class='field-content'>Answer</div>",
        "Path": "/sitecore/content/faq",
    }

    raw = raw_document_from_faq_result(result, BASE_URL, topic, category_position=5)

    assert raw is not None
    assert raw.metadata["category"] == "Vay xây mới cơ sở lưu trú du lịch"
    assert (
        "questiontype=Vay%20x%C3%A2y%20m%E1%BB%9Bi%20c%C6%A1%20s%E1%BB%9F%20l%C6%B0u%20tr%C3%BA%20du%20l%E1%BB%8Bch"
        in raw.source_url
    )
    assert "faq=Vay---Xay-moi-co-so-luu-tru---Q5" in raw.source_url


def test_normalize_faq_category_label_maps_known_loan_slugs() -> None:
    assert normalize_faq_category_label("Xay moi co so luu tru") == "Vay xây mới cơ sở lưu trú du lịch"
    assert normalize_faq_category_label("Nang cap co so luu tru du lich") == "Vay nâng cấp cơ sở lưu trú du lịch"
