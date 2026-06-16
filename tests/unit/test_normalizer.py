from datetime import UTC, datetime

from packages.data_pipeline.normalizer import infer_product_type, normalize_raw_document
from packages.shared.schemas import RawDocument


def test_normalize_raw_document_keeps_product_content_and_drops_page_chrome() -> None:
    raw = RawDocument(
        source_url="https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/The/Credit-Test",
        html="""
        <html>
          <head><title>Thẻ kiểm thử</title></head>
          <body>
            <header>Cá nhân Tổ chức Khách hàng Ưu tiên</header>
            <div class="accessbility">
              <svg><path d="M0 0"></path></svg>
            </div>
            <div class="component page-content hero-container col-12">
              <h1>Thẻ kiểm thử</h1>
              <p>Hoàn tiền đến 5% cho giao dịch hợp lệ.</p>
            </div>
            <div class="component information-detail-component col-12">
              <h2>Thông tin chi tiết</h2>
              <h3>Điều kiện mở thẻ</h3>
              <p>Khách hàng từ đủ 18 tuổi và đáp ứng quy định của Vietcombank.</p>
            </div>
            <div class="related-search-results">Có thể bạn quan tâm</div>
            <footer>Bản quyền thuộc về Ngân hàng TMCP Ngoại thương Việt Nam</footer>
            <div>Chúng tôi sử dụng cookie để phục vụ tốt hơn.</div>
          </body>
        </html>
        """,
        status_code=200,
        content_hash="abc",
        crawled_at=datetime.now(UTC),
    )

    normalized = normalize_raw_document(raw)

    assert normalized is not None
    assert "Hoàn tiền đến 5%" in normalized.text
    assert "Điều kiện mở thẻ" in normalized.text
    assert "Cá nhân Tổ chức" not in normalized.text
    assert "Bản quyền thuộc về" not in normalized.text
    assert "Có thể bạn quan tâm" not in normalized.text
    assert "Chúng tôi sử dụng cookie" not in normalized.text
    assert "<svg" not in normalized.text
    assert normalized.product_type == "card"
    assert normalized.section == "product_detail"


def test_infer_product_type_uses_product_url_before_text_noise() -> None:
    text = "Cá nhân Sản phẩm & Dịch vụ Thẻ Vay Tiết kiệm Bảo hiểm"

    assert (
        infer_product_type(
            "https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Dich-vu-tai-khoan/Tai-khoan-so-dep",
            text,
        )
        == "account"
    )
    assert (
        infer_product_type(
            "https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Chuyen-va-nhan-tien/Nhan-kieu-hoi",
            text,
        )
        == "transfer"
    )
    assert (
        infer_product_type(
            "https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Bao-hiem",
            text,
        )
        == "insurance"
    )


def test_normalize_product_catalog_raw_document_builds_complete_catalog() -> None:
    raw = RawDocument(
        source_url="https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Bao-hiem",
        html="<html><body><h1>Danh sách gói Bảo hiểm Vietcombank</h1></body></html>",
        status_code=200,
        content_hash="abc",
        crawled_at=datetime.now(UTC),
        metadata={
            "document_type": "product_catalog",
            "category_url": "https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Bao-hiem",
            "category_title": "Bảo hiểm",
            "product_type": "insurance",
            "item_count": 3,
            "items": [
                {
                    "title": "FWD Con vươn xa 2.0",
                    "url": "https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Con-vuon-xa-2",
                    "summary": "300% số tiền bảo hiểm",
                    "category": "Bảo hiểm tiết kiệm",
                },
                {
                    "title": "FWD Vững ước mơ",
                    "url": "https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Vung-uoc-mo",
                    "summary": "Thanh toán khoản vay",
                    "category": "Bảo hiểm bảo vệ",
                },
                {
                    "title": "FWD Cả nhà vui khỏe",
                    "url": "https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Bao-hiem/FWD-Ca-nha-vui-khoe",
                    "summary": "Bảo vệ sức khỏe gia đình",
                    "category": "Bảo hiểm bảo vệ",
                },
            ],
        },
    )

    normalized = normalize_raw_document(raw)

    assert normalized is not None
    assert normalized.title == "Danh sách gói Bảo hiểm Vietcombank"
    assert normalized.section == "product_catalog"
    assert normalized.product_type == "insurance"
    assert "Danh sách sản phẩm/gói Bảo hiểm Vietcombank" in normalized.text
    assert "1. FWD Con vươn xa 2.0" in normalized.text
    assert "2. FWD Vững ước mơ" in normalized.text
    assert "3. FWD Cả nhà vui khỏe" in normalized.text
    assert normalized.metadata["item_count"] == 3


def test_normalize_raw_document_preserves_tab_context_and_guide_blocks() -> None:
    raw = RawDocument(
        source_url="https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Dau-tu/Structured-Test",
        html="""
        <html>
          <head><title>Structured Product</title></head>
          <body>
            <div class="component page-content hero-container col-12">
              <h1>Structured Product</h1>
              <p>Main product summary with enough content for normalization.</p>
            </div>
            <div class="component information-detail-component">
              <h2 class="title">Service information</h2>
              <div class="select-item-wrapper">
                <div class="select-item active">Margin</div>
                <div class="select-item">Cash advance</div>
              </div>
              <div class="content-wrapper active">
                <div class="content-item">
                  <p class="name">Overview</p>
                  <div class="documents">
                    <p>Margin overview belongs only to the margin tab.</p>
                  </div>
                </div>
              </div>
              <div class="content-wrapper">
                <div class="content-item">
                  <p class="name">Overview</p>
                  <div class="documents">
                    <p>Cash advance overview belongs only to the cash advance tab.</p>
                  </div>
                </div>
              </div>
            </div>
            <div class="component forms-of-investment col-12">
              <h2>Investment guide</h2>
              <p>Step one: open an account.</p>
              <p>Step two: place a transaction.</p>
            </div>
          </body>
        </html>
        """,
        status_code=200,
        content_hash="abc",
        crawled_at=datetime.now(UTC),
    )

    normalized = normalize_raw_document(raw)

    assert normalized is not None
    assert "[Product]" in normalized.text
    assert "[Tab]\nMargin" in normalized.text
    assert "[Tab]\nCash advance" in normalized.text
    assert normalized.text.index("[Tab]\nMargin") < normalized.text.index("Margin overview")
    assert normalized.text.index("Margin overview") < normalized.text.index("[Tab]\nCash advance")
    assert normalized.text.index("[Tab]\nCash advance") < normalized.text.index("Cash advance overview")
    assert "[Guide]\nInvestment guide" in normalized.text
    assert "Step two: place a transaction." in normalized.text


def test_normalize_raw_document_preserves_reference_links() -> None:
    raw = RawDocument(
        source_url="https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Tiet-kiem/Test",
        html="""
        <html>
          <body>
            <div class="component page-content hero-container col-12">
              <h1>Saving Product</h1>
              <p>Main product summary with enough content for normalization.</p>
            </div>
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
        crawled_at=datetime.now(UTC),
    )

    normalized = normalize_raw_document(raw)

    assert normalized is not None
    assert "[Linked Resource]" in normalized.text
    assert "[Section]\nInterest rate reference\n[Linked Resource]\nURL: https://www.vietcombank.com.vn/-/media/rate.pdf?ts=1" in normalized.text
    assert "Interest rate reference: https://www.vietcombank.com.vn/-/media/rate.pdf?ts=1" not in normalized.text


def test_normalize_linked_resource_raw_document_keeps_parent_metadata() -> None:
    raw = RawDocument(
        source_url="https://www.vietcombank.com.vn/-/media/rate.pdf?ts=1",
        html="""
        <html>
          <body>
            <h1>Interest rate reference</h1>
            <p>Resource URL: https://www.vietcombank.com.vn/-/media/rate.pdf?ts=1</p>
          </body>
        </html>
        """,
        status_code=200,
        content_hash="abc",
        crawled_at=datetime.now(UTC),
        metadata={
            "document_type": "linked_resource",
            "parent_url": "https://www.vietcombank.com.vn/vi-VN/KHCN/SPDV/Tiet-kiem/Test",
            "parent_title": "Saving Product",
            "link_text": "Xem t\u1ea1i \u0111\u00e2y",
            "link_label": "Interest rate reference",
            "section_title": "Interest rate reference",
            "tab_title": "Reference documents",
            "resource_type": "pdf",
            "content_type": "application/pdf",
            "file_name": "rate.pdf",
            "text_extraction_status": "metadata_only",
        },
    )

    normalized = normalize_raw_document(raw)

    assert normalized is not None
    assert normalized.section == "linked_resource"
    assert normalized.product_type == "saving"
    assert "Product: Saving Product" in normalized.text
    assert "Section: Interest rate reference" in normalized.text
    assert "Label: Interest rate reference" not in normalized.text
    assert "Resource type: pdf" in normalized.text
    assert normalized.metadata["parent_title"] == "Saving Product"
    assert normalized.metadata["text_extraction_status"] == "metadata_only"


def test_normalize_faq_raw_document_extracts_question_and_answer() -> None:
    raw = RawDocument(
        source_url=(
            "https://www.vietcombank.com.vn/vi-VN/Data/Question-List/KHCN/"
            "Ngan-hang-so---Content/VCB-Digibank/Q1"
        ),
        html="""
        <div class="component accordion col-xs-12 detail-faq">
          <div class="field-heading">
            Who can use VCB Digibank?
          </div>
          <div class="field-content">
            Customers with a Vietcombank current account can register and use the service.
          </div>
        </div>
        """,
        status_code=200,
        content_hash="abc",
        crawled_at=datetime.now(UTC),
        metadata={
            "document_type": "faq",
            "topic": "Ngân hàng số",
            "category": "VCB Digibank",
            "category_slug": "VCB Digibank",
            "topic_url": "https://www.vietcombank.com.vn/faq-topic",
            "category_position": 11,
            "category_page_offset": 10,
            "sitecore_item_url": (
                "https://www.vietcombank.com.vn/vi-VN/Data/Question-List/KHCN/"
                "Ngan-hang-so---Content/VCB-Digibank/Q1"
            ),
            "path": "/sitecore/content/faq",
        },
    )

    normalized = normalize_raw_document(raw)

    assert normalized is not None
    assert normalized.title == "Who can use VCB Digibank?"
    assert normalized.section == "faq"
    assert normalized.product_type == "digital_banking"
    assert "[FAQ]" in normalized.text
    assert "Topic: Ngân hàng số" in normalized.text
    assert "Category: VCB Digibank" in normalized.text
    assert "Question: Who can use VCB Digibank?" in normalized.text
    assert "Customers with a Vietcombank current account" in normalized.text
    assert normalized.metadata["category_slug"] == "VCB Digibank"
    assert normalized.metadata["category_position"] == 11
    assert normalized.metadata["category_page_offset"] == 10
    assert normalized.metadata["sitecore_item_url"].endswith("/VCB-Digibank/Q1")
