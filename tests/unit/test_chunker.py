from datetime import UTC, datetime

from packages.data_pipeline.chunker import chunk_document
from packages.shared.schemas import NormalizedDocument


def test_chunk_document_preserves_source_metadata() -> None:
    document = NormalizedDocument(
        document_id="doc-1",
        source_url="https://www.vietcombank.com.vn/example",
        title="Sản phẩm vay",
        text="Đoạn một. Đoạn hai. Đoạn ba.",
        content_hash="abc",
        crawled_at=datetime.now(UTC),
        product_type="loan",
        section="overview",
    )

    chunks = chunk_document(document, max_chars=20, overlap=5)

    assert chunks
    assert chunks[0].source_url == document.source_url
    assert chunks[0].product_type == "loan"
