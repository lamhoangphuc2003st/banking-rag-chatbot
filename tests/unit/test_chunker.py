from datetime import UTC, datetime

import pytest

from packages.data_pipeline.chunker import chunk_document, merge_chunk_files
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


def test_merge_chunk_files_validates_and_combines_jsonl(tmp_path) -> None:
    document = NormalizedDocument(
        document_id="doc-1",
        source_url="https://www.vietcombank.com.vn/example",
        title="Product",
        text="Paragraph one. Paragraph two.",
        content_hash="abc",
        crawled_at=datetime.now(UTC),
    )
    chunks = chunk_document(document, max_chars=20, overlap=0)
    first_input = tmp_path / "first.jsonl"
    second_input = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    first_input.write_text(chunks[0].model_dump_json() + "\n", encoding="utf-8")
    second_input.write_text(chunks[1].model_dump_json() + "\n", encoding="utf-8")

    count = merge_chunk_files([first_input, second_input], output)

    assert count == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_merge_chunk_files_rejects_duplicate_chunk_ids(tmp_path) -> None:
    document = NormalizedDocument(
        document_id="doc-1",
        source_url="https://www.vietcombank.com.vn/example",
        title="Product",
        text="Paragraph one. Paragraph two.",
        content_hash="abc",
        crawled_at=datetime.now(UTC),
    )
    chunk = chunk_document(document, max_chars=200, overlap=0)[0]
    first_input = tmp_path / "first.jsonl"
    second_input = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    first_input.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")
    second_input.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        merge_chunk_files([first_input, second_input], output)
