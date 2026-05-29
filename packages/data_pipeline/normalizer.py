from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

from bs4 import BeautifulSoup

from packages.shared.schemas import NormalizedDocument, RawDocument


def normalize_file(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            raw = RawDocument.model_validate_json(line)
            normalized = normalize_raw_document(raw)
            if normalized is None:
                continue
            target.write(normalized.model_dump_json() + "\n")
            count += 1
    return count


def normalize_raw_document(raw: RawDocument) -> NormalizedDocument | None:
    soup = BeautifulSoup(raw.html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = _clean_text(_first_text([soup.find("h1"), soup.find("title")]))
    text = _clean_text(soup.get_text(" "))
    if len(text) < 120 or not title:
        return None

    document_id = stable_document_id(raw.source_url)
    return NormalizedDocument(
        document_id=document_id,
        source_url=raw.source_url,
        title=title,
        text=text,
        content_hash=raw.content_hash,
        crawled_at=raw.crawled_at,
        product_type=infer_product_type(raw.source_url, text),
        section=infer_section(raw.source_url, title, text),
        metadata={"status_code": raw.status_code},
    )


def normalize_documents(raw_docs: Iterable[RawDocument]) -> list[NormalizedDocument]:
    return [doc for raw in raw_docs if (doc := normalize_raw_document(raw)) is not None]


def stable_document_id(source_url: str) -> str:
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:24]


def infer_product_type(url: str, text: str) -> str | None:
    lowered = f"{url} {text[:1000]}".lower()
    if "thẻ" in lowered or "the-" in lowered or "credit" in lowered:
        return "card"
    if "vay" in lowered or "loan" in lowered:
        return "loan"
    if "biểu phí" in lowered or "bieu-phi" in lowered:
        return "fee"
    return None


def infer_section(url: str, title: str, text: str) -> str | None:
    lowered = f"{url} {title} {text[:1000]}".lower()
    if "điều kiện" in lowered or "dieu-kien" in lowered:
        return "condition"
    if "biểu phí" in lowered or "bieu-phi" in lowered:
        return "fee"
    if "lãi suất" in lowered or "lai-suat" in lowered:
        return "interest_rate"
    if "hồ sơ" in lowered or "ho-so" in lowered:
        return "documents"
    if "faq" in lowered or "câu hỏi" in lowered:
        return "faq"
    return "overview"


def _first_text(nodes: list[object | None]) -> str:
    for node in nodes:
        if node is not None:
            return str(getattr(node, "get_text", lambda *_: "")(" "))
    return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_jsonl(path: Path) -> list[NormalizedDocument]:
    docs: list[NormalizedDocument] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                docs.append(NormalizedDocument.model_validate(json.loads(line)))
    return docs
