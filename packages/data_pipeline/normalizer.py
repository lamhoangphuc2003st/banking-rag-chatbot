from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from bs4.element import Tag

from packages.shared.schemas import NormalizedDocument, RawDocument

MAIN_CONTENT_SELECTORS = [
    ".hero-container",
    ".benefit",
    ".information-detail-component",
    ".register-online-accounts",
    ".forms-of-investment",
    ".accordion.detail-faq",
]

NOISE_SELECTORS = [
    "header",
    "footer",
    ".footer",
    ".mega-menu",
    ".megamenu",
    ".accessbility",
    ".modal",
    ".cookie",
    ".related-search-results",
    ".newest-promotion",
    ".similar-cards",
    ".compare-cards",
    ".calculation-tool",
    ".repayment-calculation",
    ".repayment-calculator",
    ".repayment-table",
    ".advertisement",
]

DROP_TEXT_LINES = {
    "Previous",
    "Next",
    "Mở rộng",
    "Thu gọn",
    "Xem thêm",
    "Xem chi tiết",
    "Xem tất cả",
    "Đăng ký ngay",
    "Đặt lịch hẹn",
    "Đặt lịch tư vấn",
    "Mở thẻ ngay",
    "So sánh thẻ",
    "Từ chối",
    "Chấp nhận",
    "Tìm hiểu thêm",
    "Xem tại đây",
}


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
    if raw.metadata.get("document_type") == "linked_resource":
        return normalize_linked_resource_raw_document(raw)
    if raw.metadata.get("document_type") == "faq":
        return normalize_faq_raw_document(raw)
    if raw.metadata.get("document_type") == "product_catalog":
        return normalize_product_catalog_raw_document(raw)

    soup = BeautifulSoup(raw.html, "html.parser")
    _remove_noise(soup)

    title = _clean_text(_first_text([soup.find("h1"), soup.find("title")]))
    text = extract_main_text(soup, raw.source_url)
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


def normalize_product_catalog_raw_document(raw: RawDocument) -> NormalizedDocument | None:
    metadata = raw.metadata
    items = [item for item in metadata.get("items") or [] if isinstance(item, dict)]
    if not items:
        return None

    category_title = str(metadata.get("category_title") or "").strip() or "Sản phẩm"
    parent_category_title = str(metadata.get("parent_category_title") or "").strip()
    title = f"Danh sách gói {category_title} Vietcombank"
    product_type = str(metadata.get("product_type") or "").strip() or infer_product_type(
        raw.source_url,
        category_title,
    )
    product_names = [
        str(item.get("title") or "").strip()
        for item in items
        if str(item.get("title") or "").strip()
    ]

    lines = [
        "[Product Catalog]",
        f"Category: {category_title}",
        f"Source: {raw.source_url}",
        f"Danh sách sản phẩm/gói {category_title} Vietcombank:",
    ]
    if parent_category_title:
        lines.insert(2, f"Parent category: {parent_category_title}")
    if product_names:
        lines.append("; ".join(f"{index}. {name}" for index, name in enumerate(product_names, start=1)))
        lines.append("Chi tiết từng sản phẩm:")

    for index, item in enumerate(items, start=1):
        product_title = str(item.get("title") or "").strip()
        product_url = str(item.get("url") or "").strip()
        summary = str(item.get("summary") or "").strip()
        category = str(item.get("category") or "").strip()
        if not product_title:
            continue

        lines.append(f"{index}. {product_title}")
        if category:
            lines.append(f"Nhóm: {category}")
        if summary:
            lines.append(f"Tóm tắt: {summary}")
        if product_url:
            lines.append(f"URL: {product_url}")

    text = _clean_document_text("\n".join(lines))
    if len(text) < 80:
        return None

    return NormalizedDocument(
        document_id=stable_document_id(f"{raw.source_url}|product_catalog"),
        source_url=raw.source_url,
        title=title,
        text=text,
        content_hash=raw.content_hash,
        crawled_at=raw.crawled_at,
        product_type=product_type or None,
        section="product_catalog",
        metadata={
            "status_code": raw.status_code,
            "document_type": "product_catalog",
            "category_url": metadata.get("category_url") or raw.source_url,
            "category_title": category_title,
            "parent_category_title": parent_category_title or None,
            "item_count": len(items),
            "items": items,
        },
    )


def normalize_linked_resource_raw_document(raw: RawDocument) -> NormalizedDocument | None:
    metadata = raw.metadata
    resource_type = str(metadata.get("resource_type") or "").strip() or "resource"
    parent_url = str(metadata.get("parent_url") or "").strip()
    parent_title = str(metadata.get("parent_title") or "").strip()
    link_label = str(metadata.get("link_label") or "").strip()
    tab_title = str(metadata.get("tab_title") or "").strip()
    section_title = str(metadata.get("section_title") or "").strip()

    soup = BeautifulSoup(raw.html, "html.parser")
    _remove_noise(soup)
    html_title = _clean_text(_first_text([soup.find("h1"), soup.find("title")]))
    title = link_label or html_title or raw.source_url

    body = ""
    if resource_type == "html":
        body = extract_main_text(soup, raw.source_url)
        if body == html_title:
            body = ""

    lines = ["[Linked Resource]"]
    if parent_title:
        lines.append(f"Product: {parent_title}")
    if tab_title:
        lines.append(f"Tab: {tab_title}")
    if section_title:
        lines.append(f"Section: {section_title}")
    if link_label and not _same_text_label(link_label, section_title):
        lines.append(f"Label: {link_label}")
    lines.extend(
        [
            f"Resource type: {resource_type}",
            f"URL: {raw.source_url}",
        ]
    )
    if body:
        lines.extend(["Content:", body])
    else:
        lines.append("Content: Binary file discovered from the product page; text extraction is not enabled yet.")

    text = _clean_document_text("\n".join(lines))
    if len(text) < 40:
        return None

    return NormalizedDocument(
        document_id=stable_document_id(f"{parent_url}|{raw.source_url}|{link_label}"),
        source_url=raw.source_url,
        title=title,
        text=text,
        content_hash=raw.content_hash,
        crawled_at=raw.crawled_at,
        product_type=infer_product_type(parent_url, text),
        section="linked_resource",
        metadata={
            "status_code": raw.status_code,
            "document_type": "linked_resource",
            "parent_url": parent_url or None,
            "parent_title": parent_title or None,
            "link_text": metadata.get("link_text"),
            "link_label": link_label or None,
            "section_title": section_title or None,
            "tab_title": tab_title or None,
            "resource_type": resource_type,
            "content_type": metadata.get("content_type"),
            "file_name": metadata.get("file_name"),
            "text_extraction_status": metadata.get("text_extraction_status"),
        },
    )


def normalize_faq_raw_document(raw: RawDocument) -> NormalizedDocument | None:
    soup = BeautifulSoup(raw.html, "html.parser")
    _remove_noise(soup)

    question = _clean_document_text(
        _first_text(
            [
                soup.select_one(".field-heading"),
                soup.select_one(".header-item"),
            ]
        )
    )
    answer = _clean_document_text(
        _first_text(
            [
                soup.select_one(".field-content"),
                soup.select_one(".answer-content"),
            ]
        )
    )
    if len(question) < 8 or len(answer) < 8:
        return None

    topic = str(raw.metadata.get("topic") or "").strip()
    category = str(raw.metadata.get("category") or "").strip()
    text_lines = ["[FAQ]"]
    if topic:
        text_lines.append(f"Topic: {topic}")
    if category:
        text_lines.append(f"Category: {category}")
    text_lines.extend([f"Question: {question}", "Answer:", answer])
    text = _clean_document_text("\n".join(text_lines))

    return NormalizedDocument(
        document_id=stable_document_id(raw.source_url),
        source_url=raw.source_url,
        title=question,
        text=text,
        content_hash=raw.content_hash,
        crawled_at=raw.crawled_at,
        product_type=infer_faq_product_type(topic, category),
        section="faq",
        metadata={
            "status_code": raw.status_code,
            "document_type": "faq",
            "topic": topic or None,
            "category": category or None,
            "category_slug": raw.metadata.get("category_slug"),
            "topic_url": raw.metadata.get("topic_url"),
            "category_position": raw.metadata.get("category_position"),
            "category_page_offset": raw.metadata.get("category_page_offset"),
            "sitecore_item_url": raw.metadata.get("sitecore_item_url"),
            "path": raw.metadata.get("path"),
        },
    )


def normalize_documents(raw_docs: Iterable[RawDocument]) -> list[NormalizedDocument]:
    return [doc for raw in raw_docs if (doc := normalize_raw_document(raw)) is not None]


def stable_document_id(source_url: str) -> str:
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:24]


def infer_product_type(url: str, text: str) -> str | None:
    path = urlparse(url).path.lower()
    path_segments = {part for part in path.strip("/").split("/") if part}
    lowered = f"{path} {text[:1000]}".lower()
    if "dich-vu-tai-khoan" in path_segments:
        return "account"
    if "ngan-hang-so" in path_segments:
        return "digital_banking"
    if "the" in path_segments:
        return "card"
    if "vay" in path_segments:
        return "loan"
    if "tiet-kiem" in path_segments:
        return "saving"
    if "bao-hiem" in path_segments:
        return "insurance"
    if "dau-tu" in path_segments:
        return "investment"
    if "chuyen-va-nhan-tien" in path_segments:
        return "transfer"

    if "credit_" in lowered or "debit_" in lowered:
        return "card"
    if "vay" in lowered or "loan" in lowered:
        return "loan"
    if "tiết kiệm" in lowered:
        return "saving"
    if "bảo hiểm" in lowered:
        return "insurance"
    if "đầu tư" in lowered:
        return "investment"
    if "chuyển và nhận tiền" in lowered:
        return "transfer"
    if "ngân hàng số" in lowered:
        return "digital_banking"
    if "tài khoản" in lowered:
        return "account"
    return None


def infer_faq_product_type(topic: str, category: str) -> str | None:
    lowered = f"{topic} {category}".lower()
    if "ngân hàng số" in lowered or "digibank" in lowered or "sms banking" in lowered:
        return "digital_banking"
    if "loyalty" in lowered:
        return "loyalty"
    if "thẻ" in lowered:
        return "card"
    if "vay" in lowered:
        return "loan"
    if "chuyển" in lowered or "nhận tiền" in lowered:
        return "transfer"
    return None


def infer_section(url: str, title: str, text: str) -> str | None:
    path = urlparse(url).path.lower()
    if _is_product_detail_path(path):
        return "product_detail"

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


def extract_main_text(soup: BeautifulSoup, source_url: str = "") -> str:
    blocks = []
    seen_elements: set[int] = set()

    for element in soup.select(".component"):
        element_id = id(element)
        if element_id in seen_elements:
            continue

        block = _extract_component_text(element, source_url)
        if not block:
            continue
        seen_elements.add(element_id)
        blocks.append(block)

    if not blocks:
        blocks.append(_clean_extracted_text(soup.get_text("\n", strip=True)))

    return _clean_document_text("\n\n".join(blocks))


def _extract_component_text(element: Tag, source_url: str) -> str:
    raw_classes = element.get("class")
    if isinstance(raw_classes, str):
        classes = {raw_classes}
    elif raw_classes is None:
        classes = set()
    else:
        classes = {str(class_name) for class_name in raw_classes}

    if "hero-container" in classes:
        return _format_block("Product", _clean_extracted_text(element.get_text("\n", strip=True)))
    if "benefit" in classes:
        return _format_block("Highlights", _clean_extracted_text(element.get_text("\n", strip=True)))
    if "information-detail-component" in classes:
        return _extract_information_detail_text(element, source_url)
    if "register-online-accounts" in classes or "forms-of-investment" in classes:
        return _format_block("Guide", _clean_extracted_text(element.get_text("\n", strip=True)))
    if "accordion" in classes and "detail-faq" in classes:
        return _format_block("FAQ", _clean_extracted_text(element.get_text("\n", strip=True)))

    return ""


def _extract_information_detail_text(component: Tag, source_url: str) -> str:
    heading = _clean_text(_first_text([component.select_one("h2.title"), component.select_one("h2")]))
    tab_titles = [
        text
        for tab in component.select(".select-item-wrapper .select-item")
        if (text := _clean_text(tab.get_text(" ", strip=True)))
    ]
    wrappers = component.select(".content-wrapper")

    lines = []
    if heading:
        lines.extend(["[Section]", heading])

    if tab_titles and wrappers:
        for index, wrapper in enumerate(wrappers):
            tab_title = tab_titles[index] if index < len(tab_titles) else f"Tab {index + 1}"
            tab_body = _extract_tab_body(wrapper, source_url)
            if tab_body:
                lines.extend(["", "[Tab]", tab_title, tab_body])
        return _clean_document_text("\n".join(lines))

    fallback = _clean_extracted_text(component.get_text("\n", strip=True))
    return _format_block("Section", fallback)


def _extract_tab_body(wrapper: Tag, source_url: str) -> str:
    lines = []
    for item in wrapper.find_all("div", class_="content-item", recursive=False):
        section_title = _clean_text(_first_text([item.select_one(".name")]))
        body_source = item.select_one(".documents") or item.select_one(".label") or item
        body = _clean_extracted_text(body_source.get_text("\n", strip=True))
        link_lines = _extract_link_reference_lines(item, source_url, fallback_label=section_title)
        if not body:
            body = ""
        if section_title:
            lines.extend(["[Section]", section_title])
        if body:
            lines.append(body)
        lines.extend(link_lines)
    return _clean_document_text("\n".join(lines))


def _extract_link_reference_lines(
    container: Tag,
    source_url: str,
    *,
    fallback_label: str | None = None,
) -> list[str]:
    lines = []
    seen: set[str] = set()
    for anchor in container.find_all("a", href=True):
        href = str(anchor["href"])
        absolute_url = urljoin(source_url, href)
        anchor_text = _clean_text(anchor.get_text(" ", strip=True))
        parent = anchor.find_parent()
        context = _clean_text(parent.get_text(" ", strip=True)) if parent else ""
        if not _is_reference_link_text(anchor_text):
            continue
        label = (
            _strip_trailing_reference_word(_clean_link_label(context.replace(anchor_text, " ")))
            or fallback_label
            or anchor_text
            or "Linked resource"
        )
        if _is_reference_only_label(label) and fallback_label:
            label = fallback_label
        if _same_text_label(label, fallback_label):
            line = f"[Linked Resource]\nURL: {absolute_url}"
        else:
            line = f"[Linked Resource]\n{label}: {absolute_url}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _format_block(label: str, text: str) -> str:
    if not text:
        return ""
    return _clean_document_text(f"[{label}]\n{text}")


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()


def _clean_extracted_text(text: str) -> str:
    text = re.sub(r"<svg\b.*?</svg>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<path\b.*?</path>", " ", text, flags=re.IGNORECASE | re.DOTALL)

    cleaned_lines: list[str] = []
    seen_lines: set[str] = set()
    previous_line = ""
    for raw_line in text.splitlines():
        line = _clean_text(raw_line)
        if not line or line in DROP_TEXT_LINES or line == previous_line or line in seen_lines:
            continue
        if line.startswith("{") and line.endswith("}"):
            continue
        cleaned_lines.append(line)
        seen_lines.add(line)
        previous_line = line

    return "\n".join(cleaned_lines)


def _is_reference_link_text(text: str) -> bool:
    return "tai day" in _normalize_text_key(text)


def _clean_link_label(label: str) -> str:
    label = label.replace("Xem tại đây", " ")
    label = label.replace("xem tại đây", " ")
    label = label.replace("tại đây", " ")
    label = label.replace("Tại đây", " ")
    return _clean_text(label.strip(" :-,.;()"))


def _strip_trailing_reference_word(label: str) -> str:
    words = label.split()
    if words and _normalize_text_key(words[-1]) == "xem":
        return " ".join(words[:-1])
    return label


def _is_reference_only_label(label: str) -> bool:
    key = _normalize_text_key(label)
    compact = key.replace(" ", "")
    return key in {"xem", "xem tai day", "tai day"} or compact in {"xem", "xemtaiday", "taiday"}


def _same_text_label(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return _normalize_text_key(left) == _normalize_text_key(right)


def _normalize_text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(no_accents.split()).casefold()


def _clean_document_text(text: str) -> str:
    lines = [_clean_text(line) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_product_detail_path(path: str) -> bool:
    parts = [part for part in path.strip("/").split("/") if part]
    try:
        spdv_index = parts.index("spdv")
    except ValueError:
        return False
    return len(parts) > spdv_index + 2


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
