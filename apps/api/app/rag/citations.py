from __future__ import annotations

import re
import unicodedata
from typing import Any
from urllib.parse import quote, urldefrag

from apps.api.app.models.chat import SourceCitation
from packages.shared.schemas import RetrievedChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
FAQ_CATEGORY_LABEL_OVERRIDES = {
    "cho vay giay to co gia": "Vay cầm cố giấy tờ có giá",
    "nang cap co so luu tru du lich": "Vay nâng cấp cơ sở lưu trú du lịch",
    "o to": "Vay mua ô tô",
    "tin chap nguoi lao dong": "Vay tín chấp đối với Người lao động",
    "vay mua nha dat": "Vay mua nhà ở, đất ở",
    "xay moi co so luu tru": "Vay xây mới cơ sở lưu trú du lịch",
}
CITATION_STOPWORDS = {
    "a",
    "anh",
    "bao",
    "bi",
    "cac",
    "can",
    "cho",
    "co",
    "cua",
    "duoc",
    "gi",
    "hang",
    "hay",
    "khach",
    "khong",
    "la",
    "lam",
    "nhu",
    "ngan",
    "o",
    "phai",
    "qua",
    "ra",
    "sao",
    "tai",
    "the",
    "thi",
    "toi",
    "trong",
    "va",
    "vcb",
    "vietcombank",
}


def build_citations(
    chunks: list[RetrievedChunk],
    *,
    query: str = "",
    answer: str = "",
    max_sources: int = 30,
    include_grouped_catalog_items: bool = False,
) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    seen_urls: set[str] = set()
    query_tokens = _tokenize(query)
    normalized_query = _normalize_text_key(query)
    normalized_answer = _normalize_text_key(answer)
    anchor_title_tokens = _tokenize(chunks[0].title) if chunks else set()
    use_strict_title = _should_use_strict_title_filter(
        query_tokens,
        anchor_title_tokens,
        normalized_query=normalized_query,
    )
    filter_detail_chunks_by_answer = (
        bool(normalized_answer)
        and any(
            _is_answer_scoped_detail_chunk(chunk)
            and _answer_mentions_title(normalized_answer, chunk.title)
            for chunk in chunks
        )
    )
    filter_catalog_chunks_by_answer = (
        bool(normalized_answer)
        and (
            filter_detail_chunks_by_answer
            or any(
                _is_product_catalog_chunk(chunk)
                and _product_catalog_answer_citations(
                    chunk,
                    normalized_answer=normalized_answer,
                )
                for chunk in chunks
            )
        )
    )

    for chunk in chunks:
        if (
            filter_catalog_chunks_by_answer
            and _is_product_catalog_chunk(chunk)
            and not _product_catalog_answer_citations(
                chunk,
                normalized_answer=normalized_answer,
            )
        ):
            continue
        if (
            filter_detail_chunks_by_answer
            and _is_answer_scoped_detail_chunk(chunk)
            and not _answer_mentions_title(normalized_answer, chunk.title)
        ):
            continue
        if (
            not _is_product_catalog_chunk(chunk)
            and query_tokens
            and use_strict_title
            and _tokenize(chunk.title) != anchor_title_tokens
        ):
            continue
        if query_tokens and citations and not _is_relevant_citation(query_tokens, chunk):
            continue

        for citation in _candidate_citations(
            chunk,
            query_tokens=query_tokens,
            normalized_answer=normalized_answer,
            include_grouped_catalog_items=include_grouped_catalog_items,
        ):
            source_key = citation.source_url or citation.chunk_id
            if source_key in seen_urls:
                continue

            seen_urls.add(source_key)
            citations.append(citation)
            if len(citations) >= max_sources:
                break
        if len(citations) >= max_sources:
            break

    return citations


def _candidate_citations(
    chunk: RetrievedChunk,
    *,
    query_tokens: set[str],
    normalized_answer: str,
    include_grouped_catalog_items: bool,
) -> list[SourceCitation]:
    if _is_product_catalog_chunk(chunk):
        answer_citations = _product_catalog_answer_citations(
            chunk,
            normalized_answer=normalized_answer,
        )
        if answer_citations:
            return answer_citations

        exact_item_citations = _product_catalog_exact_title_citations(chunk, query_tokens=query_tokens)
        if exact_item_citations:
            return exact_item_citations

        targeted_group = _targeted_catalog_group(chunk, query_tokens=query_tokens)
        if targeted_group:
            return [
                _catalog_group_citation(chunk, targeted_group, index=0),
                *_product_catalog_item_citations(
                    chunk,
                    query_tokens=query_tokens,
                    category_filter=targeted_group,
                ),
            ]

        group_citations = _product_catalog_group_citations(chunk, query_tokens=query_tokens)
        if group_citations:
            if include_grouped_catalog_items:
                item_citations = _product_catalog_item_citations(
                    chunk,
                    query_tokens=query_tokens,
                )
                return [*item_citations, *group_citations] if item_citations else group_citations
            return group_citations

        item_citations = _product_catalog_item_citations(chunk, query_tokens=query_tokens)
        if item_citations:
            return item_citations

    return [
        SourceCitation(
            chunk_id=chunk.chunk_id,
            title=_source_title_for_chunk(chunk),
            source_url=_source_url_for_chunk(chunk),
            section=chunk.section,
            score=chunk.score,
        )
    ]


def _source_title_for_chunk(chunk: RetrievedChunk) -> str:
    if chunk.section != "faq":
        return chunk.title

    category = _faq_display_category(chunk)
    if not category:
        return chunk.title
    if _normalize_text_key(category) in _normalize_text_key(chunk.title):
        return chunk.title
    return f"{category}: {chunk.title}"


def _source_url_for_chunk(chunk: RetrievedChunk) -> str:
    if chunk.section == "faq":
        return _faq_source_url_for_chunk(chunk)
    return chunk.source_url


def _faq_source_url_for_chunk(chunk: RetrievedChunk) -> str:
    category = _faq_display_category(chunk)
    if not category:
        return chunk.source_url

    base_url, fragment = urldefrag(chunk.source_url)
    if "comp=faq_ls_tp" not in fragment and "questiontype=" not in fragment:
        return chunk.source_url

    questiontype_part = f"questiontype={quote(category, safe='')}"
    parts: list[str] = []
    replaced = False
    for part in fragment.split("&"):
        if not part:
            continue
        if part.split("=", 1)[0] == "questiontype":
            parts.append(questiontype_part)
            replaced = True
        else:
            parts.append(part)
    if not replaced:
        insert_at = next((index for index, part in enumerate(parts) if part.startswith("comp=")), len(parts))
        parts.insert(insert_at, questiontype_part)
    return f"{base_url}#{'&'.join(parts)}"


def _faq_display_category(chunk: RetrievedChunk) -> str | None:
    metadata = chunk.metadata or {}
    raw_category = metadata.get("category") or metadata.get("category_slug")
    if not isinstance(raw_category, str) or not raw_category.strip():
        return None
    category = raw_category.strip()
    return FAQ_CATEGORY_LABEL_OVERRIDES.get(_normalize_text_key(category), category)


def _is_product_catalog_chunk(chunk: RetrievedChunk) -> bool:
    return chunk.section == "product_catalog"


def _is_answer_scoped_detail_chunk(chunk: RetrievedChunk) -> bool:
    return (
        chunk.section == "product_detail"
        and (
            chunk.metadata.get("retrieval_source") in {"graph", "query_composition"}
            or chunk.chunk_id.startswith(("graph:product:", "composed:product:"))
        )
    )


def _product_catalog_answer_citations(
    chunk: RetrievedChunk,
    *,
    normalized_answer: str,
) -> list[SourceCitation]:
    if not normalized_answer:
        return []

    items = chunk.metadata.get("items")
    if not isinstance(items, list):
        return []

    selected_items: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        source_url = str(item.get("url") or "").strip()
        if not title or not source_url:
            continue
        if _answer_mentions_text(normalized_answer, title):
            selected_items.append((index, item))

    if selected_items:
        return _catalog_item_citations_from_selected(chunk, selected_items)

    return [
        _catalog_group_citation(chunk, category, index=index)
        for index, category in enumerate(_catalog_item_categories(items))
        if _answer_mentions_text(normalized_answer, category)
    ]


def _product_catalog_group_citations(
    chunk: RetrievedChunk,
    *,
    query_tokens: set[str],
) -> list[SourceCitation]:
    items = chunk.metadata.get("items")
    if not _should_cite_catalog_groups(chunk, items, query_tokens=query_tokens):
        return []

    categories = _catalog_item_categories(items)
    return [
        _catalog_group_citation(chunk, category, index=index)
        for index, category in enumerate(categories)
    ]


def _catalog_group_citation(chunk: RetrievedChunk, category: str, *, index: int) -> SourceCitation:
    return SourceCitation(
        chunk_id=f"{chunk.chunk_id}:group:{index}",
        title=category,
        source_url=_catalog_group_url(chunk, category),
        section=chunk.section,
        score=chunk.score,
    )


def _targeted_catalog_group(
    chunk: RetrievedChunk,
    *,
    query_tokens: set[str],
) -> str | None:
    category_title = chunk.metadata.get("category_title")
    if chunk.metadata.get("parent_category_title") and isinstance(category_title, str) and category_title.strip():
        return category_title.strip()

    items = chunk.metadata.get("items")
    matching_categories = [
        category
        for category in _catalog_item_categories(items)
        if _tokens_match_query(query_tokens, _tokenize(category))
    ]
    if len(matching_categories) == 1:
        return matching_categories[0]
    return None


def _should_cite_catalog_groups(
    chunk: RetrievedChunk,
    items: object,
    *,
    query_tokens: set[str],
) -> bool:
    if not isinstance(items, list):
        return False
    if chunk.metadata.get("parent_category_title"):
        return False

    category_counts = _catalog_item_category_counts(items)
    if len(category_counts) <= 1 or max(category_counts.values(), default=0) <= 1:
        return False

    matching_categories = [
        category
        for category in category_counts
        if _tokens_match_query(query_tokens, _tokenize(category))
    ]
    return len(matching_categories) != 1


def _catalog_item_categories(items: object) -> list[str]:
    if not isinstance(items, list):
        return []

    categories: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        if not category or category in seen:
            continue
        seen.add(category)
        categories.append(category)
    return categories


def _catalog_item_category_counts(items: object) -> dict[str, int]:
    category_counts: dict[str, int] = {}
    if not isinstance(items, list):
        return category_counts

    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        if not category:
            continue
        category_counts[category] = category_counts.get(category, 0) + 1
    return category_counts


def _catalog_group_url(chunk: RetrievedChunk, category: str) -> str:
    base_url = chunk.metadata.get("category_url")
    if not isinstance(base_url, str) or not base_url.strip():
        base_url = chunk.source_url
    base_url = urldefrag(base_url.strip()).url
    category_value = quote(category, safe="")
    list_type_param = "card-list_type" if chunk.product_type == "card" else "loan-list_type"
    return f"{base_url}#subcategory={category_value}&{list_type_param}={category_value}&e=0"


def _product_catalog_exact_title_citations(
    chunk: RetrievedChunk,
    *,
    query_tokens: set[str],
) -> list[SourceCitation]:
    items = chunk.metadata.get("items")
    if not isinstance(items, list):
        return []

    selected_items: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        source_url = str(item.get("url") or "").strip()
        if not title or not source_url:
            continue
        if _tokens_match_query(query_tokens, _tokenize(title)):
            selected_items.append((index, item))

    return _catalog_item_citations_from_selected(chunk, selected_items)


def _product_catalog_item_citations(
    chunk: RetrievedChunk,
    *,
    query_tokens: set[str],
    category_filter: str | None = None,
) -> list[SourceCitation]:
    items = chunk.metadata.get("items")
    if not isinstance(items, list):
        return []

    scored_items: list[tuple[int, bool, bool, int, dict[str, Any]]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        source_url = str(item.get("url") or "").strip()
        if not title or not source_url:
            continue
        if category_filter and _normalize_text_key(str(item.get("category") or "")) != _normalize_text_key(
            category_filter
        ):
            continue
        title_exact = _catalog_item_field_matches_query(query_tokens, item, "title")
        category_exact = _catalog_item_field_matches_query(query_tokens, item, "category")
        scored_items.append(
            (
                _catalog_item_relevance_score(query_tokens, item),
                title_exact,
                category_exact,
                index,
                item,
            )
        )

    if query_tokens and any(title_exact or category_exact for _, title_exact, category_exact, _, _ in scored_items):
        scored_items.sort(
            key=lambda scored_item: (
                -int(scored_item[1]),
                -int(scored_item[2]),
                -scored_item[0] if scored_item[1] else 0,
                scored_item[3],
            )
        )

    return _catalog_item_citations_from_selected(
        chunk,
        [(index, item) for _, _, _, index, item in scored_items],
    )


def _catalog_item_citations_from_selected(
    chunk: RetrievedChunk,
    selected_items: list[tuple[int, dict[str, Any]]],
) -> list[SourceCitation]:
    title_counts: dict[str, int] = {}
    for _, item in selected_items:
        title = str(item.get("title") or "").strip()
        title_counts[title] = title_counts.get(title, 0) + 1

    title_positions: dict[str, int] = {}
    citations: list[SourceCitation] = []
    for index, item in selected_items:
        title = str(item.get("title") or "").strip()
        title_positions[title] = title_positions.get(title, 0) + 1
        citations.append(
            SourceCitation(
                chunk_id=f"{chunk.chunk_id}:item:{index}",
                title=_catalog_item_display_title(
                    item,
                    duplicate_count=title_counts.get(title, 0),
                    duplicate_index=title_positions[title],
                ),
                source_url=str(item.get("url") or "").strip(),
                section=chunk.section,
                score=chunk.score,
            )
        )
    return citations


def _catalog_item_display_title(
    item: dict[str, Any],
    *,
    duplicate_count: int,
    duplicate_index: int,
) -> str:
    title = str(item.get("title") or "").strip()
    if duplicate_count <= 1:
        return title

    qualifier = _catalog_item_duplicate_qualifier(item)
    if qualifier:
        return f"{title} ({qualifier})"
    return f"{title} - bản {duplicate_index}"


def _catalog_item_duplicate_qualifier(item: dict[str, Any]) -> str:
    item_key = _normalize_text_key(
        " ".join(
            str(value or "")
            for value in (
                item.get("url"),
                item.get("summary"),
                item.get("category"),
            )
        )
    )
    if "khdn" in item_key or "doanh nghiep" in item_key:
        return "KHDN"
    if "khcn" in item_key or "ca nhan" in item_key:
        return "KHCN"
    return ""


def _catalog_item_relevance_score(query_tokens: set[str], item: dict[str, Any]) -> int:
    if not query_tokens:
        return 0

    title_tokens = _tokenize(str(item.get("title") or ""))
    category_tokens = _tokenize(str(item.get("category") or ""))
    summary_tokens = _tokenize(str(item.get("summary") or ""))

    score = 0
    score += 3 * len(query_tokens & title_tokens)
    score += 4 * len(query_tokens & category_tokens)
    score += len(query_tokens & summary_tokens)
    if title_tokens and title_tokens.issubset(query_tokens):
        score += 10
    if category_tokens and category_tokens.issubset(query_tokens):
        score += 8
    return score


def _catalog_item_field_matches_query(
    query_tokens: set[str],
    item: dict[str, Any],
    field_name: str,
) -> bool:
    field_tokens = _tokenize(str(item.get(field_name) or ""))
    return _tokens_match_query(query_tokens, field_tokens)


def _tokens_match_query(query_tokens: set[str], field_tokens: set[str]) -> bool:
    return bool(query_tokens and field_tokens and field_tokens.issubset(query_tokens))


def _answer_mentions_text(normalized_answer: str, text: str) -> bool:
    normalized_text = _normalize_text_key(text)
    return bool(normalized_text and f" {normalized_text} " in f" {normalized_answer} ")


def _answer_mentions_title(normalized_answer: str, title: str) -> bool:
    if _answer_mentions_text(normalized_answer, title):
        return True
    without_qualifier = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return without_qualifier != title and _answer_mentions_text(normalized_answer, without_qualifier)


def _should_use_strict_title_filter(
    query_tokens: set[str],
    title_tokens: set[str],
    *,
    normalized_query: str,
) -> bool:
    if not query_tokens or not title_tokens or len(title_tokens) > 6:
        return False
    if _is_broad_availability_query(normalized_query):
        return False
    if "khac" in query_tokens or {"so", "sanh"}.issubset(query_tokens) or {"phan", "biet"}.issubset(query_tokens):
        return False
    return len(query_tokens & title_tokens) / len(title_tokens) >= 0.7


def _is_broad_availability_query(normalized_query: str) -> bool:
    return any(
        marker in normalized_query
        for marker in (
            "co cho",
            "co goi",
            "co ho tro",
            "co san pham",
            "co vay",
            "hien co",
            "vay de",
        )
    ) or (" co " in f" {normalized_query} " and " khong " in f" {normalized_query} ")


def _is_relevant_citation(query_tokens: set[str], chunk: RetrievedChunk) -> bool:
    title_tokens = _tokenize(chunk.title)
    text_tokens = _tokenize(chunk.text[:500])
    title_overlap = len(query_tokens & title_tokens)
    text_overlap = len(query_tokens & text_tokens)
    return title_overlap >= 2 or (title_overlap >= 1 and text_overlap >= 2)


def _tokenize(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return {
        token
        for token in TOKEN_PATTERN.findall(ascii_text)
        if len(token) > 1 and token not in CITATION_STOPWORDS
    }


def _normalize_text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(TOKEN_PATTERN.findall(ascii_text))
