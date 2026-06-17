from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

from apps.api.app.models.chat import ChatMessage
from packages.shared.schemas import RetrievedChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

GRAPH_STOPWORDS = {
    "anh",
    "bao",
    "cac",
    "cho",
    "co",
    "cua",
    "duoc",
    "gi",
    "hay",
    "khong",
    "la",
    "lam",
    "nao",
    "nhu",
    "o",
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

LIST_QUERY_MARKERS = (
    "bao gom",
    "cac goi",
    "cac goi tren",
    "cac san pham",
    "cac dich vu",
    "co cac",
    "co nhung",
    "danh sach",
    "gom cac",
    "gom nhung",
    "liet ke",
    "goi tren",
    "hien co",
    "moi goi",
    "moi san pham",
    "nhung goi",
    "nhung san pham",
    "nhung dich vu",
    "nhung goi tren",
    "nhung san pham tren",
    "san pham nao",
    "san pham tren",
    "tung goi",
    "tung san pham",
)

COMPARISON_QUERY_MARKERS = (
    "co gi khac",
    "diem khac",
    "khac gi",
    "khac nhau",
    "phan biet",
    "so sanh",
)

CONTEXTUAL_FOLLOW_UP_MARKERS = (
    "cua no",
    "dich vu do",
    "dich vu nay",
    "goi do",
    "goi nay",
    "goi tren",
    "mo the do",
    "nhom do",
    "nhom nay",
    "san pham do",
    "san pham nay",
    "san pham tren",
    "the do",
    "the nay",
    "ve no",
    "ben tren",
    "cai do",
    "cai nay",
    "cai tren",
    "vua noi",
)

PRODUCT_TYPE_QUERY_MARKERS = {
    "insurance": ("bao hiem", "fwd"),
    "card": ("cac the", "loai the", "mo the", "the ghi no", "the thanh toan", "the tin dung"),
    "loan": ("khoan vay", "vay"),
    "saving": ("tien gui", "tiet kiem"),
    "transfer": ("chuyen khoan", "chuyen tien", "chuyen va nhan tien", "kieu hoi", "nhan tien"),
    "digital_banking": ("digibank", "ngan hang so", "sms banking"),
    "account": ("tai khoan",),
    "investment": ("chung khoan", "dau tu", "quy"),
}

MARKETING_CARD_TITLE_KEYS = {
    "hoan tien khong gioi han",
    "phi thuong nien it nhat",
    "the tot nhat cho hoan tien",
}

REQUESTED_FIELD_MARKERS = {
    "condition": (
        "co nhu cau",
        "co tai san bao dam",
        "cong dan",
        "dap ung yeu cau",
        "doi tuong khach hang",
        "thu nhap",
        "tuoi",
        "yeu cau cap tin dung",
    ),
    "documents": (
        "chung minh",
        "giay to",
        "ho so",
        "ho so chuan bi",
    ),
    "fee": (
        "bieu phi",
        "chi phi",
        "phi",
    ),
    "interest_rate": ("lai suat",),
    "limit": (
        "han muc",
        "muc vay",
        "so tien vay",
    ),
    "procedure": (
        "cac buoc",
        "huong dan",
        "quy trinh",
        "thu tuc",
    ),
    "registration": (
        "cach dang ky",
        "dang ky",
    ),
    "benefits": (
        "dac diem",
        "loi ich",
        "quyen loi",
        "uu dai",
    ),
}


@dataclass(frozen=True)
class GraphProduct:
    title: str
    url: str
    product_type: str | None
    category_title: str | None
    parent_category_title: str | None
    catalog_url: str
    summary: str = ""
    rank: int = 0


@dataclass(frozen=True)
class GraphCategory:
    title: str
    url: str
    product_type: str | None
    parent_title: str | None
    items: tuple[GraphProduct, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GraphSubjectOption:
    title: str
    subject_type: str
    url: str
    product_type: str | None = None
    category_title: str | None = None
    parent_title: str | None = None


@dataclass(frozen=True)
class GraphRetrievalResult:
    chunks: list[RetrievedChunk]
    route: str = "graph"
    resolved_query: str = ""
    clarification: str | None = None
    matched_subject: str | None = None
    clarification_options: tuple[GraphSubjectOption, ...] = field(default_factory=tuple)


@dataclass
class ProductGraph:
    categories_by_key: dict[str, GraphCategory]
    products_by_url: dict[str, GraphProduct]
    category_aliases: tuple[tuple[str, str], ...]
    product_aliases: tuple[tuple[str, str], ...]
    detail_chunks_by_url: dict[str, list[RetrievedChunk]]


class ProductGraphRetriever:
    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or Path(__file__).resolve().parents[5] / "data"
        self.graph = _load_product_graph(self.data_root)

    def retrieve(
        self,
        query: str,
        *,
        history: list[ChatMessage],
        top_k: int = 12,
    ) -> GraphRetrievalResult:
        resolved_query, subject, needs_clarification = self.resolve_query(query, history=history)
        if needs_clarification:
            options = self.suggest_subjects(query, limit=5)
            return GraphRetrievalResult(
                chunks=[],
                resolved_query=resolved_query,
                route="graph_clarification",
                clarification=(
                    "Bạn đang muốn hỏi về sản phẩm, nhóm sản phẩm hoặc dịch vụ nào của Vietcombank? "
                    "Bạn vui lòng nêu rõ tên để tôi tra đúng nguồn."
                ),
                clarification_options=options,
            )

        normalized_query = _normalize_query_key(resolved_query)
        product_matches = self._match_products(normalized_query)
        category_matches = self._match_categories(normalized_query)
        product_type = _infer_product_type(normalized_query)

        is_list_query = _is_list_query(normalized_query)
        chunks: list[RetrievedChunk] = []
        if product_matches and (_is_comparison_query(normalized_query) or not is_list_query):
            chunks.extend(self._product_chunks(product_matches, query=resolved_query, top_k=top_k))
        elif is_list_query or category_matches:
            chunks.extend(
                self._category_chunks(
                    category_matches=category_matches,
                    product_type=product_type,
                    query=resolved_query,
                    top_k=top_k,
                )
            )
        chunks = _dedupe_chunks(chunks)[:top_k]
        return GraphRetrievalResult(
            chunks=chunks,
            route="graph" if chunks else "default",
            resolved_query=resolved_query,
            matched_subject=subject,
        )

    def resolve_query(self, query: str, *, history: list[ChatMessage]) -> tuple[str, str | None, bool]:
        normalized_query = _normalize_query_key(query)
        if not _is_contextual_follow_up(normalized_query):
            return query, None, False

        subject = self.latest_subject(history)
        if not subject:
            return query, None, True

        if _has_phrase(normalized_query, _normalize_query_key(subject)):
            return query, subject, False
        return f"{query} {subject}", subject, False

    def latest_subject(self, history: list[ChatMessage]) -> str | None:
        for message in reversed(history[-8:]):
            if message.role != "user" or not message.content.strip():
                continue

            specific_matches = _specific_subject_options(self.match_subjects(message.content, limit=8))
            if specific_matches:
                return specific_matches[0].title

            normalized_text = _normalize_query_key(message.content)
            product_matches = self._match_products(normalized_text)
            if product_matches:
                return product_matches[0].title

            category_matches = self._match_categories(normalized_text)
            if category_matches:
                return category_matches[0].title

        for message in reversed(history[-8:]):
            if message.role != "assistant" or not message.content.strip():
                continue

            specific_matches = _specific_subject_options(self.match_subjects(message.content, limit=8))
            if len(specific_matches) == 1:
                return specific_matches[0].title
        return None

    def match_subjects(self, query: str, *, limit: int = 8) -> tuple[GraphSubjectOption, ...]:
        normalized_query = _normalize_query_key(query)
        options = [
            *(_product_subject_option(product) for product in self._match_products(normalized_query)),
            *(_category_subject_option(category) for category in self._match_categories(normalized_query)),
        ]
        return tuple(_dedupe_subject_options(options)[:limit])

    def product_options_for_category(
        self,
        option: GraphSubjectOption,
    ) -> tuple[GraphSubjectOption, ...]:
        if option.subject_type != "category":
            return ()
        category_key = _category_key(option.title, option.parent_title, option.product_type)
        category = self.graph.categories_by_key.get(category_key)
        if category is None:
            return ()
        return tuple(_product_subject_option(product) for product in category.items)

    def suggest_subjects(self, query: str, *, limit: int = 5) -> tuple[GraphSubjectOption, ...]:
        normalized_query = _normalize_query_key(query)
        query_tokens = _tokenize(query)
        if not normalized_query:
            return ()

        scored: list[tuple[float, GraphSubjectOption]] = []
        for product in self.graph.products_by_url.values():
            score = _best_subject_score(
                normalized_query,
                query_tokens,
                aliases=_aliases_for_title(product.title),
                text=" ".join(
                    [
                        product.title,
                        product.summary,
                        product.category_title or "",
                        product.parent_category_title or "",
                    ]
                ),
            )
            if score >= 0.52:
                scored.append((score, _product_subject_option(product)))

        for category in self.graph.categories_by_key.values():
            score = _best_subject_score(
                normalized_query,
                query_tokens,
                aliases=_aliases_for_title(category.title),
                text=" ".join(
                    [
                        category.title,
                        category.parent_title or "",
                        " ".join(product.title for product in category.items),
                    ]
                ),
            )
            if score >= 0.52:
                scored.append((score, _category_subject_option(category)))

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].subject_type == "category",
                -len(_normalize_query_key(item[1].title).split()),
            ),
            reverse=True,
        )
        options = _dedupe_subject_options([option for _, option in scored])

        product_type = _infer_product_type(normalized_query)
        if len(options) < limit and product_type is not None:
            fallback_categories = [
                _category_subject_option(category)
                for category in self._suggestion_categories_for_product_type(product_type, query=query)
            ]
            options = _dedupe_subject_options([*options, *fallback_categories])

        return tuple(options[:limit])

    def product_detail_chunks_for_catalog_chunks(
        self,
        catalog_chunks: list[RetrievedChunk],
        *,
        query: str,
        max_products: int = 12,
        top_k_per_product: int = 2,
        requested_field: str | None = None,
    ) -> list[RetrievedChunk]:
        products = self._products_from_catalog_chunks(catalog_chunks, max_products=max_products)
        if not products:
            return []

        chunks: list[RetrievedChunk] = []
        query_tokens = _tokenize(query)
        for index, product in enumerate(products):
            chunks.append(_product_to_chunk(product, score=2.35 - (index * 0.01)))
            detail_chunks = self.graph.detail_chunks_by_url.get(_canonical_url(product.url), [])
            if not detail_chunks:
                continue

            ranked_details = sorted(
                detail_chunks,
                key=lambda chunk: (
                    _requested_field_score(requested_field, chunk),
                    len(query_tokens & _tokenize(f"{chunk.title} {chunk.text[:1600]}")),
                    chunk.score or 0,
                ),
                reverse=True,
            )
            chunks.extend(
                _detail_chunk_for_product(chunk, product=product).model_copy(
                    update={"score": 2.05 - (index * 0.01)}
                )
                for chunk in ranked_details[:top_k_per_product]
            )
        return _dedupe_chunks(chunks)

    def product_dossier_chunks_for_catalog_chunks(
        self,
        catalog_chunks: list[RetrievedChunk],
        *,
        query: str,
        max_products: int = 12,
        requested_field: str | None = None,
        max_chars_per_product: int = 2800,
    ) -> list[RetrievedChunk]:
        products = self._products_from_catalog_chunks(catalog_chunks, max_products=max_products)
        if not products:
            return []

        dossiers: list[RetrievedChunk] = []
        query_tokens = _tokenize(query)
        for index, product in enumerate(products):
            detail_chunks = self.graph.detail_chunks_by_url.get(_canonical_url(product.url), [])
            ranked_details = _rank_detail_chunks_for_dossier(
                detail_chunks,
                query_tokens=query_tokens,
                requested_field=requested_field,
            )
            text = _product_dossier_text(
                product,
                detail_chunks=ranked_details,
                max_chars=max_chars_per_product,
            )
            source_chunk_ids = [chunk.chunk_id for chunk in ranked_details]
            dossiers.append(
                RetrievedChunk(
                    chunk_id=f"composed:product:{_normalize_query_key(product.url)}",
                    document_id=f"composed-product-{_normalize_query_key(product.url)}",
                    title=_product_display_title(product),
                    source_url=product.url,
                    text=text,
                    score=2.6 - (index * 0.01),
                    section="product_detail",
                    product_type=product.product_type,
                    metadata={
                        "category_title": product.category_title,
                        "parent_category_title": product.parent_category_title,
                        "catalog_url": product.catalog_url,
                        "summary": product.summary,
                        "retrieval_source": "query_composition",
                        "source_chunk_ids": source_chunk_ids,
                        "source_chunk_count": len(source_chunk_ids),
                        "requested_field": requested_field,
                    },
                )
            )
        return dossiers

    def _products_from_catalog_chunks(
        self,
        catalog_chunks: list[RetrievedChunk],
        *,
        max_products: int,
    ) -> list[GraphProduct]:
        products: list[GraphProduct] = []
        seen_urls: set[str] = set()
        for chunk in catalog_chunks:
            if chunk.section != "product_catalog":
                continue
            items = chunk.metadata.get("items")
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                if not title or not url:
                    continue
                canonical_url = _canonical_url(url)
                if canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)

                product = self.graph.products_by_url.get(canonical_url)
                if product is None:
                    product = GraphProduct(
                        title=title,
                        url=url,
                        product_type=chunk.product_type,
                        category_title=str(item.get("category") or "").strip() or None,
                        parent_category_title=_metadata_string(
                            chunk.metadata,
                            "parent_category_title",
                        ),
                        catalog_url=_metadata_string(chunk.metadata, "category_url")
                        or chunk.source_url,
                        summary=str(item.get("summary") or "").strip(),
                        rank=index,
                    )
                products.append(product)
                if len(products) >= max_products:
                    return products
        return products

    def _match_products(self, normalized_query: str) -> list[GraphProduct]:
        candidates: list[tuple[int, int, int, int, str]] = []
        for alias, url in self.graph.product_aliases:
            for start, end in _find_phrase_spans(normalized_query, alias):
                candidates.append((start, end, len(alias.split()), len(alias), url))

        if not candidates:
            return []

        filtered_candidates = [
            candidate
            for candidate in candidates
            if not _is_subsumed_product_match(candidate, candidates)
        ]
        filtered_candidates.sort(key=lambda item: (item[0], -item[2], -item[3], item[4]))

        matches: list[GraphProduct] = []
        seen_urls: set[str] = set()
        for _, _, _, _, url in filtered_candidates:
            product = self.graph.products_by_url.get(url)
            if product is None or product.url in seen_urls:
                continue
            seen_urls.add(product.url)
            matches.append(product)
        return matches

    def _match_categories(self, normalized_query: str) -> list[GraphCategory]:
        matches: list[tuple[int, GraphCategory]] = []
        seen_keys: set[str] = set()
        query_tokens = normalized_query.split()
        for alias, key in self.graph.category_aliases:
            alias_tokens = alias.split()
            if not _has_phrase(normalized_query, alias) and (
                len(alias_tokens) < 2
                or not _contains_near_ordered_subsequence(query_tokens, alias_tokens)
            ):
                continue
            category = self.graph.categories_by_key.get(key)
            if category is None or key in seen_keys:
                continue
            seen_keys.add(key)
            matches.append((len(alias_tokens), category))
        max_alias_length = max((alias_length for alias_length, _ in matches), default=0)
        return [category for alias_length, category in matches if alias_length == max_alias_length]

    def _category_chunks(
        self,
        *,
        category_matches: list[GraphCategory],
        product_type: str | None,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        categories = category_matches or self._categories_for_product_type(product_type, query=query)
        return [
            _category_to_chunk(category, score=2.4 - (index * 0.05))
            for index, category in enumerate(categories[:top_k])
        ]

    def _categories_for_product_type(self, product_type: str | None, *, query: str) -> list[GraphCategory]:
        if product_type is None:
            return []

        categories = [
            category
            for category in self.graph.categories_by_key.values()
            if category.product_type == product_type
        ]
        if not categories:
            return []

        query_tokens = _tokenize(query)
        scored = []
        for category in categories:
            text = " ".join(
                [
                    category.title,
                    category.parent_title or "",
                    " ".join(product.title for product in category.items),
                    " ".join(product.summary for product in category.items),
                ]
            )
            overlap = len(query_tokens & _tokenize(text))
            parent_boost = 2 if category.parent_title is None else 0
            scored.append((overlap, parent_boost, -len(category.items), category))

        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if any(overlap > 0 for overlap, _, _, _ in scored):
            return [category for overlap, _, _, category in scored if overlap > 0]
        return [category for _, _, _, category in scored if category.parent_title is None]

    def _product_chunks(
        self,
        products: list[GraphProduct],
        *,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        query_tokens = _tokenize(query)

        for index, product in enumerate(products):
            chunks.append(_product_to_chunk(product, score=2.5 - (index * 0.01)))

        for product in products:
            detail_chunks = self.graph.detail_chunks_by_url.get(_canonical_url(product.url), [])
            if detail_chunks:
                ranked_details = sorted(
                    detail_chunks,
                    key=lambda chunk: len(query_tokens & _tokenize(f"{chunk.title} {chunk.text[:1200]}")),
                    reverse=True,
                )
                detail_limit = max(3, min(5, max(1, top_k // max(1, len(products)))))
                chunks.extend(
                    _detail_chunk_for_product(chunk, product=product).model_copy(
                        update={"score": 2.2}
                    )
                    for chunk in ranked_details[:detail_limit]
                )
        return chunks[:top_k]

    def _suggestion_categories_for_product_type(
        self,
        product_type: str,
        *,
        query: str,
    ) -> list[GraphCategory]:
        ranked_categories = self._categories_for_product_type(product_type, query=query)
        child_categories = [
            category
            for category in self.graph.categories_by_key.values()
            if category.product_type == product_type and category.parent_title
        ]
        parent_categories = [
            category
            for category in self.graph.categories_by_key.values()
            if category.product_type == product_type and not category.parent_title
        ]
        if child_categories:
            return _dedupe_categories([*child_categories, *_non_parent_categories(ranked_categories)])
        return _dedupe_categories([*ranked_categories, *parent_categories])


@lru_cache(maxsize=4)
def _load_product_graph(data_root: Path) -> ProductGraph:
    products_by_url: dict[str, GraphProduct] = {}
    category_items: dict[str, list[GraphProduct]] = {}
    category_defs: dict[str, tuple[str, str, str | None, str | None]] = {}

    catalog_path = data_root / "normalized" / "vietcombank_product_catalogs_normalized.jsonl"
    for payload in _read_jsonl(catalog_path):
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue

        category_title = str(metadata.get("category_title") or payload.get("title") or "").strip()
        if not category_title:
            continue

        parent_title = str(metadata.get("parent_category_title") or "").strip() or None
        category_url = str(metadata.get("category_url") or payload.get("source_url") or "").strip()
        product_type = str(payload.get("product_type") or "").strip() or None
        category_key = _category_key(category_title, parent_title, product_type)
        category_defs[category_key] = (category_title, category_url, product_type, parent_title)

        items = metadata.get("items")
        if not isinstance(items, list):
            continue

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue

            summary = str(item.get("summary") or "").strip()
            title = _catalog_product_title(
                title,
                summary=summary,
                product_type=product_type,
            )
            item_category = str(item.get("category") or "").strip()
            product_category_title = item_category or category_title
            product_parent_title = parent_title or (category_title if item_category else None)
            product = GraphProduct(
                title=title,
                url=url,
                product_type=product_type,
                category_title=product_category_title or None,
                parent_category_title=product_parent_title,
                catalog_url=category_url,
                summary=summary,
                rank=index,
            )
            products_by_url.setdefault(_canonical_url(url), product)

            product_category_key = _category_key(product_category_title, product_parent_title, product_type)
            category_defs.setdefault(
                product_category_key,
                (
                    product_category_title,
                    category_url,
                    product_type,
                    product_parent_title,
                ),
            )
            if all(existing.url != product.url for existing in category_items.setdefault(category_key, [])):
                category_items[category_key].append(product)
            if all(existing.url != product.url for existing in category_items.setdefault(product_category_key, [])):
                category_items[product_category_key].append(product)

    categories_by_key = {
        key: GraphCategory(
            title=title,
            url=url,
            product_type=product_type,
            parent_title=parent_title,
            items=tuple(category_items.get(key, [])),
        )
        for key, (title, url, product_type, parent_title) in category_defs.items()
    }

    return ProductGraph(
        categories_by_key=categories_by_key,
        products_by_url=products_by_url,
        category_aliases=_category_aliases(categories_by_key),
        product_aliases=_product_aliases(products_by_url),
        detail_chunks_by_url=_load_detail_chunks(data_root),
    )


def _catalog_product_title(
    title: str,
    *,
    summary: str,
    product_type: str | None,
) -> str:
    if product_type != "card":
        return title
    if _normalize_query_key(title) not in MARKETING_CARD_TITLE_KEYS:
        return title

    candidate = summary.split(";", 1)[0].strip()
    if _looks_like_card_product_title(candidate):
        return candidate
    return title


def _looks_like_card_product_title(title: str) -> bool:
    normalized = _normalize_query_key(title)
    if not normalized:
        return False
    return any(_has_phrase(normalized, marker) for marker in ("vietcombank", "vcb", "saigon centre"))


def _load_detail_chunks(data_root: Path) -> dict[str, list[RetrievedChunk]]:
    chunks_by_url: dict[str, list[RetrievedChunk]] = {}
    product_chunks_path = data_root / "chunks" / "vietcombank_products_chunks.jsonl"
    for payload in _read_jsonl(product_chunks_path):
        source_url = str(payload.get("source_url") or "").strip()
        if not source_url:
            continue
        chunk = RetrievedChunk(
            chunk_id=str(payload.get("chunk_id") or ""),
            document_id=str(payload.get("document_id") or ""),
            title=str(payload.get("title") or "Vietcombank"),
            source_url=source_url,
            text=str(payload.get("text") or ""),
            score=1.0,
            section=payload.get("section"),
            product_type=payload.get("product_type"),
            metadata={
                **dict(payload.get("metadata") or {}),
                "chunk_index": int(payload.get("chunk_index") or 0),
            },
        )
        chunks_by_url.setdefault(_canonical_url(source_url), []).append(chunk)
    return chunks_by_url


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    payloads: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return payloads


def _category_to_chunk(category: GraphCategory, *, score: float) -> RetrievedChunk:
    title = f"Danh sách gói {category.title} Vietcombank"
    relation_lines = [
        "[GraphRAG]",
        f"Category: {category.title}",
        f"Source: {category.url}",
    ]
    if category.parent_title:
        relation_lines.insert(2, f"Parent category: {category.parent_title}")

    relation_lines.append(f"Danh sách sản phẩm/gói {category.title} Vietcombank:")
    relation_lines.append("; ".join(f"{index}. {item.title}" for index, item in enumerate(category.items, start=1)))
    relation_lines.append("Chi tiết quan hệ sản phẩm:")
    for index, product in enumerate(category.items, start=1):
        relation_lines.append(f"{index}. {product.title}")
        if product.category_title:
            relation_lines.append(f"Nhóm: {product.category_title}")
        if product.summary:
            relation_lines.append(f"Tóm tắt: {product.summary}")
        relation_lines.append(f"URL: {product.url}")

    return RetrievedChunk(
        chunk_id=f"graph:category:{_category_key(category.title, category.parent_title, category.product_type)}",
        document_id=f"graph-category-{_category_key(category.title, category.parent_title, category.product_type)}",
        title=title,
        source_url=category.url,
        text="\n".join(line for line in relation_lines if line),
        score=score,
        section="product_catalog",
        product_type=category.product_type,
        metadata={
            "document_type": "product_catalog",
            "category_url": category.url,
            "category_title": category.title,
            "parent_category_title": category.parent_title,
            "item_count": len(category.items),
            "items": [
                {
                    "title": product.title,
                    "url": product.url,
                    "summary": product.summary,
                    "category": product.category_title,
                }
                for product in category.items
            ],
            "retrieval_source": "graph",
        },
    )


def _product_to_chunk(product: GraphProduct, *, score: float) -> RetrievedChunk:
    display_title = _product_display_title(product)
    lines = [
        "[GraphRAG]",
        f"Product: {display_title}",
        f"Product type: {product.product_type or 'unknown'}",
    ]
    if product.parent_category_title:
        lines.append(f"Parent category: {product.parent_category_title}")
    if product.category_title:
        lines.append(f"Category: {product.category_title}")
    if product.summary:
        lines.append(f"Summary: {product.summary}")
    lines.append(f"URL: {product.url}")

    return RetrievedChunk(
        chunk_id=f"graph:product:{_normalize_query_key(product.url)}",
        document_id=f"graph-product-{_normalize_query_key(product.url)}",
        title=display_title,
        source_url=product.url,
        text="\n".join(lines),
        score=score,
        section="product_detail",
        product_type=product.product_type,
        metadata={
            "category_title": product.category_title,
            "parent_category_title": product.parent_category_title,
            "catalog_url": product.catalog_url,
            "summary": product.summary,
            "retrieval_source": "graph",
        },
    )


def _product_display_title(product: GraphProduct) -> str:
    url_key = _normalize_query_key(product.url)
    text_key = _normalize_query_key(f"{product.summary} {product.category_title or ''}")
    if "khdn" in url_key or "doanh nghiep" in text_key:
        return f"{product.title} (KHDN)"
    if "khcn fwd" in url_key or "ca nhan" in text_key:
        return f"{product.title} (KHCN)"
    return product.title


def _detail_chunk_for_product(chunk: RetrievedChunk, *, product: GraphProduct) -> RetrievedChunk:
    display_title = _product_display_title(product)
    metadata = dict(chunk.metadata)
    if chunk.title != display_title:
        metadata.setdefault("original_title", chunk.title)
    metadata.setdefault("catalog_title", display_title)
    metadata.setdefault("catalog_url", product.catalog_url)
    metadata.setdefault("category_title", product.category_title)
    metadata.setdefault("parent_category_title", product.parent_category_title)
    return chunk.model_copy(
        update={
            "title": display_title,
            "metadata": metadata,
        }
    )


def _rank_detail_chunks_for_dossier(
    detail_chunks: list[RetrievedChunk],
    *,
    query_tokens: set[str],
    requested_field: str | None,
) -> list[RetrievedChunk]:
    if not requested_field:
        return sorted(detail_chunks, key=lambda chunk: int(chunk.metadata.get("chunk_index") or 0))

    return sorted(
        detail_chunks,
        key=lambda chunk: (
            _requested_field_score(requested_field, chunk),
            len(query_tokens & _tokenize(f"{chunk.title} {chunk.text[:1600]}")),
            -(int(chunk.metadata.get("chunk_index") or 0)),
        ),
        reverse=True,
    )


def _product_dossier_text(
    product: GraphProduct,
    *,
    detail_chunks: list[RetrievedChunk],
    max_chars: int,
) -> str:
    lines = [
        "[QueryComposition]",
        f"Product: {_product_display_title(product)}",
        f"Product type: {product.product_type or 'unknown'}",
    ]
    if product.parent_category_title:
        lines.append(f"Parent category: {product.parent_category_title}")
    if product.category_title:
        lines.append(f"Category: {product.category_title}")
    if product.summary:
        lines.append(f"Catalog summary: {product.summary}")
    lines.append(f"URL: {product.url}")

    if detail_chunks:
        lines.append("Composed detail context:")
        for chunk in detail_chunks:
            lines.append(_trim_chunk_text(chunk.text, remaining=max_chars - len("\n".join(lines))))
            if len("\n".join(lines)) >= max_chars:
                break
    else:
        lines.append("Composed detail context: no product detail chunks were found.")

    text = "\n".join(line for line in lines if line).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _trim_chunk_text(text: str, *, remaining: int) -> str:
    if remaining <= 0:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= remaining:
        return normalized
    return normalized[: max(0, remaining - 3)].rstrip() + "..."


def _category_aliases(categories_by_key: dict[str, GraphCategory]) -> tuple[tuple[str, str], ...]:
    aliases: dict[str, str] = {}
    for key, category in categories_by_key.items():
        for alias in _aliases_for_title(category.title):
            if len(alias.split()) < 2 and category.parent_title is None:
                continue
            aliases.setdefault(alias, key)
    return _sorted_aliases(aliases)


def _product_aliases(products_by_url: dict[str, GraphProduct]) -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for url, product in products_by_url.items():
        for alias in _aliases_for_title(product.title):
            item = (alias, url)
            if item in seen:
                continue
            seen.add(item)
            aliases.append(item)
    return tuple(
        sorted(
            aliases,
            key=lambda item: (-len(item[0].split()), -len(item[0]), item[0], item[1]),
        )
    )


def _aliases_for_title(title: str) -> list[str]:
    normalized = _normalize_query_key(title)
    aliases = [normalized]
    for prefix in ("vietcombank ", "vcb ", "fwd "):
        if not normalized.startswith(prefix):
            continue
        tail = normalized.removeprefix(prefix).strip()
        if len(tail.split()) >= 2:
            aliases.append(tail)
    return aliases


def _sorted_aliases(aliases: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            aliases.items(),
            key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]),
        )
    )


def _is_contextual_follow_up(normalized_query: str) -> bool:
    return any(_has_phrase(normalized_query, marker) for marker in CONTEXTUAL_FOLLOW_UP_MARKERS)


def _is_list_query(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in LIST_QUERY_MARKERS)


def _is_comparison_query(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in COMPARISON_QUERY_MARKERS)


def _infer_product_type(normalized_query: str) -> str | None:
    padded_query = f" {normalized_query} "
    for product_type, markers in PRODUCT_TYPE_QUERY_MARKERS.items():
        for marker in markers:
            if f" {marker} " in padded_query or marker in normalized_query:
                return product_type
    return None


def _category_key(title: str, parent_title: str | None, product_type: str | None) -> str:
    return "|".join(
        [
            _normalize_query_key(product_type or ""),
            _normalize_query_key(parent_title or ""),
            _normalize_query_key(title),
        ]
    )


def _canonical_url(url: str) -> str:
    return urldefrag(url.strip()).url


def _requested_field_score(requested_field: str | None, chunk: RetrievedChunk) -> int:
    if not requested_field:
        return 0
    markers = REQUESTED_FIELD_MARKERS.get(requested_field)
    if not markers:
        return 0
    text_key = _normalize_query_key(f"{chunk.title} {chunk.text[:1800]}")
    return sum(1 for marker in markers if marker in text_key)


def _metadata_string(metadata: dict[str, Any], field_name: str) -> str | None:
    value = metadata.get(field_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _has_phrase(normalized_text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {normalized_text} "


def _find_phrase_spans(normalized_text: str, phrase: str) -> list[tuple[int, int]]:
    if not normalized_text or not phrase:
        return []

    pattern = re.compile(rf"(?:^|(?<=\s)){re.escape(phrase)}(?=\s|$)")
    return [(match.start(), match.end()) for match in pattern.finditer(normalized_text)]


def _contains_near_ordered_subsequence(
    tokens: list[str],
    subsequence: list[str],
    *,
    max_extra_tokens: int = 2,
) -> bool:
    if not subsequence:
        return True

    for start_index, token in enumerate(tokens):
        if token != subsequence[0]:
            continue

        next_index = 1
        for end_index in range(start_index + 1, len(tokens)):
            if tokens[end_index] == subsequence[next_index]:
                next_index += 1
                if next_index == len(subsequence):
                    span_length = end_index - start_index + 1
                    return span_length - len(subsequence) <= max_extra_tokens
    return False


def _is_subsumed_product_match(
    candidate: tuple[int, int, int, int, str],
    all_candidates: list[tuple[int, int, int, int, str]],
) -> bool:
    start, end, token_count, char_count, url = candidate
    for other_start, other_end, other_token_count, other_char_count, other_url in all_candidates:
        if other_url == url:
            continue
        if other_token_count < token_count:
            continue
        if other_token_count == token_count and other_char_count <= char_count:
            continue
        if other_start <= start and end <= other_end:
            return True
    return False


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_query_key(text)
    return {
        token
        for token in TOKEN_PATTERN.findall(normalized)
        if len(token) > 1 and token not in GRAPH_STOPWORDS
    }


def _normalize_query_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(TOKEN_PATTERN.findall(no_accents.casefold()))


def _product_subject_option(product: GraphProduct) -> GraphSubjectOption:
    return GraphSubjectOption(
        title=_product_display_title(product),
        subject_type="product",
        url=product.url,
        product_type=product.product_type,
        category_title=product.category_title,
        parent_title=product.parent_category_title,
    )


def _category_subject_option(category: GraphCategory) -> GraphSubjectOption:
    return GraphSubjectOption(
        title=category.title,
        subject_type="category",
        url=category.url,
        product_type=category.product_type,
        category_title=category.title,
        parent_title=category.parent_title,
    )


def _best_subject_score(
    normalized_query: str,
    query_tokens: set[str],
    *,
    aliases: list[str],
    text: str,
) -> float:
    text_tokens = _tokenize(text)
    best = 0.0
    for alias in aliases:
        alias_tokens = _tokenize(alias)
        if not alias_tokens:
            continue

        overlap = len(query_tokens & alias_tokens)
        alias_coverage = overlap / len(alias_tokens)
        query_coverage = overlap / len(query_tokens) if query_tokens else 0.0
        text_coverage = len(query_tokens & text_tokens) / len(query_tokens) if query_tokens else 0.0
        ratio = SequenceMatcher(None, normalized_query, alias).ratio()
        contains_boost = 0.35 if _has_phrase(normalized_query, alias) else 0.0
        score = (
            (0.42 * ratio)
            + (0.32 * alias_coverage)
            + (0.18 * query_coverage)
            + (0.08 * text_coverage)
            + contains_boost
        )
        best = max(best, score)
    return best


def _specific_subject_options(
    options: list[GraphSubjectOption] | tuple[GraphSubjectOption, ...],
) -> tuple[GraphSubjectOption, ...]:
    return tuple(
        option
        for option in options
        if option.subject_type == "product"
        or option.parent_title
        or len(_normalize_query_key(option.title).split()) > 2
    )


def _dedupe_subject_options(options: list[GraphSubjectOption]) -> list[GraphSubjectOption]:
    deduped: list[GraphSubjectOption] = []
    seen: set[tuple[str, str]] = set()
    for option in options:
        if option.subject_type == "category":
            key = (
                option.subject_type,
                "|".join(
                    [
                        _normalize_query_key(option.product_type or ""),
                        _normalize_query_key(option.parent_title or ""),
                        _normalize_query_key(option.title),
                    ]
                ),
            )
        else:
            key = (option.subject_type, option.url or _normalize_query_key(option.title))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return deduped


def _dedupe_categories(categories: list[GraphCategory]) -> list[GraphCategory]:
    deduped: list[GraphCategory] = []
    seen: set[str] = set()
    for category in categories:
        key = _category_key(category.title, category.parent_title, category.product_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(category)
    return deduped


def _non_parent_categories(categories: list[GraphCategory]) -> list[GraphCategory]:
    return [category for category in categories if category.parent_title]


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        key = chunk.chunk_id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped
