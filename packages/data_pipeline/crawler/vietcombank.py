from __future__ import annotations

import asyncio
import hashlib
import json
import time
import unicodedata
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote, urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from packages.shared.schemas import RawDocument

DEFAULT_PRODUCT_CATEGORY_PATHS = [
    "/vi-VN/KHCN/SPDV/Dich-vu-tai-khoan",
    "/vi-VN/KHCN/SPDV/Ngan-hang-so",
    "/vi-VN/KHCN/SPDV/The",
    "/vi-VN/KHCN/SPDV/Vay",
    "/vi-VN/KHCN/SPDV/Tiet-kiem",
    "/vi-VN/KHCN/SPDV/Bao-hiem",
    "/vi-VN/KHCN/SPDV/Dau-tu",
    "/vi-VN/KHCN/SPDV/Chuyen-va-nhan-tien",
]

DEFAULT_FAQ_INDEX_PATH = "/vi-VN/KHCN/Lien-he-va-Ho-tro/Danh-sach-cau-hoi-thuong-gap"
FAQ_BROWSER_PAGE_SIZE = 10
FAQ_CATEGORY_LABEL_OVERRIDES = {
    "cho vay giay to co gia": "Vay cầm cố giấy tờ có giá",
    "nang cap co so luu tru du lich": "Vay nâng cấp cơ sở lưu trú du lịch",
    "o to": "Vay mua ô tô",
    "tin chap nguoi lao dong": "Vay tín chấp đối với Người lao động",
    "vay mua nha dat": "Vay mua nhà ở, đất ở",
    "xay moi co so luu tru": "Vay xây mới cơ sở lưu trú du lịch",
}


@dataclass(frozen=True)
class CrawlConfig:
    base_url: str
    user_agent: str
    request_delay_seconds: float
    max_pages: int = 200
    timeout_seconds: float = 20


@dataclass(frozen=True)
class SitecoreSearchConfig:
    endpoint: str
    params: dict[str, str]
    source_url: str


@dataclass(frozen=True)
class FaqTopic:
    title: str
    url: str


@dataclass(frozen=True)
class LinkedResource:
    url: str
    parent_url: str
    parent_title: str | None
    link_text: str
    link_label: str
    section_title: str | None
    tab_title: str | None


@dataclass(frozen=True)
class ProductCatalogItem:
    title: str
    url: str
    summary: str | None = None
    category: str | None = None


class VietcombankCrawler:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self._last_request_at = 0.0

    async def crawl(self, output_path: Path, seed_urls: Iterable[str] | None = None) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        urls = list(seed_urls or await self.discover_sitemap_urls())
        urls = urls[: self.config.max_pages]
        count = 0

        async with httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        ) as client:
            robots = await self._load_robots(client)
            with output_path.open("w", encoding="utf-8") as file:
                for url in urls:
                    if not robots.can_fetch(self.config.user_agent, url):
                        continue
                    raw = await self._fetch(client, url)
                    if raw is None:
                        continue
                    file.write(raw.model_dump_json() + "\n")
                    count += 1

        return count

    async def discover_product_urls(
        self,
        category_urls: Iterable[str] | None = None,
        page_size: int = 100,
    ) -> list[str]:
        urls = list(category_urls or default_product_category_urls(self.config.base_url))
        discovered: list[str] = []
        seen: set[str] = set()

        async with httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        ) as client:
            robots = await self._load_robots(client)
            for category_url in urls:
                if not robots.can_fetch(self.config.user_agent, category_url):
                    continue

                raw = await self._fetch(client, category_url)
                if raw is None:
                    continue

                search_configs = extract_product_search_configs(raw.html, raw.source_url)
                for search_config in search_configs:
                    if not robots.can_fetch(self.config.user_agent, search_config.endpoint):
                        continue

                    product_urls = await self._fetch_product_urls(client, search_config, page_size)
                    for product_url in product_urls:
                        if product_url in seen:
                            continue
                        seen.add(product_url)
                        discovered.append(product_url)

        return discovered

    async def crawl_product_catalogs(
        self,
        output_path: Path,
        category_urls: Iterable[str] | None = None,
        page_size: int = 100,
    ) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        urls = list(category_urls or default_product_category_urls(self.config.base_url))
        count = 0

        async with httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        ) as client:
            robots = await self._load_robots(client)
            with output_path.open("w", encoding="utf-8") as file:
                for category_url in urls:
                    if not robots.can_fetch(self.config.user_agent, category_url):
                        continue

                    category_raw = await self._fetch(client, category_url)
                    if category_raw is None:
                        continue

                    search_results: list[dict[str, Any]] = []
                    list_sig: str | None = None
                    for search_config in extract_product_search_configs(
                        category_raw.html,
                        category_raw.source_url,
                    ):
                        if not robots.can_fetch(self.config.user_agent, search_config.endpoint):
                            continue
                        list_sig = list_sig or search_config.params.get("sig")
                        search_results.extend(
                            await self._fetch_search_results(client, search_config, page_size)
                        )

                    catalog_raws = raw_documents_from_product_catalog_results(
                        category_raw,
                        search_results,
                        self.config.base_url,
                        list_sig=list_sig,
                    )
                    for catalog_raw in catalog_raws:
                        file.write(catalog_raw.model_dump_json() + "\n")
                        count += 1

        return count

    async def crawl_faq(
        self,
        output_path: Path,
        faq_index_url: str | None = None,
        page_size: int = 100,
    ) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faq_index_url = faq_index_url or default_faq_index_url(self.config.base_url)
        count = 0

        async with httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        ) as client:
            robots = await self._load_robots(client)
            topics = await self._discover_faq_topics(client, robots, faq_index_url, page_size)

            with output_path.open("w", encoding="utf-8") as file:
                for topic in topics:
                    if not robots.can_fetch(self.config.user_agent, topic.url):
                        continue

                    topic_page = await self._fetch(client, topic.url)
                    if topic_page is None:
                        continue

                    search_configs = extract_sitecore_search_configs(
                        topic_page.html,
                        topic_page.source_url,
                    )
                    category_labels = await self._fetch_faq_category_labels(
                        client,
                        topic_page.html,
                        topic_page.source_url,
                        search_configs,
                    )
                    category_positions: dict[tuple[str, str], int] = {}
                    for search_config in search_configs:
                        if not robots.can_fetch(self.config.user_agent, search_config.endpoint):
                            continue

                        for result in await self._fetch_search_results(client, search_config, page_size):
                            if not str(result.get("Html") or "").strip():
                                continue

                            sitecore_item_url = urljoin(self.config.base_url, str(result.get("Url") or topic.url))
                            category_slug = faq_category_from_url(sitecore_item_url)
                            category_label = category_labels.get(faq_category_key(category_slug), category_slug)
                            position_key = (topic.url, category_slug or "")
                            category_positions[position_key] = category_positions.get(position_key, 0) + 1

                            raw = raw_document_from_faq_result(
                                result,
                                self.config.base_url,
                                topic,
                                category_label=category_label,
                                category_position=category_positions[position_key],
                            )
                            if raw is None:
                                continue
                            file.write(raw.model_dump_json() + "\n")
                            count += 1

        return count

    async def crawl_linked_resources(
        self,
        product_raw_path: Path,
        output_path: Path,
        max_resources: int | None = None,
    ) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resources = discover_linked_resources_from_raw_file(product_raw_path, self.config.base_url)
        if max_resources is not None:
            resources = resources[:max_resources]

        count = 0
        async with httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        ) as client:
            robots = await self._load_robots(client)
            with output_path.open("w", encoding="utf-8") as file:
                for resource in resources:
                    if not robots.can_fetch(self.config.user_agent, resource.url):
                        continue

                    raw = await self._fetch_linked_resource(client, resource)
                    if raw is None:
                        continue
                    file.write(raw.model_dump_json() + "\n")
                    count += 1

        return count

    async def discover_sitemap_urls(self) -> list[str]:
        sitemap_url = urljoin(self.config.base_url, "/sitemap.xml")
        async with httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = await client.get(sitemap_url)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        urls: list[str] = []
        for loc in root.findall(".//{*}loc"):
            if loc.text and self._is_relevant_url(loc.text):
                urls.append(loc.text.strip())
        return urls

    async def _fetch_product_urls(
        self,
        client: httpx.AsyncClient,
        search_config: SitecoreSearchConfig,
        page_size: int,
    ) -> list[str]:
        urls: list[str] = []
        for payload_result in await self._fetch_search_results(client, search_config, page_size):
            page_urls = extract_product_urls_from_search_payload(
                {"Results": [payload_result]},
                self.config.base_url,
            )
            urls.extend(page_urls)
        return urls

    async def _fetch_search_results(
        self,
        client: httpx.AsyncClient,
        search_config: SitecoreSearchConfig,
        page_size: int,
    ) -> list[dict[str, Any]]:
        offset = 0
        results: list[dict[str, Any]] = []

        while True:
            await self._wait_for_rate_limit()
            params = dict(search_config.params)
            params["p"] = str(page_size)
            params["e"] = str(offset)
            if "defaultSortOrder" in params and "o" not in params:
                params["o"] = params["defaultSortOrder"]

            response = await client.get(
                search_config.endpoint,
                params=params,
                headers={"Referer": search_config.source_url},
            )
            response.raise_for_status()
            payload = response.json()
            page_results = [result for result in payload.get("Results") or [] if isinstance(result, dict)]
            results.extend(page_results)

            result_count = len(page_results)
            total_count = int(payload.get("Count") or result_count)
            offset += result_count
            if result_count == 0 or offset >= total_count:
                break

        return results

    async def _discover_faq_topics(
        self,
        client: httpx.AsyncClient,
        robots: urllib.robotparser.RobotFileParser,
        faq_index_url: str,
        page_size: int,
    ) -> list[FaqTopic]:
        if not robots.can_fetch(self.config.user_agent, faq_index_url):
            return []

        raw = await self._fetch(client, faq_index_url)
        if raw is None:
            return []

        topics: list[FaqTopic] = []
        seen: set[str] = set()
        for search_config in extract_sitecore_search_configs(raw.html, raw.source_url):
            if not robots.can_fetch(self.config.user_agent, search_config.endpoint):
                continue

            payload = {
                "Results": await self._fetch_search_results(client, search_config, page_size),
            }
            for topic in extract_faq_topics_from_search_payload(payload, self.config.base_url):
                if topic.url in seen:
                    continue
                seen.add(topic.url)
                topics.append(topic)
        return topics

    async def _fetch_faq_category_labels(
        self,
        client: httpx.AsyncClient,
        topic_html: str,
        source_url: str,
        search_configs: Iterable[SitecoreSearchConfig],
    ) -> dict[str | None, str]:
        facet_configs = []
        if html_facet_config := extract_faq_facet_config(topic_html, source_url):
            facet_configs.append(html_facet_config)
        facet_configs.extend(
            config
            for search_config in search_configs
            if (config := faq_facet_config_from_search_config(search_config, self.config.base_url)) is not None
        )

        for facet_config in facet_configs:
            await self._wait_for_rate_limit()
            response = await client.get(
                facet_config.endpoint,
                params=facet_config.params,
                headers={"Referer": source_url},
            )
            response.raise_for_status()
            labels = extract_faq_category_labels_from_facet_payload(response.json())
            if labels:
                return labels
        return {}

    async def _load_robots(self, client: httpx.AsyncClient) -> urllib.robotparser.RobotFileParser:
        robots_url = urljoin(self.config.base_url, "/robots.txt")
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            response = await client.get(robots_url)
            response.raise_for_status()
            parser.parse(response.text.splitlines())
        except Exception:
            parser.parse(["User-agent: *", "Disallow:"])
        return parser

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> RawDocument | None:
        await self._wait_for_rate_limit()
        response = await client.get(url)
        if response.status_code >= 400:
            return None

        html = response.text
        content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
        return RawDocument(
            source_url=url,
            html=html,
            status_code=response.status_code,
            content_hash=content_hash,
            metadata={"fetched_at_unix": time.time()},
        )

    async def _fetch_linked_resource(
        self,
        client: httpx.AsyncClient,
        resource: LinkedResource,
    ) -> RawDocument | None:
        await self._wait_for_rate_limit()
        response = await client.get(resource.url)
        if response.status_code >= 400:
            return None

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        resource_type = infer_linked_resource_type(resource.url, content_type)
        if resource_type == "html":
            html = response.text
            content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
        else:
            content_hash = hashlib.sha256(response.content).hexdigest()
            html = linked_resource_placeholder_html(resource, content_type, resource_type)

        return RawDocument(
            source_url=resource.url,
            html=html,
            status_code=response.status_code,
            content_hash=content_hash,
            metadata={
                "document_type": "linked_resource",
                "parent_url": resource.parent_url,
                "parent_title": resource.parent_title,
                "link_text": resource.link_text,
                "link_label": resource.link_label,
                "section_title": resource.section_title,
                "tab_title": resource.tab_title,
                "resource_type": resource_type,
                "content_type": content_type or None,
                "file_name": linked_resource_file_name(resource.url),
                "text_extraction_status": "html" if resource_type == "html" else "metadata_only",
                "fetched_at_unix": time.time(),
            },
        )

    async def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.config.request_delay_seconds - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _is_relevant_url(self, url: str) -> bool:
        lowered = url.lower()
        keywords = [
            "ca-nhan",
            "khach-hang-ca-nhan",
            "the",
            "vay",
            "bieu-phi",
            "lai-suat",
            "faq",
        ]
        return self.config.base_url in url and any(keyword in lowered for keyword in keywords)


def default_product_category_urls(base_url: str) -> list[str]:
    return [urljoin(base_url, path) for path in DEFAULT_PRODUCT_CATEGORY_PATHS]


def default_faq_index_url(base_url: str) -> str:
    return urljoin(base_url, DEFAULT_FAQ_INDEX_PATH)


def raw_documents_from_product_catalog_results(
    category_raw: RawDocument,
    search_results: Iterable[dict[str, Any]],
    base_url: str,
    *,
    list_sig: str | None = None,
) -> list[RawDocument]:
    category_title = product_catalog_category_title(category_raw.source_url, category_raw.html)
    product_type = product_catalog_product_type(category_raw.source_url)
    list_sig = list_sig or product_catalog_list_sig(category_raw.source_url, product_type)
    items = extract_product_catalog_items_from_search_payload(
        {"Results": list(search_results)},
        base_url,
        product_type=product_type,
    )
    if not items:
        return []

    item_payloads = [_product_catalog_item_payload(item) for item in items]
    raw_docs = [
        raw_document_from_product_catalog_items(
            category_raw=category_raw,
            category_title=category_title,
            item_payloads=item_payloads,
            product_type=product_type,
            list_sig=list_sig,
        )
    ]

    for subcategory_title, subcategory_items in _group_product_catalog_items_by_category(
        item_payloads
    ).items():
        raw_docs.append(
            raw_document_from_product_catalog_items(
                category_raw=category_raw,
                category_title=subcategory_title,
                item_payloads=subcategory_items,
                product_type=product_type,
                list_sig=list_sig,
                parent_category_title=category_title,
            )
        )

    return raw_docs


def raw_document_from_product_catalog_results(
    category_raw: RawDocument,
    search_results: Iterable[dict[str, Any]],
    base_url: str,
) -> RawDocument | None:
    raw_docs = raw_documents_from_product_catalog_results(category_raw, search_results, base_url)
    return raw_docs[0] if raw_docs else None


def raw_document_from_product_catalog_items(
    *,
    category_raw: RawDocument,
    category_title: str,
    item_payloads: list[dict[str, Any]],
    product_type: str | None,
    list_sig: str | None,
    parent_category_title: str | None = None,
) -> RawDocument:
    if not item_payloads:
        raise ValueError("Product catalog must contain at least one item.")

    is_subcategory = parent_category_title is not None
    source_url = category_raw.source_url
    if is_subcategory:
        source_url = product_catalog_subcategory_source_url(
            category_raw.source_url,
            category_title,
            list_sig=list_sig,
        )

    content_payload = {
        "source_url": source_url,
        "category_title": category_title,
        "parent_category_title": parent_category_title,
        "items": item_payloads,
    }
    html = product_catalog_placeholder_html(
        category_title=category_title,
        category_url=category_raw.source_url,
        items=item_payloads,
    )
    content_hash = hashlib.sha256(
        json.dumps(content_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return RawDocument(
        source_url=source_url,
        html=html,
        status_code=category_raw.status_code,
        content_hash=content_hash,
        crawled_at=category_raw.crawled_at,
        metadata={
            "document_type": "product_catalog",
            "category_url": category_raw.source_url,
            "category_title": category_title,
            "parent_category_title": parent_category_title,
            "product_type": product_type,
            "list_sig": list_sig,
            "item_count": len(item_payloads),
            "items": item_payloads,
            "fetched_at_unix": category_raw.metadata.get("fetched_at_unix"),
        },
    )


def _product_catalog_item_payload(item: ProductCatalogItem) -> dict[str, Any]:
    return {
        "title": item.title,
        "url": item.url,
        "summary": item.summary,
        "category": item.category,
    }


def _group_product_catalog_items_by_category(
    items: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        category = str(item.get("category") or "").strip()
        if not category:
            continue
        grouped.setdefault(category, []).append(item)

    return {
        category: category_items
        for category, category_items in grouped.items()
        if len(category_items) < len(items)
    }


def extract_product_catalog_items_from_search_payload(
    payload: dict[str, Any],
    base_url: str,
    product_type: str | None = None,
) -> list[ProductCatalogItem]:
    items: list[ProductCatalogItem] = []
    seen: set[str] = set()

    for result in payload.get("Results") or []:
        if not isinstance(result, dict):
            continue

        html = str(result.get("Html") or "")
        soup = BeautifulSoup(html, "html.parser")
        product_url = _product_catalog_result_url(result, soup, base_url)
        if product_url is None or product_url in seen:
            continue

        title = _product_catalog_result_title(result, soup)
        if not title:
            title = linked_resource_file_name(product_url) or product_url
        summary = _product_catalog_result_summary(soup, title)
        category = _product_catalog_result_category(result, soup) or infer_product_catalog_item_category(
            product_type or product_catalog_product_type(product_url),
            product_url,
            title,
            summary,
        )

        items.append(
            ProductCatalogItem(
                title=title,
                url=product_url,
                summary=summary,
                category=category,
            )
        )
        seen.add(product_url)

    return items


def product_catalog_category_title(source_url: str, html: str) -> str:
    slug = urlparse(source_url).path.rstrip("/").split("/")[-1].lower()
    labels = {
        "dich-vu-tai-khoan": "Dịch vụ tài khoản",
        "ngan-hang-so": "Ngân hàng số",
        "the": "Thẻ",
        "vay": "Vay",
        "tiet-kiem": "Tiết kiệm",
        "bao-hiem": "Bảo hiểm",
        "dau-tu": "Đầu tư",
        "chuyen-va-nhan-tien": "Chuyển và nhận tiền",
    }
    if slug in labels:
        return labels[slug]

    soup = BeautifulSoup(html, "html.parser")
    page_title = _clean_text(_first_text([soup.find("h1"), soup.find("title")]))
    if page_title and _normalize_text_key(page_title) not in {"danh sach san pham", "san pham"}:
        return page_title
    return slug.replace("-", " ").title() if slug else "Sản phẩm"


def product_catalog_product_type(source_url: str) -> str | None:
    segments = {part.lower() for part in urlparse(source_url).path.strip("/").split("/") if part}
    mapping = {
        "dich-vu-tai-khoan": "account",
        "ngan-hang-so": "digital_banking",
        "the": "card",
        "vay": "loan",
        "tiet-kiem": "saving",
        "bao-hiem": "insurance",
        "dau-tu": "investment",
        "chuyen-va-nhan-tien": "transfer",
    }
    for segment, product_type in mapping.items():
        if segment in segments:
            return product_type
    return None


def product_catalog_list_sig(source_url: str, product_type: str | None) -> str | None:
    if product_type == "account":
        return "account-list"
    if product_type == "digital_banking":
        return "digital-banklist"
    if product_type == "card":
        return "card-list"
    if product_type == "saving":
        return "saving-list"
    if product_type in {"loan", "insurance", "investment", "transfer"}:
        return "loan-list"

    slug = urlparse(source_url).path.rstrip("/").split("/")[-1].lower()
    return {
        "dich-vu-tai-khoan": "account-list",
        "ngan-hang-so": "digital-banklist",
        "the": "card-list",
        "vay": "loan-list",
        "tiet-kiem": "saving-list",
        "bao-hiem": "loan-list",
        "dau-tu": "loan-list",
        "chuyen-va-nhan-tien": "loan-list",
    }.get(slug)


def product_catalog_subcategory_source_url(
    category_url: str,
    category_title: str,
    *,
    list_sig: str | None,
) -> str:
    base_url = urldefrag(category_url).url
    encoded_title = quote(category_title)
    if not list_sig:
        return f"{base_url}#subcategory={encoded_title}"
    return f"{base_url}#subcategory={encoded_title}&{list_sig}_type={encoded_title}&e=0"


def infer_product_catalog_item_category(
    product_type: str | None,
    url: str,
    title: str,
    summary: str | None = None,
) -> str | None:
    path_key = _normalize_text_key(urlparse(url).path)
    text_key = _normalize_text_key(f"{title} {summary or ''} {path_key}")

    if product_type == "card":
        if "credit" in path_key or "tin dung" in text_key:
            return "Thẻ tín dụng"
        if "debit" in path_key or "connect24" in text_key or "thanh toan" in text_key:
            return "Thẻ thanh toán"
        if "tra gop" in text_key:
            return "Dịch vụ thẻ"

    if product_type == "loan":
        if "oto" in text_key or "o to" in text_key:
            return "Vay mua ô tô"
        if "kinh doanh" in text_key or "co so luu tru" in text_key or "tai loc" in text_key:
            return "Vay sản xuất kinh doanh"
        if "nha" in text_key or "dat" in text_key or "bat dong san" in text_key:
            return "Vay nhu cầu bất động sản"
        if "tieu dung" in text_key or "tin chap" in text_key or "cam co" in text_key:
            return "Vay tiêu dùng"

    if product_type == "insurance":
        if "bao hiem tiet kiem" in text_key:
            return "Bảo hiểm tiết kiệm"
        if "bao hiem dau tu" in text_key:
            return "Bảo hiểm đầu tư"
        if "bao hiem bao ve" in text_key:
            return "Bảo hiểm bảo vệ"

    if product_type == "transfer":
        if "trong nuoc" in text_key:
            return "Chuyển và nhận tiền trong nước"
        if "kieu hoi" in text_key:
            return "Nhận kiều hối"
        if "ra nuoc ngoai" in text_key:
            return "Chuyển tiền ra nước ngoài"

    if product_type == "saving":
        if "truc tuyen" in text_key:
            return "Tiết kiệm trực tuyến"
        if "tich luy" in text_key or "cho con" in text_key:
            return "Tiết kiệm tích lũy"
        if "tra lai" in text_key or "an vui" in text_key or "rut goc" in text_key:
            return "Tiền gửi tiết kiệm"

    if product_type == "investment":
        if "chung khoan" in text_key:
            return "Chứng khoán"
        if "quy mo" in text_key or "vcbf" in text_key:
            return "Quỹ mở"
        if "uy thac" in text_key:
            return "Ủy thác quản lý tài khoản"
        if "chung chi tien gui" in text_key:
            return "Chứng chỉ tiền gửi"
        if "ho tro tai chinh" in text_key:
            return "Hỗ trợ tài chính"
        if "bao cao phan tich" in text_key:
            return "Báo cáo phân tích"

    return None


def product_catalog_placeholder_html(
    *,
    category_title: str,
    category_url: str,
    items: list[dict[str, Any]],
) -> str:
    lines = [
        "<html><body>",
        f"<h1>Danh sách gói {escape(category_title)} Vietcombank</h1>",
        f"<p>Category URL: {escape(category_url)}</p>",
        "<ol>",
    ]
    for item in items:
        title = escape(str(item.get("title") or ""))
        url = escape(str(item.get("url") or ""))
        summary = escape(str(item.get("summary") or ""))
        category = escape(str(item.get("category") or ""))
        lines.append("<li>")
        lines.append(f"<a href=\"{url}\">{title}</a>")
        if category:
            lines.append(f"<p>Nhóm: {category}</p>")
        if summary:
            lines.append(f"<p>{summary}</p>")
        lines.append("</li>")
    lines.extend(["</ol>", "</body></html>"])
    return "\n".join(lines)


def discover_linked_resources_from_raw_file(path: Path, base_url: str) -> list[LinkedResource]:
    resources: list[LinkedResource] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            raw = RawDocument.model_validate_json(line)
            for resource in extract_linked_resources_from_raw_document(raw, base_url):
                dedupe_key = (resource.parent_url, resource.url, resource.link_label)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                resources.append(resource)
    return resources


def extract_linked_resources_from_raw_document(raw: RawDocument, base_url: str) -> list[LinkedResource]:
    soup = BeautifulSoup(raw.html, "html.parser")
    parent_title = _clean_text(_first_text([soup.find("h1"), soup.find("title")])) or None
    resources: list[LinkedResource] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()

    for anchor in soup.find_all("a", href=True):
        if not is_reference_anchor(anchor):
            continue

        resource_url = canonicalize_linked_resource_url(str(anchor["href"]), base_url)
        if resource_url is None:
            continue

        section_title = linked_resource_section_title(anchor)
        tab_title = linked_resource_tab_title(anchor)
        link_text = _clean_text(anchor.get_text(" ", strip=True))
        link_label = linked_resource_label(anchor, section_title)
        dedupe_key = (resource_url, link_label, section_title, tab_title)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        resources.append(
            LinkedResource(
                url=resource_url,
                parent_url=raw.source_url,
                parent_title=parent_title,
                link_text=link_text,
                link_label=link_label,
                section_title=section_title,
                tab_title=tab_title,
            )
        )

    return resources


def canonicalize_linked_resource_url(url: str, base_url: str) -> str | None:
    if not url or url.startswith(("#", "javascript:", "mailto:", "tel:")):
        return None

    absolute_url = urldefrag(urljoin(base_url, url)).url
    parsed = urlparse(absolute_url)
    base = urlparse(base_url)
    if parsed.netloc.lower() != base.netloc.lower():
        return None
    return absolute_url


def is_reference_anchor(anchor: Any) -> bool:
    text = _normalize_text_key(anchor.get_text(" ", strip=True))
    return "tai day" in text


def linked_resource_label(anchor: Any, section_title: str | None = None) -> str:
    parent = anchor.find_parent(["p", "li", "div"])
    context = _clean_text(parent.get_text(" ", strip=True) if parent is not None else "")
    anchor_text = _clean_text(anchor.get_text(" ", strip=True))
    label = context.replace(anchor_text, " ")
    label = _clean_text(label)
    label = _clean_reference_label(label)
    label = _strip_trailing_reference_word(label)

    if label and not _is_reference_only_label(label):
        return label[:180]
    if section_title:
        return section_title[:180]
    return linked_resource_file_name(str(anchor.get("href") or "")) or anchor_text or "Linked resource"


def linked_resource_section_title(anchor: Any) -> str | None:
    content_item = anchor.find_parent(class_="content-item")
    if content_item is not None:
        title = _clean_text(_first_text([content_item.select_one(".name")]))
        if title:
            return title

    for selector in [".question-title", ".title", "h2", "h3", "strong"]:
        ancestor = anchor.find_parent()
        while ancestor is not None:
            title_node = ancestor.select_one(selector) if hasattr(ancestor, "select_one") else None
            title = _clean_text(_first_text([title_node]))
            if title:
                return title
            ancestor = ancestor.find_parent() if hasattr(ancestor, "find_parent") else None
    return None


def linked_resource_tab_title(anchor: Any) -> str | None:
    component = anchor.find_parent(class_="information-detail-component")
    wrapper = anchor.find_parent(class_="content-wrapper")
    if component is None or wrapper is None:
        return None

    tab_titles = [
        _clean_text(tab.get_text(" ", strip=True))
        for tab in component.select(".select-item-wrapper .select-item")
    ]
    wrappers = component.select(".content-wrapper")
    for index, candidate in enumerate(wrappers):
        if candidate is wrapper and index < len(tab_titles):
            return tab_titles[index] or None
    return None


def infer_linked_resource_type(url: str, content_type: str) -> str:
    path = urlparse(url).path.lower()
    suffix = Path(path).suffix.lstrip(".")
    if suffix in {"pdf", "doc", "docx", "xls", "xlsx", "csv"}:
        return suffix
    if "pdf" in content_type:
        return "pdf"
    if "word" in content_type:
        return "docx"
    if "excel" in content_type or "spreadsheet" in content_type:
        return "xlsx"
    if "html" in content_type or not suffix:
        return "html"
    return content_type.split("/")[-1] if "/" in content_type else suffix or "binary"


def linked_resource_file_name(url: str) -> str | None:
    name = Path(urlparse(url).path).name
    return name or None


def linked_resource_placeholder_html(
    resource: LinkedResource,
    content_type: str,
    resource_type: str,
) -> str:
    lines = [
        "<html><body>",
        f"<h1>{resource.link_label}</h1>",
        f"<p>Parent product: {resource.parent_title or resource.parent_url}</p>",
        f"<p>Resource URL: {resource.url}</p>",
        f"<p>Resource type: {resource_type}</p>",
        f"<p>Content type: {content_type}</p>",
        "</body></html>",
    ]
    return "\n".join(lines)


def _clean_reference_label(label: str) -> str:
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


def extract_sitecore_search_configs(html: str, source_url: str) -> list[SitecoreSearchConfig]:
    soup = BeautifulSoup(html, "html.parser")
    configs: list[SitecoreSearchConfig] = []

    for element in soup.select(".search-results[data-properties]"):
        raw_value = element.get("data-properties")
        raw_properties = raw_value[0] if isinstance(raw_value, list) else raw_value
        if not isinstance(raw_properties, str) or not raw_properties:
            continue

        try:
            properties = json.loads(raw_properties)
        except json.JSONDecodeError:
            continue

        endpoint = properties.get("endpoint")
        if not endpoint or "customresults" not in endpoint:
            continue

        params = {
            key: str(value)
            for key, value in properties.items()
            if key not in {"endpoint", "autoFireSearch"} and value is not None and value != ""
        }
        configs.append(
            SitecoreSearchConfig(
                endpoint=urljoin(source_url, endpoint),
                params=params,
                source_url=source_url,
            )
        )

    return configs


def extract_product_search_configs(html: str, source_url: str) -> list[SitecoreSearchConfig]:
    return extract_sitecore_search_configs(html, source_url)


def extract_faq_facet_config(html: str, source_url: str) -> SitecoreSearchConfig | None:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.select("[data-facet-endpoint]"):
        endpoint = str(element.get("data-facet-endpoint") or "")
        if "customfacets" not in endpoint or "questiontype" not in endpoint.lower():
            continue
        return SitecoreSearchConfig(
            endpoint=urljoin(source_url, endpoint),
            params={},
            source_url=source_url,
        )
    return None


def faq_facet_config_from_search_config(
    search_config: SitecoreSearchConfig,
    base_url: str,
) -> SitecoreSearchConfig | None:
    scope = search_config.params.get("s")
    item_id = search_config.params.get("itemid")
    if not scope or not item_id:
        return None

    return SitecoreSearchConfig(
        endpoint=urljoin(base_url, "/sxa/searchapi/customfacets/"),
        params={
            "f": "questiontype",
            "s": scope,
            "l": search_config.params.get("l", "vi-VN"),
            "itemid": item_id,
            "sig": search_config.params.get("sig", ""),
        },
        source_url=search_config.source_url,
    )


def extract_faq_category_labels_from_facet_payload(payload: dict[str, Any]) -> dict[str | None, str]:
    labels: dict[str | None, str] = {}
    for facet in payload.get("Facets") or []:
        if not isinstance(facet, dict):
            continue
        if str(facet.get("Name") or facet.get("Key") or "").casefold() != "questiontype":
            continue
        for value in facet.get("Values") or []:
            if not isinstance(value, dict):
                continue
            name = _clean_text(str(value.get("Name") or ""))
            if name:
                labels[faq_category_key(name)] = name
    return labels


def extract_faq_topics_from_search_payload(payload: dict[str, Any], base_url: str) -> list[FaqTopic]:
    topics: list[FaqTopic] = []
    seen: set[str] = set()

    for result in payload.get("Results") or []:
        if not isinstance(result, dict) or not result.get("Html"):
            continue

        soup = BeautifulSoup(str(result["Html"]), "html.parser")
        link = soup.select_one(".question-category__button[href]") or soup.find("a", href=True)
        if link is None:
            continue

        url = canonicalize_faq_topic_url(str(link["href"]), base_url)
        if url is None or url in seen:
            continue

        title = _clean_text(
            _first_text(
                [
                    soup.select_one(".question-category__title"),
                    soup.find("strong"),
                    link,
                ]
            )
        )
        if not title:
            title = str(result.get("Name") or result.get("Path") or "FAQ")

        seen.add(url)
        topics.append(FaqTopic(title=title, url=url))

    return topics


def raw_document_from_faq_result(
    result: dict[str, Any],
    base_url: str,
    topic: FaqTopic,
    *,
    category_label: str | None = None,
    category_position: int | None = None,
) -> RawDocument | None:
    html = str(result.get("Html") or "")
    if not html.strip():
        return None

    sitecore_item_url = urljoin(base_url, str(result.get("Url") or topic.url))
    category_slug = faq_category_from_url(sitecore_item_url)
    category = normalize_faq_category_label(category_label or category_slug)
    question_slug = urlparse(sitecore_item_url).path.rstrip("/").split("/")[-1]
    source_url = faq_source_url(topic.url, category, question_slug, category_position=category_position)
    category_page_offset = faq_category_page_offset(category_position)
    content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
    return RawDocument(
        source_url=source_url,
        html=html,
        status_code=200,
        content_hash=content_hash,
        metadata={
            "document_type": "faq",
            "topic": topic.title,
            "topic_url": topic.url,
            "category": category,
            "category_slug": category_slug,
            "category_position": category_position,
            "category_page_offset": category_page_offset,
            "sitecore_item_url": sitecore_item_url,
            "path": result.get("Path"),
            "fetched_at_unix": time.time(),
        },
    )


def canonicalize_faq_topic_url(url: str, base_url: str) -> str | None:
    absolute_url = urldefrag(urljoin(base_url, url)).url
    parsed = urlparse(absolute_url)
    base = urlparse(base_url)

    if parsed.netloc.lower() != base.netloc.lower():
        return None
    if "danh-sach-cau-hoi-theo-chu-de" not in parsed.path.lower():
        return None

    return parsed._replace(query="").geturl()


def faq_category_from_url(url: str) -> str | None:
    parts = [part for part in urlparse(url).path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    category = parts[-2]
    if "---content" in category.lower():
        return None
    return category.replace("-", " ")


def faq_category_key(category: str | None) -> str | None:
    if not category:
        return None
    return _normalize_text_key(category)


def normalize_faq_category_label(category: str | None) -> str | None:
    if not category:
        return None
    stripped = category.strip()
    return FAQ_CATEGORY_LABEL_OVERRIDES.get(_normalize_text_key(stripped), stripped)


def _normalize_text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(no_accents.split()).casefold()


def faq_category_page_offset(category_position: int | None) -> int:
    if category_position is None or category_position < 1:
        return 0
    return ((category_position - 1) // FAQ_BROWSER_PAGE_SIZE) * FAQ_BROWSER_PAGE_SIZE


def faq_source_url(
    topic_url: str,
    category: str | None,
    question_slug: str,
    *,
    category_position: int | None = None,
) -> str:
    fragment_parts = [f"p={FAQ_BROWSER_PAGE_SIZE}", f"e={faq_category_page_offset(category_position)}"]
    if category:
        fragment_parts.append(f"questiontype={quote(category)}")
    fragment_parts.append("comp=faq_ls_tp")
    fragment_parts.append(f"faq={quote(question_slug)}")
    return f"{urldefrag(topic_url).url}#{'&'.join(fragment_parts)}"


def extract_product_urls_from_search_payload(payload: dict[str, Any], base_url: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for result in payload.get("Results") or []:
        candidates = []
        if isinstance(result, dict):
            if result.get("Url"):
                candidates.append(str(result["Url"]))
            if result.get("Html"):
                candidates.extend(_extract_href_candidates(str(result["Html"])))

        for candidate in candidates:
            product_url = canonicalize_product_url(candidate, base_url)
            if product_url is None or product_url in seen:
                continue
            seen.add(product_url)
            urls.append(product_url)

    return urls


def _product_catalog_result_url(
    result: dict[str, Any],
    soup: BeautifulSoup,
    base_url: str,
) -> str | None:
    candidates: list[str] = []
    if result.get("Url"):
        candidates.append(str(result["Url"]))
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if isinstance(href, str):
            candidates.append(href)

    for candidate in candidates:
        product_url = canonicalize_product_url(candidate, base_url)
        if product_url is not None:
            return product_url
    return None


def _product_catalog_result_title(result: dict[str, Any], soup: BeautifulSoup) -> str:
    selectors = [
        ".card-title",
        ".product-title",
        ".title",
        ".name",
        "h3",
        "h2",
        "a[href]",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        value = _clean_text(_first_text([node]))
        if value and not _is_catalog_action_text(value):
            return value[:180]

    for raw_line in soup.get_text("\n", strip=True).splitlines():
        value = _clean_text(raw_line)
        if value and not _is_catalog_action_text(value):
            return value[:180]

    for key in ["Name", "Title", "name", "title"]:
        value = _clean_text(str(result.get(key) or ""))
        if value and not _is_catalog_action_text(value):
            return value
    return ""


def _product_catalog_result_category(result: dict[str, Any], soup: BeautifulSoup) -> str | None:
    for key in ["Category", "ProductType", "category", "productType"]:
        value = _clean_text(str(result.get(key) or ""))
        if value:
            return value[:120]

    for selector in [".category", ".tag", ".label", ".field-category"]:
        node = soup.select_one(selector)
        value = _clean_text(_first_text([node]))
        if value and not _is_catalog_action_text(value):
            return value[:120]
    return None


def _product_catalog_result_summary(soup: BeautifulSoup, title: str) -> str | None:
    if soup is None:
        return None

    lines: list[str] = []
    seen: set[str] = set()
    title_key = _normalize_text_key(title)
    for raw_line in soup.get_text("\n", strip=True).splitlines():
        line = _clean_text(raw_line)
        key = _normalize_text_key(line)
        if not line or key == title_key or key in seen or _is_catalog_action_text(line):
            continue
        seen.add(key)
        lines.append(line)

    return "; ".join(lines[:8]) if lines else None


def _is_catalog_action_text(text: str) -> bool:
    key = _normalize_text_key(text)
    return key in {
        "dat lich tu van",
        "dang ky ngay",
        "mo the ngay",
        "so sanh the",
        "tim hieu them",
        "xem chi tiet",
        "xem them",
        "xem tat ca",
    }


def canonicalize_product_url(url: str, base_url: str) -> str | None:
    absolute_url = urljoin(base_url, url)
    absolute_url = urldefrag(absolute_url).url
    parsed = urlparse(absolute_url)
    base = urlparse(base_url)

    if parsed.netloc.lower() != base.netloc.lower():
        return None
    if not _is_product_detail_path(parsed.path):
        return None

    return parsed._replace(query="").geturl()


def _extract_href_candidates(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    hrefs: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if isinstance(href, str):
            hrefs.append(href)
    return hrefs


def _first_text(nodes: list[object | None]) -> str:
    for node in nodes:
        if node is not None:
            return str(getattr(node, "get_text", lambda *_: "")(" "))
    return ""


def _clean_text(text: str) -> str:
    text = " ".join(text.split())
    return text.strip()


def _is_product_detail_path(path: str) -> bool:
    parts = [part.lower() for part in path.strip("/").split("/") if part]
    try:
        spdv_index = parts.index("spdv")
    except ValueError:
        return False
    return len(parts) > spdv_index + 2


def load_seed_urls(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        return list(json.loads(path.read_text(encoding="utf-8")))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
