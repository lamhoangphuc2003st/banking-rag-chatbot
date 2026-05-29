from __future__ import annotations

import asyncio
import hashlib
import json
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import httpx

from packages.shared.schemas import RawDocument


@dataclass(frozen=True)
class CrawlConfig:
    base_url: str
    user_agent: str
    request_delay_seconds: float
    max_pages: int = 200
    timeout_seconds: float = 20


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


def load_seed_urls(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        return list(json.loads(path.read_text(encoding="utf-8")))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
