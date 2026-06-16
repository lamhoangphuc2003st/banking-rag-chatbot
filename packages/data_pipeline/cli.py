from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from apps.api.app.core.config import Settings, get_settings
from packages.data_pipeline.chunker import chunk_file, merge_chunk_files
from packages.data_pipeline.crawler.vietcombank import (
    CrawlConfig,
    VietcombankCrawler,
    load_seed_urls,
)
from packages.data_pipeline.indexer import QdrantIndexer
from packages.data_pipeline.normalizer import normalize_file
from packages.data_pipeline.paths import CHUNKS_DIR, NORMALIZED_DIR, RAW_DIR, ensure_data_dirs

app = typer.Typer(help="Vietcombank data pipeline commands.")

RUNTIME_DATA_FILES = (
    Path("chunks") / "vietcombank_chunks.jsonl",
    Path("chunks") / "vietcombank_products_chunks.jsonl",
    Path("chunks") / "vietcombank_faq_chunks.jsonl",
    Path("chunks") / "vietcombank_product_catalogs_chunks.jsonl",
    Path("normalized") / "vietcombank_product_catalogs_normalized.jsonl",
)


@app.command()
def crawl(
    output: Path = RAW_DIR / "vietcombank_raw.jsonl",
    seeds: Path | None = None,
    max_pages: int = 200,
) -> None:
    settings = get_settings()
    ensure_data_dirs()
    seed_urls = load_seed_urls(seeds) if seeds else None
    crawler = VietcombankCrawler(
        CrawlConfig(
            base_url=settings.vietcombank_base_url,
            user_agent=settings.crawler_user_agent,
            request_delay_seconds=settings.crawler_request_delay_seconds,
            max_pages=max_pages,
        )
    )
    count = asyncio.run(crawler.crawl(output, seed_urls=seed_urls))
    typer.echo(f"Crawled {count} pages into {output}")


@app.command()
def discover_products(
    output: Path = RAW_DIR / "vietcombank_product_urls.txt",
    categories: Path | None = None,
    page_size: int = 100,
) -> None:
    settings = get_settings()
    ensure_data_dirs()
    category_urls = load_seed_urls(categories) if categories else None
    crawler = VietcombankCrawler(
        CrawlConfig(
            base_url=settings.vietcombank_base_url,
            user_agent=settings.crawler_user_agent,
            request_delay_seconds=settings.crawler_request_delay_seconds,
        )
    )
    urls = asyncio.run(crawler.discover_product_urls(category_urls=category_urls, page_size=page_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
    typer.echo(f"Discovered {len(urls)} product URLs into {output}")


@app.command("crawl-catalogs")
def crawl_catalogs(
    output: Path = RAW_DIR / "vietcombank_product_catalogs_raw.jsonl",
    categories: Path | None = None,
    page_size: int = 100,
) -> None:
    settings = get_settings()
    ensure_data_dirs()
    category_urls = load_seed_urls(categories) if categories else None
    crawler = VietcombankCrawler(
        CrawlConfig(
            base_url=settings.vietcombank_base_url,
            user_agent=settings.crawler_user_agent,
            request_delay_seconds=settings.crawler_request_delay_seconds,
        )
    )
    count = asyncio.run(
        crawler.crawl_product_catalogs(
            output_path=output,
            category_urls=category_urls,
            page_size=page_size,
        )
    )
    typer.echo(f"Crawled {count} product catalogs into {output}")


@app.command()
def crawl_faq(
    output: Path = RAW_DIR / "vietcombank_faq_raw.jsonl",
    page_size: int = 100,
) -> None:
    settings = get_settings()
    ensure_data_dirs()
    crawler = VietcombankCrawler(
        CrawlConfig(
            base_url=settings.vietcombank_base_url,
            user_agent=settings.crawler_user_agent,
            request_delay_seconds=settings.crawler_request_delay_seconds,
        )
    )
    count = asyncio.run(crawler.crawl_faq(output, page_size=page_size))
    typer.echo(f"Crawled {count} FAQ entries into {output}")


@app.command()
def crawl_linked_resources(
    product_raw: Path = RAW_DIR / "vietcombank_products_raw.jsonl",
    output: Path = RAW_DIR / "vietcombank_linked_resources_raw.jsonl",
    max_resources: int | None = None,
) -> None:
    settings = get_settings()
    ensure_data_dirs()
    crawler = VietcombankCrawler(
        CrawlConfig(
            base_url=settings.vietcombank_base_url,
            user_agent=settings.crawler_user_agent,
            request_delay_seconds=settings.crawler_request_delay_seconds,
        )
    )
    count = asyncio.run(
        crawler.crawl_linked_resources(
            product_raw_path=product_raw,
            output_path=output,
            max_resources=max_resources,
        )
    )
    typer.echo(f"Crawled {count} linked resources into {output}")


@app.command()
def normalize(
    input_path: Path = RAW_DIR / "vietcombank_raw.jsonl",
    output: Path = NORMALIZED_DIR / "vietcombank_normalized.jsonl",
) -> None:
    ensure_data_dirs()
    count = normalize_file(input_path, output)
    typer.echo(f"Normalized {count} documents into {output}")


@app.command()
def chunk(
    input_path: Path = NORMALIZED_DIR / "vietcombank_normalized.jsonl",
    output: Path = CHUNKS_DIR / "vietcombank_chunks.jsonl",
    max_chars: int = 1200,
    overlap: int = 160,
) -> None:
    ensure_data_dirs()
    count = chunk_file(input_path, output, max_chars=max_chars, overlap=overlap)
    typer.echo(f"Wrote {count} chunks into {output}")


@app.command()
def merge_chunks(
    output: Path = CHUNKS_DIR / "vietcombank_chunks.jsonl",
    include_product_catalogs: bool = False,
    include_linked_resources: bool = False,
) -> None:
    ensure_data_dirs()
    input_paths = []
    if include_product_catalogs:
        input_paths.append(CHUNKS_DIR / "vietcombank_product_catalogs_chunks.jsonl")
    input_paths.extend(
        [
            CHUNKS_DIR / "vietcombank_products_chunks.jsonl",
            CHUNKS_DIR / "vietcombank_faq_chunks.jsonl",
        ]
    )
    if include_linked_resources:
        input_paths.append(CHUNKS_DIR / "vietcombank_linked_resources_chunks.jsonl")

    count = merge_chunk_files(input_paths, output)
    typer.echo(f"Merged {count} chunks into {output}")


@app.command()
def index(
    chunks: Path = CHUNKS_DIR / "vietcombank_chunks.jsonl",
    batch_size: int = 64,
    recreate: bool = typer.Option(
        False,
        "--recreate",
        help="Drop and recreate the target Qdrant collection before indexing.",
    ),
) -> None:
    settings = get_settings()
    indexer = QdrantIndexer(settings)
    count = asyncio.run(
        indexer.index_file(
            chunks,
            batch_size=batch_size,
            recreate=recreate or settings.qdrant_recreate_collection,
        )
    )
    typer.echo(f"Indexed {count} chunks into {settings.qdrant_collection}")


@app.command("verify-runtime")
def verify_runtime(
    check_external: bool = typer.Option(
        False,
        "--check-external",
        help="Also check database, Qdrant collection, and Redis connectivity.",
    ),
) -> None:
    settings = get_settings()
    failures = _verify_runtime_data_files(settings)
    if check_external:
        failures.extend(asyncio.run(_verify_external_services(settings)))

    if failures:
        typer.echo("Runtime verification failed:", err=True)
        for failure in failures:
            typer.echo(f"- {failure}", err=True)
        raise typer.Exit(1)

    data_root = _runtime_data_root(settings)
    typer.echo(f"Runtime verification passed. data_root={data_root}")


def _runtime_data_root(settings: Settings) -> Path:
    return Path(settings.rag_data_root) if settings.rag_data_root else Path("data")


def _verify_runtime_data_files(settings: Settings) -> list[str]:
    data_root = _runtime_data_root(settings)
    failures: list[str] = []
    for relative_path in RUNTIME_DATA_FILES:
        path = data_root / relative_path
        if not path.exists():
            failures.append(f"Missing runtime data file: {path}")
            continue
        if path.stat().st_size <= 0:
            failures.append(f"Runtime data file is empty: {path}")
    return failures


async def _verify_external_services(settings: Settings) -> list[str]:
    checks = await asyncio.gather(
        _verify_database(settings),
        _verify_qdrant(settings),
        _verify_redis(settings),
    )
    return [failure for failures in checks for failure in failures]


async def _verify_database(settings: Settings) -> list[str]:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("select 1"))
    except Exception as exc:  # pragma: no cover - provider/network boundary
        return [f"DATABASE_URL check failed: {exc}"]
    finally:
        await engine.dispose()
    return []


async def _verify_qdrant(settings: Settings) -> list[str]:
    try:
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(settings.qdrant_request_timeout_seconds),
        )
        try:
            exists = await client.collection_exists(settings.qdrant_collection)
            if not exists:
                return [f"Qdrant collection not found: {settings.qdrant_collection}"]
            point_count = await client.count(
                collection_name=settings.qdrant_collection,
                exact=False,
            )
        finally:
            await client.close()
    except Exception as exc:  # pragma: no cover - provider/network boundary
        return [f"Qdrant check failed: {exc}"]

    if int(point_count.count or 0) <= 0:
        return [f"Qdrant collection is empty: {settings.qdrant_collection}"]
    return []


async def _verify_redis(settings: Settings) -> list[str]:
    needs_redis = {
        settings.api_rate_limit_backend.strip().casefold(),
        settings.rag_cache_backend.strip().casefold(),
    } & {"redis"}
    if not needs_redis:
        return []
    if not settings.redis_url.strip():
        return ["REDIS_URL is required when Redis-backed rate limiting or cache is enabled."]

    try:
        from redis.asyncio import Redis

        redis = Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
        )
        try:
            await redis.ping()
        finally:
            await redis.aclose()
    except Exception as exc:  # pragma: no cover - provider/network boundary
        return [f"Redis check failed: {exc}"]
    return []


if __name__ == "__main__":
    app()
