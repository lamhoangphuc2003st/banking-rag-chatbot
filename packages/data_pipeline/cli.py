from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from apps.api.app.core.config import get_settings
from packages.data_pipeline.chunker import chunk_file
from packages.data_pipeline.crawler.vietcombank import CrawlConfig, VietcombankCrawler, load_seed_urls
from packages.data_pipeline.indexer import QdrantIndexer
from packages.data_pipeline.normalizer import normalize_file
from packages.data_pipeline.paths import CHUNKS_DIR, NORMALIZED_DIR, RAW_DIR, ensure_data_dirs

app = typer.Typer(help="Vietcombank data pipeline commands.")


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
def index(
    chunks: Path = CHUNKS_DIR / "vietcombank_chunks.jsonl",
    batch_size: int = 64,
) -> None:
    settings = get_settings()
    indexer = QdrantIndexer(settings)
    count = asyncio.run(indexer.index_file(chunks, batch_size=batch_size))
    typer.echo(f"Indexed {count} chunks into {settings.qdrant_collection}")


if __name__ == "__main__":
    app()
