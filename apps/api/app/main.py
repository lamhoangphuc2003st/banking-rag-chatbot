from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.core.config import get_settings
from apps.api.app.core.logging import configure_logging, get_logger
from apps.api.app.core.rate_limit import RateLimiter, create_rate_limiter
from apps.api.app.db.audit import save_chat_audit
from apps.api.app.db.session import AsyncSessionLocal, get_db_session
from apps.api.app.models.chat import ChatRequest, ChatResponse, SourceCitation
from apps.api.app.rag.pipeline import RagPipeline

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RagPipeline(settings)
    app.state.rate_limiter = create_rate_limiter(
        backend=settings.api_rate_limit_backend,
        limit_per_minute=settings.api_rate_limit_per_minute,
        redis_url=settings.redis_url,
    )
    logger.info("api_started", env=settings.app_env)
    try:
        yield
    finally:
        with suppress(Exception):
            await cast(RagPipeline, app.state.pipeline).close()
        await cast(RateLimiter, app.state.rate_limiter).close()
        logger.info("api_stopped")


app = FastAPI(
    title="Vietcombank RAG Platform API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


async def enforce_rate_limit(request: Request) -> None:
    limiter = cast(RateLimiter, request.app.state.rate_limiter)
    client_host = request.client.host if request.client else "unknown"
    key = request.headers.get("Authorization") or client_host
    try:
        allowed = await limiter.allow(key)
    except Exception as exc:  # pragma: no cover - provider/network boundary
        logger.warning("rate_limit_check_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiter is unavailable.",
        ) from exc
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
        )


def get_pipeline(request: Request) -> RagPipeline:
    return cast(RagPipeline, request.app.state.pipeline)


PipelineDependency = Annotated[RagPipeline, Depends(get_pipeline)]
DbSessionDependency = Annotated[AsyncSession, Depends(get_db_session)]


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready(request: Request) -> dict[str, Any]:
    checks = await _readiness_checks(request)
    failed = [name for name, check in checks.items() if check["status"] != "ok"]
    if failed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "failed": failed, "checks": checks},
        )
    return {
        "status": "ready",
        "collection": settings.qdrant_collection,
        "env": settings.app_env,
        "checks": checks,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
async def chat(
    request: ChatRequest,
    pipeline: PipelineDependency,
    db_session: DbSessionDependency,
) -> ChatResponse:
    response = await pipeline.answer(request)
    await _save_audit_best_effort(db_session, request, response)
    return response


@app.post("/v1/chat/stream", dependencies=[Depends(enforce_rate_limit)])
async def chat_stream(
    request: ChatRequest,
    pipeline: PipelineDependency,
    db_session: DbSessionDependency,
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        answer_parts: list[str] = []
        sources: list[SourceCitation] = []
        metadata_event: dict[str, Any] | None = None

        async for event in pipeline.stream_events(request):
            if event.get("type") == "token":
                answer_parts.append(str(event.get("content") or ""))
            elif event.get("type") == "sources":
                sources = [
                    SourceCitation.model_validate(source)
                    for source in event.get("sources", [])
                ]
            elif event.get("type") == "metadata":
                metadata_event = event
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        if metadata_event is not None:
            response = ChatResponse(
                answer="".join(answer_parts),
                session_id=request.session_id,
                trace_id=str(metadata_event.get("trace_id") or ""),
                sources=sources,
                refusal=bool(metadata_event.get("refusal")),
                latency_ms=int(metadata_event.get("latency_ms") or 0),
                metadata=dict(metadata_event.get("metadata") or {}),
            )
            await _save_audit_best_effort(db_session, request, response)
        yield "data: {\"type\":\"done\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _save_audit_best_effort(
    db_session: AsyncSession,
    request: ChatRequest,
    response: ChatResponse,
) -> None:
    try:
        await save_chat_audit(db_session, request, response)
    except Exception as exc:  # pragma: no cover - provider/network boundary
        with suppress(Exception):
            await db_session.rollback()
        logger.warning("chat_audit_failed", trace_id=response.trace_id, error=str(exc))


async def _readiness_checks(request: Request) -> dict[str, dict[str, Any]]:
    return {
        "graph": _check_graph(cast(RagPipeline, request.app.state.pipeline)),
        "qdrant": await _check_qdrant(),
        "database": await _check_database(),
        **(
            {"redis": await _check_redis()}
            if settings.api_rate_limit_backend.strip().casefold() == "redis"
            else {}
        ),
    }


def _check_graph(pipeline: RagPipeline) -> dict[str, Any]:
    graph = pipeline.graph_retriever.graph
    categories = len(graph.categories_by_key)
    products = len(graph.products_by_url)
    detail_sources = len(graph.detail_chunks_by_url)
    if categories == 0 or products == 0 or detail_sources == 0:
        return {
            "status": "failed",
            "categories": categories,
            "products": products,
            "detail_sources": detail_sources,
        }
    return {
        "status": "ok",
        "categories": categories,
        "products": products,
        "detail_sources": detail_sources,
    }


async def _check_qdrant() -> dict[str, Any]:
    try:
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=5,
        )
        try:
            exists = await client.collection_exists(settings.qdrant_collection)
        finally:
            await client.close()
    except Exception as exc:  # pragma: no cover - provider/network boundary
        return {"status": "failed", "error": str(exc)}

    if not exists:
        return {
            "status": "failed",
            "error": f"Qdrant collection not found: {settings.qdrant_collection}",
        }
    return {"status": "ok", "collection": settings.qdrant_collection}


async def _check_database() -> dict[str, Any]:
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("select 1"))
    except Exception as exc:  # pragma: no cover - provider/network boundary
        return {"status": "failed", "error": str(exc)}
    return {"status": "ok"}


async def _check_redis() -> dict[str, Any]:
    try:
        from redis.asyncio import Redis

        redis = Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=1.0,
        )
        try:
            await redis.ping()
        finally:
            await redis.aclose()
    except Exception as exc:  # pragma: no cover - provider/network boundary
        return {"status": "failed", "error": str(exc)}
    return {"status": "ok"}
