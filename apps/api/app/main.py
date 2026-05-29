from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from apps.api.app.core.config import get_settings
from apps.api.app.core.logging import configure_logging, get_logger
from apps.api.app.core.rate_limit import InMemoryRateLimiter
from apps.api.app.models.chat import ChatRequest, ChatResponse
from apps.api.app.rag.pipeline import RagPipeline

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RagPipeline(settings)
    app.state.rate_limiter = InMemoryRateLimiter(settings.api_rate_limit_per_minute)
    logger.info("api_started", env=settings.app_env)
    yield
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
    limiter: InMemoryRateLimiter = request.app.state.rate_limiter
    client_host = request.client.host if request.client else "unknown"
    key = request.headers.get("Authorization") or client_host
    if not await limiter.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
        )


def get_pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    return {
        "status": "ready",
        "collection": settings.qdrant_collection,
        "env": settings.app_env,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
async def chat(request: ChatRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> ChatResponse:
    return await pipeline.answer(request)


@app.post("/v1/chat/stream", dependencies=[Depends(enforce_rate_limit)])
async def chat_stream(
    request: ChatRequest,
    pipeline: RagPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    async def event_stream():
        async for token in pipeline.stream(request):
            payload = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
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
