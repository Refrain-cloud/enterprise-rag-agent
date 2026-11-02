from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.agent.ask import answer
from src.config import (
    API_MAX_CONCURRENCY,
    API_QUEUE_TIMEOUT_S,
    API_REQUEST_TIMEOUT_S,
    ROLES,
    TOP_K,
)
from src.llm import start_request_usage, summarize_records
from src.security.policy import normalize_role

logger = logging.getLogger("rag.api")
UI_DIR = Path(__file__).resolve().parents[1] / "ui"


def _warm_up() -> None:
    import jieba

    from src.retrieval.search import _load_resources
    from src.tools.table_loader import load_corpus

    jieba.initialize()
    _load_resources()
    load_corpus()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.semaphore = asyncio.Semaphore(API_MAX_CONCURRENCY)
    app.state.index_ready = False
    app.state.warmup_error = None
    try:
        await asyncio.to_thread(_warm_up)
        app.state.index_ready = True
        logger.info("index warmed up")
    except Exception as exc:  # 索引缺失时保留健康检查可见的错误
        app.state.warmup_error = f"{type(exc).__name__}: {exc}"
        logger.exception("warm-up failed")
    yield


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    role: str = "普通员工"
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    debug: bool = False

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value

    @field_validator("role")
    @classmethod
    def _valid_role(cls, value: str) -> str:
        if value not in ROLES:
            raise ValueError(f"role 必须是以下之一：{'、'.join(ROLES)}")
        return normalize_role(value)


class SourceModel(BaseModel):
    doc_id: str
    title: str = ""
    version: str = ""
    status: str = ""
    effective_date: str = ""
    path: str = ""
    classification: str = ""


class AskResponse(BaseModel):
    request_id: str
    answer: str
    route: str
    denied: bool
    sources: list[SourceModel]
    citations: list[str]
    latency_ms: float
    usage: dict
    trace: list[dict] | None = None
    self_check: dict | None = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="北辰智造知识库 RAG Agent",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:"
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        return {
            "status": "ok",
            "index_ready": request.app.state.index_ready,
            "warmup_error": request.app.state.warmup_error,
        }

    @app.get("/api/roles")
    async def roles() -> dict:
        return {"roles": list(ROLES)}

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        semaphore: asyncio.Semaphore = request.app.state.semaphore

        try:
            await asyncio.wait_for(
                semaphore.acquire(), timeout=API_QUEUE_TIMEOUT_S
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=429, detail="服务繁忙，请稍后重试"
            ) from exc

        try:
            records = start_request_usage()
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        answer,
                        payload.question,
                        role=payload.role,
                        top_k=payload.top_k,
                        request_id=request_id,
                    ),
                    timeout=API_REQUEST_TIMEOUT_S,
                )
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=504, detail="问答超时，请缩小问题范围后重试"
                ) from exc
            except Exception as exc:
                logger.exception("ask failed request_id=%s", request_id)
                raise HTTPException(
                    status_code=500, detail="内部错误，请稍后重试"
                ) from exc
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
        finally:
            semaphore.release()

        logger.info(
            "ask request_id=%s role=%s route=%s denied=%s latency_ms=%.1f",
            request_id,
            payload.role,
            result.get("route"),
            bool(result.get("denied")),
            latency_ms,
        )
        return AskResponse(
            request_id=request_id,
            answer=result.get("answer", ""),
            route=result.get("route", "rag"),
            denied=bool(result.get("denied")),
            sources=result.get("sources") or [],
            citations=result.get("citations") or [],
            latency_ms=latency_ms,
            usage=summarize_records(records),
            trace=result.get("trace") if payload.debug else None,
            self_check=result.get("self_check") if payload.debug else None,
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    return app


app = create_app()
