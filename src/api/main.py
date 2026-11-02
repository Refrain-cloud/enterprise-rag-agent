from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.agent.ask import answer
from src.agent.session import SessionError, SessionStore, SessionView
from src.agent.streaming import StreamSink, use_sink
from src.config import (
    API_MAX_CONCURRENCY,
    API_QUEUE_TIMEOUT_S,
    API_REQUEST_TIMEOUT_S,
    ROLES,
    STREAM_HEARTBEAT_S,
    TOP_K,
)
from src.llm import start_request_usage, summarize_records
from src.security.policy import normalize_role

logger = logging.getLogger("rag.api")
UI_DIR = Path(__file__).resolve().parents[1] / "ui"
_DONE = object()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _begin_session(store: SessionStore, payload: "AskRequest") -> SessionView:
    if payload.session_id:
        return store.begin_turn(payload.session_id, payload.role)
    created = store.create(payload.role)
    return store.begin_turn(created.session_id, payload.role)


def _session_payload(store: SessionStore, session_id: str) -> dict | None:
    try:
        return store.snapshot(session_id).as_dict()
    except SessionError:
        return None


def _response_payload(
    request_id: str,
    payload: "AskRequest",
    result: dict,
    latency_ms: float,
    usage: dict,
    session: dict | None,
) -> dict:
    return {
        "request_id": request_id,
        "answer": result.get("answer", ""),
        "route": result.get("route", "rag"),
        "denied": bool(result.get("denied")),
        "sources": result.get("sources") or [],
        "citations": result.get("citations") or [],
        "latency_ms": latency_ms,
        "usage": usage,
        "trace": result.get("trace") if payload.debug else None,
        "self_check": result.get("self_check") if payload.debug else None,
        "session": session,
    }


async def _session_sweeper(store: SessionStore) -> None:
    while True:
        await asyncio.sleep(300)
        try:
            await asyncio.to_thread(store.sweep)
        except Exception:
            logger.exception("session sweep failed")


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
    app.state.sessions = SessionStore()
    app.state.session_sweeper = asyncio.create_task(
        _session_sweeper(app.state.sessions)
    )
    app.state.index_ready = False
    app.state.warmup_error = None
    try:
        await asyncio.to_thread(_warm_up)
        app.state.index_ready = True
        logger.info("index warmed up")
    except Exception as exc:  # 索引缺失时保留健康检查可见的错误
        app.state.warmup_error = f"{type(exc).__name__}: {exc}"
        logger.exception("warm-up failed")
    try:
        yield
    finally:
        app.state.session_sweeper.cancel()
        try:
            await app.state.session_sweeper
        except asyncio.CancelledError:
            pass


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    role: str = "普通员工"
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    debug: bool = False
    session_id: str | None = Field(default=None, max_length=64)

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
    session: dict | None = None


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
        store: SessionStore = request.app.state.sessions

        try:
            await asyncio.wait_for(
                semaphore.acquire(), timeout=API_QUEUE_TIMEOUT_S
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=429, detail="服务繁忙，请稍后重试"
            ) from exc

        session_id: str | None = None
        turn_token: str | None = None
        try:
            try:
                view = _begin_session(store, payload)
            except SessionError as exc:
                raise HTTPException(
                    status_code=exc.status_code, detail=exc.message
                ) from exc
            session_id = view.session_id
            turn_token = view.turn_token

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
                        history=view.history_payload() or None,
                        session_id=view.session_id,
                        turn_index=view.turn_count,
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
            usage = summarize_records(records)
            store.finish_turn(
                view.session_id,
                question=payload.question,
                answer=result.get("answer", ""),
                rewritten_question=result.get("rewritten_question")
                or payload.question,
                usage=usage,
                token=turn_token,
            )
            session = _session_payload(store, view.session_id)
        finally:
            if session_id is not None:
                store.end_turn(session_id, turn_token)
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
            **_response_payload(
                request_id, payload, result, latency_ms, usage, session
            )
        )

    @app.post("/api/ask/stream")
    async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        semaphore: asyncio.Semaphore = request.app.state.semaphore
        store: SessionStore = request.app.state.sessions

        try:
            await asyncio.wait_for(
                semaphore.acquire(), timeout=API_QUEUE_TIMEOUT_S
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=429, detail="服务繁忙，请稍后重试"
            ) from exc

        try:
            view = _begin_session(store, payload)
        except SessionError as exc:
            semaphore.release()
            raise HTTPException(
                status_code=exc.status_code, detail=exc.message
            ) from exc
        except Exception:
            semaphore.release()
            raise

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sink = StreamSink(loop, queue)
        session_id = view.session_id
        turn_token = view.turn_token
        history = view.history_payload() or None
        turn_count = view.turn_count

        def worker() -> None:
            records = start_request_usage()
            started = time.perf_counter()
            try:
                with use_sink(sink):
                    result = answer(
                        payload.question,
                        role=payload.role,
                        top_k=payload.top_k,
                        request_id=request_id,
                        history=history,
                        session_id=session_id,
                        turn_index=turn_count,
                        stream_sink=sink,
                    )
                latency_ms = round((time.perf_counter() - started) * 1000, 1)
                usage = summarize_records(records)
                store.finish_turn(
                    session_id,
                    question=payload.question,
                    answer=result.get("answer", ""),
                    rewritten_question=result.get("rewritten_question")
                    or payload.question,
                    usage=usage,
                    token=turn_token,
                )
                sink.emit(
                    "final",
                    _response_payload(
                        request_id,
                        payload,
                        result,
                        latency_ms,
                        usage,
                        _session_payload(store, session_id),
                    ),
                )
            except Exception:
                logger.exception("stream ask failed request_id=%s", request_id)
                sink.emit(
                    "error",
                    {
                        "code": 500,
                        "message": "内部错误，请稍后重试",
                        "request_id": request_id,
                        "session_id": session_id,
                    },
                )
                store.end_turn(session_id, turn_token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        task = asyncio.create_task(asyncio.to_thread(worker))

        async def event_stream():
            deadline = loop.time() + API_REQUEST_TIMEOUT_S
            try:
                yield _sse(
                    "meta",
                    {
                        "request_id": request_id,
                        "session_id": session_id,
                        "role": payload.role,
                        "turn": turn_count + 1,
                        "debug": payload.debug,
                    },
                )
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        yield _sse(
                            "error",
                            {
                                "code": 504,
                                "message": "问答超时，请缩小问题范围后重试",
                                "request_id": request_id,
                                "session_id": session_id,
                            },
                        )
                        break
                    try:
                        item = await asyncio.wait_for(
                            queue.get(),
                            timeout=min(STREAM_HEARTBEAT_S, remaining),
                        )
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    if item is _DONE:
                        break
                    yield _sse(item["event"], item["data"])
                    if item["event"] in {"final", "error"}:
                        break
            except asyncio.CancelledError:
                raise
            finally:
                sink.close()
                if not task.done():
                    task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                # 只释放本轮的 busy；晚到的 worker 用旧 token 收尾时不会影响下一轮
                store.end_turn(session_id, turn_token)
                semaphore.release()

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    return app


app = create_app()
