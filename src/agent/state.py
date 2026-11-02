from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, TypedDict


class TraceEvent(TypedDict, total=False):
    node: str
    seq: int
    elapsed_ms: float
    status: str
    summary: dict[str, Any]
    error: str


class AgentState(TypedDict, total=False):
    question: str
    role: str
    top_k: int
    debug: bool
    request_id: str
    started_at: float

    history: list[dict]
    rewritten_question: str
    session_id: str
    turn_index: int
    stream_sink: Any

    hits: list[dict]
    best_overall: list[dict]
    access_margin: dict | None
    retrieval_calls: int
    retrieval_queries: list[str]
    rewrite: dict

    terminal: str
    route_decision: str
    route: str
    tool_result: dict | None

    answer: str
    sources: list[dict]
    citations: list[str]
    format_check: dict | None
    generation_check: dict | None
    self_check: dict

    trace: Annotated[list[TraceEvent], operator.add]
    error: str


def initial_state(
    question: str,
    role: str,
    top_k: int,
    debug: bool = False,
    request_id: str | None = None,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
    turn_index: int = 0,
    stream_sink: Any = None,
) -> AgentState:
    state: AgentState = {
        "question": question,
        "role": role,
        "top_k": top_k,
        "debug": debug,
        "request_id": request_id or uuid.uuid4().hex[:12],
        "started_at": time.time(),
        "turn_index": turn_index,
        "trace": [],
    }
    if history:
        state["history"] = list(history)
    if session_id:
        state["session_id"] = session_id
    if stream_sink is not None:
        state["stream_sink"] = stream_sink
    return state


def trace_event(
    state: AgentState,
    node: str,
    started: float,
    status: str = "ok",
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict:
    event: TraceEvent = {
        "node": node,
        "seq": len(state.get("trace", [])) + 1,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "status": status,
    }
    if summary:
        event["summary"] = summary
    if error:
        event["error"] = error
    return event
