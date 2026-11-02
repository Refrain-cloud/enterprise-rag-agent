from __future__ import annotations

import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.config import TOP_K

from .nodes import (
    finalize_node,
    generate_node,
    guard_node,
    prepare_node,
    retrieve_node,
    route_node,
    self_check_node,
    structured_node,
)
from .runtime import CITE_PATTERN
from .state import AgentState, initial_state


def _after_guard(state: AgentState) -> str:
    return "finalize" if state.get("terminal") else "route"


def _after_route(state: AgentState) -> str:
    return "structured" if state.get("route_decision") == "structured" else "generate"


def _after_structured(state: AgentState) -> str:
    return "self_check" if state.get("tool_result") else "generate"


@lru_cache(maxsize=1)
def get_graph():
    builder = StateGraph(AgentState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("guard", guard_node)
    builder.add_node("route", route_node)
    builder.add_node("structured", structured_node)
    builder.add_node("generate", generate_node)
    builder.add_node("self_check", self_check_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "retrieve")
    builder.add_edge("retrieve", "guard")
    builder.add_conditional_edges(
        "guard", _after_guard, {"finalize": "finalize", "route": "route"}
    )
    builder.add_conditional_edges(
        "route",
        _after_route,
        {"structured": "structured", "generate": "generate"},
    )
    builder.add_conditional_edges(
        "structured",
        _after_structured,
        {"self_check": "self_check", "generate": "generate"},
    )
    builder.add_edge("generate", "self_check")
    builder.add_edge("self_check", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def _to_result(state: AgentState) -> dict:
    tool_result = state.get("tool_result")
    if tool_result:
        result = {
            "answer": tool_result.get("answer", ""),
            "sources": tool_result.get("sources", []),
            "hits": state.get("hits", []),
            "denied": False,
            "route": tool_result.get("route", "sql"),
        }
        for key in ("sql", "sql_result", "table_doc_ids", "table_calc"):
            if key in tool_result:
                result[key] = tool_result[key]
    else:
        result = {
            "answer": state.get("answer", ""),
            "sources": state.get("sources", []),
            "hits": state.get("hits", []),
            "denied": state.get("terminal") == "denied",
            "route": state.get("route", "rag"),
        }
        for key in ("format_check", "generation_check"):
            if state.get(key):
                result[key] = state[key]

    answer = result.get("answer", "")
    result["citations"] = sorted(set(CITE_PATTERN.findall(answer)))
    result["self_check"] = state.get("self_check", {})
    result["trace"] = state.get("trace", [])
    result["rewrite"] = state.get("rewrite", {})
    result["retrieval"] = {
        "calls": state.get("retrieval_calls", 0),
        "queries": state.get("retrieval_queries", []),
    }
    if state.get("terminal"):
        result["terminal"] = state["terminal"]
    return result


def run_agent(
    question: str,
    role: str = "普通员工",
    top_k: int = TOP_K,
    debug: bool = False,
    request_id: str | None = None,
) -> dict:
    started = time.perf_counter()
    state = initial_state(
        question,
        role,
        top_k,
        debug=debug,
        request_id=request_id,
    )
    final_state = get_graph().invoke(state)
    result = _to_result(final_state)
    result["trace_meta"] = {
        "request_id": final_state.get("request_id"),
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
        "nodes": len(result.get("trace", [])),
    }
    return result
