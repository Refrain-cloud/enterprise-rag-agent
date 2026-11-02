from __future__ import annotations

import time

from src.config import (
    AGG_TOP_K,
    CANDIDATE_K,
    CHAT_MODEL,
    FOLLOWUP_MAX_TOKENS,
    FOLLOWUP_REWRITE,
    GENERATION_SEED,
    GUARD_DISTANCE_GAP,
    REWRITE_MAX_TOKENS,
    REWRITE_ON_EMPTY,
    TABLE_TOOL_ENABLED,
    TOP_K,
)
from src.llm import chat_completion
from src.retrieval.search import access_gap, hybrid_search
from src.security.policy import can_access, normalize_role
from src.tools.sql_tool import (
    answer_with_scanned_tables,
    answer_with_tables,
    eligible_table_hits,
    is_aggregation_question,
)

from .runtime import (
    CITE_PATTERN,
    contains_control_token,
    contains_system_prompt_leak,
    ensure_draft_disclosure,
    format_history,
    generate_rag_answer,
    generate_rag_answer_stream,
    sources,
)
from .state import AgentState, trace_event
from .streaming import current_sink, emit_status

DENIED_ANSWER = "该信息属于公司机密，当前角色无权访问。"
NOT_FOUND_ANSWER = "知识库中没有找到与该问题相关且你有权访问的资料，无法回答。"

REWRITE_SYSTEM_PROMPT = """你是知识库检索查询改写助手。把用户问题改写成更适合中文向量与关键词检索的查询：
补全指代和省略信息，保留专有名词、编号、日期、部门名与业务术语，可以补充同义词；
不要回答问题，不要解释，不要加引号，只输出一行改写后的查询。"""

FOLLOWUP_SYSTEM_PROMPT = """你是知识库检索查询改写助手。结合给定的对话历史，把最后一条问题改写成一个不依赖上下文、可独立检索的完整问题：
补全代词与省略信息，保留专有名词、编号、日期、部门名与业务术语；历史仅用于消歧，不要回答问题；
不要解释，不要加引号，只输出一行改写后的查询。"""


def _doc_ids(hits: list[dict], limit: int = 8) -> list[str]:
    return [hit["metadata"].get("doc_id", "") for hit in hits[:limit]]


def _rewrite_followup(question: str, history: list[dict]) -> str | None:
    history_text = format_history(history)
    if not history_text:
        return None
    try:
        response = chat_completion(
            "rewrite_followup",
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"对话历史：\n{history_text}\n\n当前问题：{question}",
                },
            ],
            temperature=0,
            max_tokens=FOLLOWUP_MAX_TOKENS,
            seed=GENERATION_SEED,
        )
        text = response.choices[0].message.content.strip().strip("\"'“”")
    except Exception:
        return None
    first_line = text.splitlines()[0].strip() if text else ""
    return first_line or None


def prepare_node(state: AgentState) -> dict:
    started = time.perf_counter()
    emit_status("prepare", "start")
    role = normalize_role(state.get("role", "普通员工"))
    history = state.get("history") or []
    rewritten = state["question"]
    followup = {"triggered": False, "succeeded": False, "query": None}
    if history and FOLLOWUP_REWRITE:
        followup["triggered"] = True
        candidate = _rewrite_followup(state["question"], history)
        followup["query"] = candidate
        if candidate and candidate != state["question"]:
            rewritten = candidate
            followup["succeeded"] = True
    return {
        "role": role,
        "rewritten_question": rewritten,
        "trace": [
            trace_event(
                state,
                "prepare",
                started,
                summary={
                    "role": role,
                    "top_k": state.get("top_k", TOP_K),
                    "history_turns": len(history),
                    "followup_rewrite": followup,
                },
            )
        ],
    }


def _rewrite_query(question: str) -> str:
    response = chat_completion(
        "rewrite",
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=REWRITE_MAX_TOKENS,
        seed=GENERATION_SEED,
    )
    text = response.choices[0].message.content.strip().strip("\"'“”")
    return text.splitlines()[0].strip() if text else ""


def retrieve_node(state: AgentState) -> dict:
    started = time.perf_counter()
    emit_status("retrieve", "start")
    question = state.get("rewritten_question") or state["question"]
    role = state["role"]
    top_k = state.get("top_k", TOP_K)

    hits = hybrid_search(question, role=role, top_k=top_k)
    calls = 1
    queries = [question]
    if (
        TABLE_TOOL_ENABLED
        and is_aggregation_question(question)
        and not eligible_table_hits(hits, role)
    ):
        hits = hybrid_search(question, role=role, top_k=max(top_k, AGG_TOP_K))
        calls += 1

    best_overall = hybrid_search(
        question, role=role, top_k=1, ignore_permissions=True
    )
    calls += 1

    rewrite = {"triggered": False, "query": None, "succeeded": False}
    best_blocked = bool(
        best_overall and not can_access(role, best_overall[0]["metadata"])
    )
    access_margin = None
    if (
        hits
        and best_blocked
        and str(best_overall[0]["metadata"].get("scope", "")).startswith("dept:")
    ):
        access_margin = access_gap(
            question, role, max(top_k * 4, CANDIDATE_K)
        )
        calls += 1
    if not hits and REWRITE_ON_EMPTY and not best_blocked:
        rewritten = _rewrite_query(question)
        if rewritten and rewritten != question:
            calls += 1
            queries.append(rewritten)
            rewrite = {"triggered": True, "query": rewritten, "succeeded": False}
            hits = hybrid_search(rewritten, role=role, top_k=top_k)
            rewrite["succeeded"] = bool(hits)

    emit_status("retrieve", "done", hits=len(hits), calls=calls)
    return {
        "hits": hits,
        "best_overall": best_overall,
        "access_margin": access_margin,
        "retrieval_calls": calls,
        "retrieval_queries": queries,
        "rewrite": rewrite,
        "trace": [
            trace_event(
                state,
                "retrieve",
                started,
                summary={
                    "doc_ids": _doc_ids(hits),
                    "calls": calls,
                    "rewrite": rewrite,
                },
            )
        ],
    }


def guard_node(state: AgentState) -> dict:
    started = time.perf_counter()
    hits = state.get("hits") or []
    best_overall = state.get("best_overall") or []
    role = state["role"]

    def denied() -> dict:
        emit_status("guard", "done", status="denied")
        return {
            "terminal": "denied",
            "answer": DENIED_ANSWER,
            "sources": [],
            "hits": [],
            "route": "rag",
        }

    if not hits:
        if best_overall and not can_access(role, best_overall[0]["metadata"]):
            return {
                **denied(),
                "trace": [
                    trace_event(
                        state,
                        "guard",
                        started,
                        status="denied",
                        summary={"reason": "unauthorized_best_match"},
                    )
                ],
            }
        emit_status("guard", "done", status="not_found")
        return {
            "terminal": "not_found",
            "answer": NOT_FOUND_ANSWER,
            "sources": [],
            "hits": [],
            "route": "rag",
            "trace": [
                trace_event(
                    state,
                    "guard",
                    started,
                    status="not_found",
                    summary={"reason": "no_accessible_hits"},
                )
            ],
        }

    if best_overall and not can_access(role, best_overall[0]["metadata"]):
        if str(best_overall[0]["metadata"].get("scope", "")).startswith("dept:"):
            gap = (state.get("access_margin") or {}).get("gap")
            # 可访问文档与受限最优命中距离接近时，问的是制度本身，按可回答处理
            if gap is None or gap > GUARD_DISTANCE_GAP:
                return {
                    **denied(),
                    "trace": [
                        trace_event(
                            state,
                            "guard",
                            started,
                            status="denied",
                            summary={
                                "reason": "dept_confidential_best_match",
                                "gap": gap,
                            },
                        )
                    ],
                }

    emit_status("guard", "done", status="passed", hits=len(hits))
    return {
        "trace": [
            trace_event(
                state,
                "guard",
                started,
                summary={"passed": True, "hits": len(hits)},
            )
        ]
    }


def route_node(state: AgentState) -> dict:
    started = time.perf_counter()
    question = state.get("rewritten_question") or state["question"]
    aggregation = is_aggregation_question(question)
    decision = "structured" if TABLE_TOOL_ENABLED and aggregation else "rag"
    emit_status("route", "done", decision=decision)
    return {
        "route_decision": decision,
        "trace": [
            trace_event(
                state,
                "route",
                started,
                summary={"decision": decision, "aggregation": aggregation},
            )
        ],
    }


def structured_node(state: AgentState) -> dict:
    started = time.perf_counter()
    question = state.get("rewritten_question") or state["question"]
    role = state["role"]
    hits = state.get("hits") or []

    tool_result = answer_with_tables(question, role, hits)
    if not tool_result:
        tool_result = answer_with_scanned_tables(question, role, hits)

    if not tool_result:
        emit_status("structured", "done", status="fallback")
        return {
            "tool_result": None,
            "trace": [
                trace_event(
                    state,
                    "structured",
                    started,
                    status="fallback",
                    summary={"reason": "no_structured_result"},
                )
            ],
        }

    used_doc_ids = set(tool_result.get("table_doc_ids", []))
    tool_result["hits"] = hits
    tool_result["sources"] = sources(
        [hit for hit in hits if hit["metadata"]["doc_id"] in used_doc_ids]
    )
    tool_result["denied"] = False
    emit_status(
        "structured",
        "done",
        tool=tool_result.get("route", "sql"),
        doc_ids=sorted(used_doc_ids),
    )
    return {
        "tool_result": tool_result,
        "answer": tool_result["answer"],
        "sources": tool_result["sources"],
        "route": tool_result.get("route", "sql"),
        "trace": [
            trace_event(
                state,
                "structured",
                started,
                summary={
                    "tool": tool_result.get("route", "sql"),
                    "doc_ids": sorted(used_doc_ids),
                },
            )
        ],
    }


def generate_node(state: AgentState) -> dict:
    started = time.perf_counter()
    emit_status("generate", "start", hits=len(state.get("hits") or []))
    question = state["question"]
    role = state["role"]
    hits = state.get("hits") or []
    if state.get("stream_sink") is not None or current_sink() is not None:
        generated = generate_rag_answer_stream(
            question, role, hits, state.get("history")
        )
    else:
        generated = generate_rag_answer(question, role, hits, state.get("history"))
    return {
        "answer": generated["text"],
        "sources": sources(hits),
        "route": "rag",
        "format_check": generated["format_check"],
        "generation_check": generated["generation_check"],
        "trace": [
            trace_event(
                state,
                "generate",
                started,
                summary={
                    "doc_ids": _doc_ids(hits),
                    "retried_on_truncation": generated["generation_check"][
                        "retried_on_truncation"
                    ],
                },
            )
        ],
    }


def self_check_node(state: AgentState) -> dict:
    started = time.perf_counter()
    answer = (state.get("answer") or "").strip()
    hits = state.get("hits") or []
    hit_ids = {hit["metadata"]["doc_id"] for hit in hits}
    cited = set(CITE_PATTERN.findall(answer))
    unresolved = sorted(cited - hit_ids)
    empty = not answer
    denied = state.get("terminal") == "denied"
    denied_consistent = not (denied and state.get("sources"))
    prompt_leak = contains_system_prompt_leak(answer)
    unsafe_markup = contains_control_token(answer)

    findings: list[dict] = []
    if empty:
        findings.append({"type": "empty_answer"})
    if unresolved:
        findings.append({"type": "unresolved_citation", "doc_ids": unresolved})
    if not denied_consistent:
        findings.append({"type": "denied_with_sources"})
    if prompt_leak:
        findings.append({"type": "system_prompt_leak"})
    if unsafe_markup:
        findings.append({"type": "unsafe_markup"})

    self_check = {
        "ok": not findings,
        "empty_answer": empty,
        "citations_total": len(cited),
        "unresolved_citations": unresolved,
        "denied_consistent": denied_consistent,
        "system_prompt_leak": prompt_leak,
        "unsafe_markup": unsafe_markup,
        "findings": findings,
    }
    return {
        "self_check": self_check,
        "trace": [
            trace_event(
                state,
                "self_check",
                started,
                status="ok" if self_check["ok"] else "warning",
                summary={
                    "findings": [item["type"] for item in findings],
                    "citations": len(cited),
                },
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    started = time.perf_counter()
    answer = state.get("answer") or ""
    if state.get("terminal") is None and state.get("route") == "rag":
        answer = ensure_draft_disclosure(answer, state.get("hits") or [])
    return {
        "answer": answer,
        "trace": [
            trace_event(
                state,
                "finalize",
                started,
                summary={"route": state.get("route", "rag")},
            )
        ],
    }
