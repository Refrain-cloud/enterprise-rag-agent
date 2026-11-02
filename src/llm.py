from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from typing import Any

from src.config import get_client

# 人民币元 / 百万 tokens，取自阿里云百炼按量计费页（2026-11 核对）。
# 价格会调整，可用 RAG_PRICE_PER_MILLION 传入 JSON 覆盖，例如
# {"qwen-plus": {"input": 0.8, "output": 2.0}}
DEFAULT_PRICES = {
    "qwen-plus": {"input": 0.8, "output": 2.0},
    "text-embedding-v3": {"input": 0.5},
    "gte-rerank-v2": {"input": 0.8},
}

_LOCK = threading.Lock()
_RECORDS: list[dict[str, Any]] = []
# 服务层按请求收集用量：ContextVar 里的 list 会被 to_thread 复制的上下文共享。
_REQUEST_RECORDS: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("request_usage_records", default=None)
)


def _prices() -> dict[str, dict[str, float]]:
    raw = os.getenv("RAG_PRICE_PER_MILLION")
    if not raw:
        return DEFAULT_PRICES
    try:
        override = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_PRICES
    prices = {model: dict(value) for model, value in DEFAULT_PRICES.items()}
    for model, value in override.items():
        prices[model] = {**prices.get(model, {}), **value}
    return prices


def _price_for(model: str) -> dict[str, float] | None:
    prices = _prices()
    if model in prices:
        return prices[model]
    # 带日期或快照后缀的模型名回退到基础模型
    for name, price in prices.items():
        if model.startswith(name):
            return price
    return None


def record_usage(
    kind: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    elapsed_s: float = 0.0,
) -> None:
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    record = {
        "kind": kind,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(total_tokens or prompt_tokens + completion_tokens),
        "elapsed_s": round(float(elapsed_s), 4),
    }
    with _LOCK:
        _RECORDS.append(record)
    request_records = _REQUEST_RECORDS.get()
    if request_records is not None:
        request_records.append(record)


def start_request_usage() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    _REQUEST_RECORDS.set(records)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize(records)


def _record_response(kind: str, model: str, response: Any, elapsed_s: float) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        record_usage(kind, model, elapsed_s=elapsed_s)
        return
    record_usage(
        kind,
        model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", None),
        elapsed_s=elapsed_s,
    )


def chat_completion(kind: str, **kwargs: Any):
    model = kwargs.get("model", "")
    started = time.perf_counter()
    response = get_client().chat.completions.create(**kwargs)
    _record_response(kind, model, response, time.perf_counter() - started)
    return response


def embed(kind: str, **kwargs: Any):
    model = kwargs.get("model", "")
    started = time.perf_counter()
    response = get_client().embeddings.create(**kwargs)
    _record_response(kind, model, response, time.perf_counter() - started)
    return response


def reset_usage() -> None:
    with _LOCK:
        _RECORDS.clear()


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = _price_for(model)
    if price is None:
        return None
    return (
        prompt_tokens / 1_000_000 * price.get("input", 0.0)
        + completion_tokens / 1_000_000 * price.get("output", 0.0)
    )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "calls": len(records),
        "prompt_tokens": sum(record["prompt_tokens"] for record in records),
        "completion_tokens": sum(record["completion_tokens"] for record in records),
        "total_tokens": sum(record["total_tokens"] for record in records),
        "cost_cny": 0.0,
        "elapsed_s": round(sum(record["elapsed_s"] for record in records), 4),
    }
    unpriced: set[str] = set()
    for record in records:
        cost = _cost(record["model"], record["prompt_tokens"], record["completion_tokens"])
        if cost is None:
            unpriced.add(record["model"])
        else:
            summary["cost_cny"] += cost
    summary["cost_cny"] = round(summary["cost_cny"], 6)
    if unpriced:
        summary["unpriced_models"] = sorted(unpriced)
    return summary


def usage_snapshot() -> dict[str, Any]:
    with _LOCK:
        records = list(_RECORDS)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_kind.setdefault(record["kind"], []).append(record)
        by_model.setdefault(record["model"], []).append(record)
    summary = _summarize(records)
    summary["by_kind"] = {kind: _summarize(items) for kind, items in sorted(by_kind.items())}
    summary["by_model"] = {model: _summarize(items) for model, items in sorted(by_model.items())}
    return summary


def merge_usage(*snapshots: dict[str, Any]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    by_model: dict[str, list[dict[str, Any]]] = {}
    unpriced: set[str] = set()
    for snapshot in snapshots:
        if not snapshot:
            continue
        for key in ("by_kind", "by_model"):
            for name, item in (snapshot.get(key) or {}).items():
                target = by_kind if key == "by_kind" else by_model
                target.setdefault(name, []).append(item)
        unpriced.update(snapshot.get("unpriced_models") or [])
    merged = {
        "calls": sum(snapshot.get("calls", 0) for snapshot in snapshots if snapshot),
        "prompt_tokens": sum(snapshot.get("prompt_tokens", 0) for snapshot in snapshots if snapshot),
        "completion_tokens": sum(
            snapshot.get("completion_tokens", 0) for snapshot in snapshots if snapshot
        ),
        "total_tokens": sum(snapshot.get("total_tokens", 0) for snapshot in snapshots if snapshot),
        "cost_cny": round(
            sum(snapshot.get("cost_cny", 0.0) for snapshot in snapshots if snapshot), 6
        ),
        "elapsed_s": round(
            sum(snapshot.get("elapsed_s", 0.0) for snapshot in snapshots if snapshot), 4
        ),
        "by_kind": {name: _merge_summaries(items) for name, items in sorted(by_kind.items())},
        "by_model": {name: _merge_summaries(items) for name, items in sorted(by_model.items())},
    }
    if unpriced:
        merged["unpriced_models"] = sorted(unpriced)
    return merged


def _merge_summaries(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "calls": sum(item.get("calls", 0) for item in items),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in items),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in items),
        "total_tokens": sum(item.get("total_tokens", 0) for item in items),
        "cost_cny": round(sum(item.get("cost_cny", 0.0) for item in items), 6),
        "elapsed_s": round(sum(item.get("elapsed_s", 0.0) for item in items), 4),
    }
    unpriced: set[str] = set()
    for item in items:
        unpriced.update(item.get("unpriced_models") or [])
    if unpriced:
        merged["unpriced_models"] = sorted(unpriced)
    return merged
