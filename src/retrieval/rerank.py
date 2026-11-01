from __future__ import annotations

import logging
import time

import httpx

from src.config import RERANK_BASE_URL, RERANK_MODEL, get_api_key
from src.llm import record_usage

LOGGER = logging.getLogger(__name__)
_FAILURE_LOGGED = False


def rerank_documents(
    query: str, documents: list[str], top_n: int, timeout: float = 20.0
) -> list[tuple[int, float]] | None:
    """按相关性重排候选，返回 [(原下标, 分数)]；接口不可用时返回 None 让调用方回退。"""
    global _FAILURE_LOGGED
    if not documents:
        return []
    body = {
        "model": RERANK_MODEL,
        "input": {"query": query, "documents": documents},
        "parameters": {"return_documents": False, "top_n": min(top_n, len(documents))},
    }
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    for attempt in range(2):
        started = time.perf_counter()
        try:
            response = httpx.post(
                RERANK_BASE_URL, headers=headers, json=body, timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") or {}
            record_usage(
                "rerank",
                RERANK_MODEL,
                prompt_tokens=usage.get("total_tokens", 0) or 0,
                elapsed_s=time.perf_counter() - started,
            )
            results = (payload.get("output") or {}).get("results") or []
            ranked = [
                (int(item["index"]), float(item.get("relevance_score", 0.0)))
                for item in results
                if 0 <= int(item.get("index", -1)) < len(documents)
            ]
            if ranked:
                ranked.sort(key=lambda item: item[1], reverse=True)
                return ranked
        except Exception as exc:
            if attempt == 0:
                time.sleep(1.0)
                continue
            if not _FAILURE_LOGGED:
                LOGGER.warning("rerank 调用失败，回退 RRF 顺序：%s", exc)
                _FAILURE_LOGGED = True
    return None
