from __future__ import annotations

import json
import pickle
from functools import lru_cache

import chromadb

from src.config import (
    CANDIDATE_K,
    CHROMA_DIR,
    EMBED_DIM,
    EMBED_MODEL,
    INDEX_DIR,
    TOP_K,
    get_client,
)
from src.retrieval.index import COLLECTION_NAME
from src.security.policy import allowed_classifications, can_access


@lru_cache(maxsize=1)
def _load_resources():
    chunks = [
        json.loads(line)
        for line in (INDEX_DIR / "chunks.jsonl").open(encoding="utf-8")
    ]
    by_id = {chunk["id"]: chunk for chunk in chunks}
    with (INDEX_DIR / "bm25.pkl").open("rb") as fh:
        bm25_payload = pickle.load(fh)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_collection(COLLECTION_NAME)
    return by_id, bm25_payload, collection


def _embed_query(query: str) -> list[float]:
    response = get_client().embeddings.create(
        model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM
    )
    return response.data[0].embedding


def _rrf_fuse(rankings: list[list[str]], weights: list[float] | None = None) -> dict[str, float]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (60 + rank)
    return scores


def hybrid_search(
    query: str,
    role: str = "普通员工",
    top_k: int = TOP_K,
    ignore_permissions: bool = False,
) -> list[dict]:
    by_id, bm25_payload, collection = _load_resources()
    query_embedding = _embed_query(query)

    where = None
    if not ignore_permissions:
        where = {"classification": {"$in": allowed_classifications(role)}}

    vector = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k * 4, CANDIDATE_K),
        where=where,
        include=["metadatas", "distances"],
    )
    vector_ids = vector["ids"][0] if vector["ids"] else []

    import jieba

    jieba.setLogLevel(20)
    query_tokens = list(jieba.cut_for_search(query))
    scores = bm25_payload["bm25"].get_scores(query_tokens)
    bm25_ids = [bm25_payload["ids"][i] for i in scores.argsort()[::-1]]

    vector_ranked: list[str] = []
    bm25_ranked: list[str] = []
    for chunk_id in vector_ids:
        chunk = by_id.get(chunk_id)
        if not chunk:
            continue
        if ignore_permissions or can_access(role, chunk["metadata"]):
            vector_ranked.append(chunk_id)
    for chunk_id in bm25_ids:
        chunk = by_id.get(chunk_id)
        if not chunk:
            continue
        if ignore_permissions or can_access(role, chunk["metadata"]):
            bm25_ranked.append(chunk_id)
        if len(bm25_ranked) >= max(top_k * 4, CANDIDATE_K):
            break

    fused = _rrf_fuse(
        [vector_ranked[: CANDIDATE_K], bm25_ranked[: CANDIDATE_K]],
        weights=[1.0, 0.8],
    )
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results: list[dict] = []
    for chunk_id, score in ordered:
        chunk = by_id[chunk_id]
        results.append(
            {
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": score,
            }
        )
    return results
