from __future__ import annotations

import argparse
import json
import pickle
import time

import chromadb

from src.config import (
    CHROMA_DIR,
    CORPUS_DIR,
    EMBED_BATCH,
    EMBED_DIM,
    EMBED_MODEL,
    INDEX_DIR,
    ensure_build_dirs,
    get_client,
)
from src.retrieval.chunking import chunk_doc

COLLECTION_NAME = "kb_chunks"


def _load_chunks(rebuild: bool) -> list[dict]:
    chunks_path = INDEX_DIR / "chunks.jsonl"
    if chunks_path.exists() and not rebuild:
        return [json.loads(line) for line in chunks_path.open(encoding="utf-8")]

    docs = [
        json.loads(line)
        for line in (CORPUS_DIR / "docs.jsonl").open(encoding="utf-8")
    ]
    chunks: list[dict] = []
    for doc in docs:
        if doc["chars"] <= 0:
            continue
        chunks.extend(chunk_doc(doc))
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return chunks


def _embed_batch(client, texts: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"embedding failed: {last_error}")


def _build_chroma(chunks: list[dict], rebuild: bool) -> int:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if collection.count() == len(chunks) and not rebuild:
        print(f"chroma already has {collection.count()} chunks, skip")
        return collection.count()

    api = get_client()
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        embeddings = _embed_batch(api, [chunk["embed_text"] for chunk in batch])
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )
        print(f"embedded {min(start + EMBED_BATCH, len(chunks))}/{len(chunks)}")
    return collection.count()


def _build_bm25(chunks: list[dict]) -> None:
    import jieba
    from rank_bm25 import BM25Okapi

    jieba.setLogLevel(20)
    tokenized = [list(jieba.cut_for_search(chunk["text"])) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "ids": [chunk["id"] for chunk in chunks],
        "tokens": tokenized,
        "bm25": bm25,
    }
    with (INDEX_DIR / "bm25.pkl").open("wb") as fh:
        pickle.dump(payload, fh)
    print(f"bm25 built: {len(tokenized)} chunks")


def build_index(rebuild: bool = False) -> None:
    ensure_build_dirs()
    chunks = _load_chunks(rebuild)
    print(f"chunks: {len(chunks)}")
    count = _build_chroma(chunks, rebuild)
    _build_bm25(chunks)
    summary = {"chunks": len(chunks), "chroma_count": count}
    (INDEX_DIR / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)
