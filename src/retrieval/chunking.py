from __future__ import annotations

import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

TABLE_FORMATS = {"csv", "xlsx"}
TABLE_CHUNK_SIZE = 2200


def _split_long(block: str, size: int) -> list[str]:
    if len(block) <= size:
        return [block]
    pieces: list[str] = []
    buf = ""
    for line in block.split("\n"):
        if len(line) > size:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(line[i : i + size] for i in range(0, len(line), size))
        elif len(buf) + len(line) + 1 <= size:
            buf = f"{buf}\n{line}" if buf else line
        else:
            pieces.append(buf)
            buf = line
    if buf:
        pieces.append(buf)
    return pieces


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(_split_long(block, size))

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if not buf:
            buf = piece
        elif len(buf) + len(piece) + 2 <= size:
            buf = f"{buf}\n\n{piece}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            candidate = f"{tail}\n\n{piece}".strip() if tail else piece
            buf = candidate if len(candidate) <= int(size * 1.3) else piece
    if buf:
        chunks.append(buf)
    return chunks


def chunk_doc(doc: dict) -> list[dict]:
    chunks: list[dict] = []
    # 表格整表保留：拆开后模型容易只看到部分行，导致统计漏项。
    size = TABLE_CHUNK_SIZE if doc["format"] in TABLE_FORMATS else CHUNK_SIZE
    for index, text in enumerate(split_text(doc["text"], size=size)):
        metadata = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "path": doc["path"],
            "format": doc["format"],
            "dept": doc["dept"],
            "scope": doc["scope"],
            "classification": doc["classification"],
            "version": doc["version"],
            "effective_date": doc["effective_date"],
            "status": doc["status"],
            "chunk_index": index,
        }
        chunks.append(
            {
                "id": f"{doc['doc_id']}#{index:03d}",
                "text": text,
                "embed_text": f"{doc['title']}（{doc['doc_id']}）\n{text}",
                "metadata": metadata,
            }
        )
    return chunks
