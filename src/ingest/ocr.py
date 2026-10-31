from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from pathlib import Path

from src.config import OCR_DIR, OCR_MODEL, get_client

PROMPT = """你是文档 OCR 引擎。请逐字转写图片中的全部可见文字，按原文顺序输出 Markdown。

要求：
- 保留标题、编号、日期、金额、单位与标点，不要改写、总结或补充；
- 表格必须输出为 Markdown 表格，行列对齐；合并单元格按行列语义拆成逐行；
- 表格跨页时续页也要补上表头并输出 Markdown 表格，不得退化成纯文本行；
- 原文中同一单元格内的换行用空格连接；
- 印章、公章、手写签名中的文字若清晰可辨，在出现位置用“（印章：…）”或“（签名：…）”标注；不可辨认时写“（印章不可辨）”“（签名不可辨）”，不要编造；
- 看不清的字用 [无法辨认] 标记，不要根据上下文猜测；
- 只输出转写内容，不要加解释，不要用代码围栏包裹。"""

MAX_TOKENS = 4096


def _prompt_hash() -> str:
    return hashlib.sha256(f"{OCR_MODEL}\n{PROMPT}".encode()).hexdigest()[:16]


def _mime(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix in ("jpg", "jpeg"):
        return "jpeg"
    if suffix == "png":
        return "png"
    from PIL import Image

    fmt = (Image.open(io.BytesIO(data)).format or "PNG").lower()
    return "jpeg" if fmt == "jpg" else fmt


def _page_images(path: Path) -> list[tuple[int, list[tuple[bytes, str]]]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, list[tuple[bytes, str]]]] = []
    for idx, page in enumerate(reader.pages, 1):
        blobs = [(image.data, _mime(image.name, image.data)) for image in page.images]
        pages.append((idx, blobs))
    return pages


def _strip_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def _transcribe(data: bytes, mime: str) -> str:
    client = get_client()
    url = f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    max_tokens = MAX_TOKENS
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=180,
            )
            choice = response.choices[0]
            text = _strip_fence(choice.message.content or "")
            if not text:
                raise RuntimeError("empty OCR output")
            if choice.finish_reason == "length":
                if attempt == 0:
                    max_tokens *= 2
                    continue
                raise RuntimeError("ocr_truncated")
            return text
        except Exception as exc:  # network/rate-limit/truncation retries
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"ocr_failed: {last_error}")


def _cache_paths(doc_id: str, page_no: int) -> tuple[Path, Path]:
    folder = OCR_DIR / doc_id
    return folder / f"p{page_no}.md", folder / f"p{page_no}.json"


def _load_cache(doc_id: str, page_no: int, digest: str) -> str | None:
    md_path, meta_path = _cache_paths(doc_id, page_no)
    if not md_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta.get("digest") != digest or meta.get("prompt") != _prompt_hash():
        return None
    return md_path.read_text(encoding="utf-8")


def _save_cache(doc_id: str, page_no: int, digest: str, text: str) -> None:
    md_path, meta_path = _cache_paths(doc_id, page_no)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {"digest": digest, "prompt": _prompt_hash(), "model": OCR_MODEL},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def ocr_pdf(path: Path, doc_id: str) -> tuple[str, list[str]]:
    """Transcribe every page image to Markdown. Returns (text, per-page errors)."""
    pages = _page_images(path)
    if not pages:
        return "", ["ocr_no_pages"]

    parts: list[str] = []
    errors: list[str] = []
    for page_no, blobs in pages:
        if not blobs:
            errors.append(f"p{page_no}: no_page_image")
            continue
        digest = hashlib.sha256(b"".join(data for data, _ in blobs)).hexdigest()
        cached = _load_cache(doc_id, page_no, digest)
        if cached is not None:
            parts.append(f"## 第 {page_no} 页\n{cached}")
            continue
        try:
            page_text = "\n\n".join(_transcribe(data, mime) for data, mime in blobs)
        except Exception as exc:
            errors.append(f"p{page_no}: {exc}")
            continue
        _save_cache(doc_id, page_no, digest, page_text)
        parts.append(f"## 第 {page_no} 页\n{page_text}")
    return "\n\n".join(parts), errors
