from __future__ import annotations

import csv
import email
import io
from dataclasses import dataclass
from email import policy
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class ParseResult:
    text: str
    error: str = ""
    scanned: bool = False


class _HTMLText(HTMLParser):
    BLOCK_TAGS = {
        "p", "br", "div", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if line.strip():
            out.append(line)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLText()
    parser.feed(html)
    return _clean("".join(parser.parts))


def _parse_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        parts.append(f"\n## {text}" if style.startswith("heading") else text)
    for idx, table in enumerate(doc.tables, 1):
        parts.append(f"\n## 表格 {idx}")
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return _clean("\n".join(parts))


def _parse_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts: list[str] = []
    for idx, slide in enumerate(prs.slides, 1):
        parts.append(f"\n## 幻灯片 {idx}")
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
    return _clean("\n".join(parts))


def _parse_xlsx(path: Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    parts: list[str] = []
    try:
        for sheet in wb.worksheets:
            parts.append(f"\n## {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if value is None else str(value) for value in row]
                if any(cell.strip() for cell in cells):
                    parts.append(" | ".join(cells))
    finally:
        wb.close()
    return _clean("\n".join(parts))


def _parse_csv(path: Path) -> str:
    text = _read_text(path)
    rows = csv.reader(io.StringIO(text))
    return _clean("\n".join(" | ".join(cell.strip() for cell in row) for row in rows if any(cell.strip() for cell in row)))


def _parse_eml(path: Path) -> str:
    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    parts = [
        f"主题：{msg.get('subject', '')}",
        f"发件人：{msg.get('from', '')}",
        f"收件人：{msg.get('to', '')}",
        f"日期：{msg.get('date', '')}",
        "",
    ]
    body_part = msg.get_body(preferencelist=("plain", "html"))
    if body_part is not None:
        body = body_part.get_content()
        if body_part.get_content_type() == "text/html":
            body = _html_to_text(body)
        lines = [line for line in body.splitlines() if not line.strip().startswith(">")]
        parts.extend(lines)
    return _clean("\n".join(parts))


def _parse_pdf(path: Path) -> ParseResult:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for idx, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"## 第 {idx} 页\n{text}")
    if not pages:
        return ParseResult(text="", scanned=True, error="no_text_layer")
    return ParseResult(text=_clean("\n\n".join(pages)))


def parse_file(path: Path) -> ParseResult:
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            return ParseResult(text=_parse_docx(path))
        if suffix == ".pptx":
            return ParseResult(text=_parse_pptx(path))
        if suffix == ".xlsx":
            return ParseResult(text=_parse_xlsx(path))
        if suffix == ".csv":
            return ParseResult(text=_parse_csv(path))
        if suffix == ".eml":
            return ParseResult(text=_parse_eml(path))
        if suffix == ".pdf":
            return _parse_pdf(path)
        if suffix in (".html", ".htm"):
            return ParseResult(text=_html_to_text(_read_text(path)))
        if suffix in (".md", ".txt", ".json"):
            return ParseResult(text=_clean(_read_text(path)))
        return ParseResult(text="", error=f"unsupported_format:{suffix}")
    except Exception as exc:  # parsing one bad file should not stop the corpus build
        return ParseResult(text="", error=f"{type(exc).__name__}: {exc}")
