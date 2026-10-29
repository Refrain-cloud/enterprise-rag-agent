from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import (
    CORPUS_DIR,
    DATA_DIR,
    TABLE_FORMATS,
    TABLE_HEADER_SCAN_ROWS,
)
from src.ingest.parsers import _read_text
from src.security.policy import can_access

SUMMARY_RE = re.compile(r"^(合\s*计|小\s*计|总\s*计|汇\s*总)([:：].*)?$")
BLANK_NUMBERS = {"", "-", "—", "/", "n/a", "N/A", "暂无"}


@dataclass
class TableSpec:
    name: str
    doc_id: str
    title: str
    sheet: str
    kind: str  # detail | summary
    columns: list[str]
    types: list[str]
    units: list[str]
    rows: list[tuple]


@dataclass
class LoadedTables:
    tables: list[TableSpec]
    notes: list[str]
    doc_ids: list[str]


@lru_cache(maxsize=1)
def load_corpus() -> dict[str, dict]:
    path = CORPUS_DIR / "docs.jsonl"
    if not path.exists():
        return {}
    corpus: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            corpus[record["doc_id"]] = record
    return corpus


def select_records(doc_ids: list[str], role: str, max_docs: int | None = None) -> list[dict]:
    corpus = load_corpus()
    data_root = DATA_DIR.resolve()
    selected: list[dict] = []
    for doc_id in doc_ids:
        record = corpus.get(doc_id)
        if not record:
            continue
        if record.get("format") not in TABLE_FORMATS:
            continue
        if record.get("parse_status") != "ok":
            continue
        # 检索层已过滤过权限，这里按语料元数据再校验一次，避免绕过。
        if not can_access(role, record):
            continue
        source = (DATA_DIR / record["path"]).resolve()
        try:
            source.relative_to(data_root)
        except ValueError:
            continue
        if not source.exists():
            continue
        selected.append(record)
        if max_docs is not None and len(selected) >= max_docs:
            break
    return selected


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "strftime"):
        text = value.strftime("%Y-%m-%d") if getattr(value, "hour", 0) == 0 else value.strftime("%Y-%m-%d %H:%M")
        return text
    return str(value).strip()


def _frame_rows(frame: pd.DataFrame) -> list[list[str]]:
    return [[_cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]


def _read_sheets(record: dict) -> list[tuple[str, list[list[str]]]]:
    path = DATA_DIR / record["path"]
    if record["format"] == "csv":
        text = _read_text(path)
        frame = pd.read_csv(
            io.StringIO(text), header=None, dtype=str, keep_default_na=False
        )
        return [("", _frame_rows(frame))]
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    return [(str(title), _frame_rows(frame)) for title, frame in sheets.items()]


def _detect_header_index(rows: list[list[str]]) -> int | None:
    scan = rows[:TABLE_HEADER_SCAN_ROWS]
    counts = [sum(1 for cell in row if cell.strip()) for row in scan]
    if not counts:
        return None
    best = max(range(len(counts)), key=lambda index: (counts[index], -index))
    width = counts[best]
    if width < 2:
        return None
    following = [
        sum(1 for cell in row if cell.strip())
        for row in rows[best + 1 : best + 4]
    ]
    if following and width < 0.6 * max(following):
        return None
    return best


def _trim_width(header: list[str], body: list[list[str]]) -> int:
    width = 0
    for row in [header, *body]:
        for index, cell in enumerate(row):
            if cell.strip():
                width = max(width, index + 1)
    return width


def _clean_columns(header: list[str]) -> list[str]:
    columns: list[str] = []
    used: dict[str, int] = {}
    for index, cell in enumerate(header):
        base = cell.strip() or f"col_{index + 1}"
        used[base] = used.get(base, 0) + 1
        columns.append(base if used[base] == 1 else f"{base}_{used[base]}")
    return columns


def _to_number(text: str) -> float | None:
    value = text.strip()
    if value in BLANK_NUMBERS:
        return None
    negative = value.startswith("(") and value.endswith(")")
    if negative:
        value = value[1:-1]
    value = (
        value.replace("，", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("￥", "")
        .replace("¥", "")
        .replace("元", "")
    )
    if value.endswith("%"):
        value = value[:-1]
    try:
        number = float(value)
    except ValueError:
        return None
    return -number if negative else number


def _column_type_and_unit(name: str, cells: list[str]) -> tuple[str, str]:
    non_blank = [cell for cell in cells if cell.strip()]
    parsed = sum(1 for cell in non_blank if _to_number(cell) is not None)
    column_type = "REAL" if non_blank and parsed / len(non_blank) >= 0.8 else "TEXT"
    if any(cell.strip().endswith("%") for cell in non_blank):
        unit = "%"
    elif "万元" in name:
        unit = "万元"
    elif "元" in name:
        unit = "元"
    else:
        unit = ""
    return column_type, unit


def _convert_rows(rows: list[list[str]], types: list[str]) -> list[tuple]:
    converted: list[tuple] = []
    for row in rows:
        values: list = []
        for index, column_type in enumerate(types):
            cell = row[index] if index < len(row) else ""
            if column_type == "REAL":
                values.append(_to_number(cell))
            else:
                values.append(cell.strip() or None)
        converted.append(tuple(values))
    return converted


def _parse_sheet(rows: list[list[str]]) -> tuple[dict | None, list[str]]:
    non_empty = [row for row in rows if any(cell.strip() for cell in row)]
    if not non_empty:
        return None, []
    header_index = _detect_header_index(non_empty)
    if header_index is None:
        text = "\n".join(" | ".join(cell for cell in row if cell.strip()) for row in non_empty)
        return None, [text[:800]]
    header = non_empty[header_index]
    body = non_empty[header_index + 1 :]
    width = _trim_width(header, body)
    header = [header[index] if index < len(header) else "" for index in range(width)]
    body = [
        [row[index] if index < len(row) else "" for index in range(width)]
        for row in body
    ]
    columns = _clean_columns(header)
    detail, summary = [], []
    for row in body:
        (summary if SUMMARY_RE.match(row[0].strip()) else detail).append(row)
    combined = detail + summary
    types, units = [], []
    for index, name in enumerate(columns):
        column_type, unit = _column_type_and_unit(name, [row[index] for row in combined])
        types.append(column_type)
        units.append(unit)
    notes = [
        " | ".join(cell for cell in row if cell.strip())
        for row in non_empty[:header_index]
    ]
    return {
        "columns": columns,
        "types": types,
        "units": units,
        "detail": _convert_rows(detail, types),
        "summary": _convert_rows(summary, types),
        "notes": [note[:800] for note in notes if note.strip()],
    }, []


def _table_base(doc_id: str) -> str:
    return "t_" + re.sub(r"[^0-9A-Za-z]+", "_", doc_id).strip("_").lower()


def build_loaded_tables(records: list[dict]) -> LoadedTables:
    tables: list[TableSpec] = []
    notes: list[str] = []
    doc_ids: list[str] = []
    for record in records:
        doc_id = record["doc_id"]
        doc_ids.append(doc_id)
        parsed_sheets: list[tuple[str, dict]] = []
        try:
            sheets = _read_sheets(record)
        except Exception as exc:  # 单张表解析失败不影响其他表
            notes.append(f"[{doc_id}] 表格读取失败：{type(exc).__name__}: {exc}")
            continue
        for sheet_title, rows in sheets:
            parsed, sheet_notes = _parse_sheet(rows)
            for note in sheet_notes:
                notes.append(f"[{doc_id}] {sheet_title}: {note}" if sheet_title else f"[{doc_id}] {note}")
            if parsed is None:
                continue
            for note in parsed["notes"]:
                notes.append(
                    f"[{doc_id}] {sheet_title}: {note}" if sheet_title else f"[{doc_id}] {note}"
                )
            parsed_sheets.append((sheet_title, parsed))
        if not parsed_sheets:
            continue
        multi_sheet = len(parsed_sheets) > 1
        for index, (sheet_title, parsed) in enumerate(parsed_sheets, 1):
            base = _table_base(doc_id)
            if multi_sheet:
                base = f"{base}_s{index}"
            if parsed["detail"]:
                tables.append(
                    TableSpec(
                        name=base,
                        doc_id=doc_id,
                        title=record.get("title", ""),
                        sheet=sheet_title,
                        kind="detail",
                        columns=parsed["columns"],
                        types=parsed["types"],
                        units=parsed["units"],
                        rows=parsed["detail"],
                    )
                )
            if parsed["summary"]:
                tables.append(
                    TableSpec(
                        name=f"{base}_summary",
                        doc_id=doc_id,
                        title=record.get("title", ""),
                        sheet=sheet_title,
                        kind="summary",
                        columns=parsed["columns"],
                        types=parsed["types"],
                        units=parsed["units"],
                        rows=parsed["summary"],
                    )
                )
    return LoadedTables(tables=tables, notes=notes, doc_ids=doc_ids)


if __name__ == "__main__":
    sample = ["LG-001", "BG-001", "MS-009", "XS-001", "XS-003"]
    loaded = build_loaded_tables(select_records(sample, role="管理层"))
    for table in loaded.tables:
        print(
            f"{table.name} <- {table.doc_id} [{table.sheet}] {table.kind} "
            f"rows={len(table.rows)} cols={table.columns}"
        )
        print(f"  types={table.types} units={table.units}")
    for note in loaded.notes:
        print(f"NOTE {note}")
    print("普通员工载入 XS-003:", [r["doc_id"] for r in select_records(["XS-003"], "普通员工")])
    print("财务载入 XS-003:", [r["doc_id"] for r in select_records(["XS-003"], "财务")])
