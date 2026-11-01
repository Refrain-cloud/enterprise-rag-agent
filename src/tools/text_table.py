from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

TEXT_TABLE_FORMATS = {"pdf", "png", "jpg", "jpeg"}

_MD_SEPARATOR = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
_BLANK = {"", "—", "-", "/", "n/a", "N/A", "暂无"}

_ENTITY_SUFFIXES = ("项目部", "工厂", "车间", "部门", "门店", "中心", "产线")
_ENTITY_RE = re.compile(
    r"([一二三四五六七八九十百零两\d]+号)(项目部|工厂|车间|部门|门店|中心|产线)"
)
_ENUM_ENTITY_RE = re.compile(
    r"((?:[一二三四五六七八九十百零两\d]+号[、，,]\s*)+"
    r"[一二三四五六七八九十百零两\d]+号)(项目部|工厂|车间|部门|门店|中心|产线)"
)
_UNIT_RE = r"([个台件套人条项]?)"


@dataclass
class TextTable:
    doc_id: str
    title: str
    columns: list[str]
    rows: list[dict[str, str]]


def _cell(value: str) -> str:
    return value.strip().strip("*").strip()


def _split_md_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_cell(part) for part in stripped.split("|")]


def _normalize(row: list[str], header: list[str]) -> dict[str, str] | None:
    if not row or len(row) < 2:
        return None
    values = row[: len(header)] + [""] * max(0, len(header) - len(row))
    return dict(zip(header, values))


def parse_text_tables(doc_id: str, title: str, text: str) -> list[TextTable]:
    """从 OCR/PDF 文本中提取 Markdown 表格与续页 TSV 表格。"""
    lines = text.splitlines()
    tables: list[TextTable] = []
    header: list[str] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            line.strip().startswith("|")
            and index + 1 < len(lines)
            and _MD_SEPARATOR.match(lines[index + 1])
        ):
            header = [_cell(cell) for cell in _split_md_row(line)]
            index += 2
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                normalized = _normalize(_split_md_row(lines[index]), header)
                if normalized:
                    rows.append(normalized)
                index += 1
            if rows:
                tables.append(TextTable(doc_id, title, header, rows))
            continue
        if line.count("\t") >= 2:
            raw_rows = []
            while index < len(lines) and lines[index].count("\t") >= 2:
                raw_rows.append([_cell(cell) for cell in lines[index].split("\t")])
                index += 1
            columns = header or [
                f"col_{position}" for position in range(1, len(raw_rows[0]) + 1)
            ]
            rows = [
                row for row in (_normalize(raw, columns) for raw in raw_rows) if row
            ]
            if rows:
                tables.append(TextTable(doc_id, title, columns, rows))
            continue
        index += 1
    return tables


def _question_groups(question: str) -> list[str]:
    groups: list[str] = []
    for match in _ENUM_ENTITY_RE.finditer(question):
        for item in re.split(r"[、，,]", match.group(1)):
            if item.strip():
                groups.append(f"{item.strip()}{match.group(2)}")
    for match in _ENTITY_RE.finditer(question):
        groups.append(f"{match.group(1)}{match.group(2)}")
    return list(dict.fromkeys(groups))


def _row_group(value: str) -> str | None:
    match = _ENTITY_RE.search(value)
    return f"{match.group(1)}{match.group(2)}" if match else None


def _numeric_ratio(frame: pd.DataFrame, column: str) -> tuple[float, int]:
    values = [
        str(value).strip()
        for value in frame[column].tolist()
        if str(value).strip() not in _BLANK
    ]
    if not values:
        return 0.0, 0
    parsed = sum(1 for value in values if _NUMBER.match(value.replace(",", "")))
    return parsed / len(values), len(values)


def _pick_measure(frame: pd.DataFrame, question: str) -> str | None:
    candidates = [
        column
        for column in frame.columns
        if (ratio := _numeric_ratio(frame, column))[0] >= 0.8 and ratio[1] >= 2
    ]
    if not candidates:
        return None
    named = [column for column in candidates if column and column in question]
    if named:
        return max(named, key=len)

    def overlap(column: str) -> int:
        best = 0
        for start in range(len(column)):
            for end in range(start + 2, len(column) + 1):
                if column[start:end] in question:
                    best = max(best, end - start)
        return best

    scored = sorted(((overlap(column), column) for column in candidates), reverse=True)
    if scored and scored[0][0] >= 2 and (
        len(scored) == 1 or scored[0][0] > scored[1][0]
    ):
        return scored[0][1]
    return candidates[0] if len(candidates) == 1 else None


def _pick_group_column(frame: pd.DataFrame, measure: str) -> str | None:
    scores: list[tuple[int, str]] = []
    for column in frame.columns:
        if column == measure:
            continue
        count = sum(1 for value in frame[column].tolist() if _row_group(str(value)))
        if count:
            scores.append((count, column))
    return max(scores)[1] if scores else None


def _declared_total(text: str, measure: str) -> tuple[float, str] | None:
    patterns = (
        re.escape(measure) + r"[^。\n]{0,12}?(\d+(?:\.\d+)?)\s*" + _UNIT_RE,
        r"(?:合计|统计|总计)[^。\n]{0,15}?(\d+(?:\.\d+)?)\s*" + _UNIT_RE,
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), match.group(2) or "个"
    return None


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _group_suffix(groups: list[str]) -> str | None:
    for suffix in _ENTITY_SUFFIXES:
        if groups and all(group.endswith(suffix) for group in groups):
            return suffix
    return None


def _cn_count(number: int) -> str:
    digits = "零一二三四五六七八九十"
    return digits[number] if 0 <= number <= 10 else str(number)


def compute_scanned_aggregate(
    doc_id: str, title: str, text: str, question: str
) -> dict | None:
    tables = parse_text_tables(doc_id, title, text)
    if not tables:
        return None
    frame = pd.concat(
        [pd.DataFrame(table.rows, columns=table.columns) for table in tables],
        ignore_index=True,
    )
    measure = _pick_measure(frame, question)
    if not measure:
        return None
    group_column = _pick_group_column(frame, measure)
    if not group_column:
        return None

    frame = frame.copy()
    frame["__group"] = frame[group_column].map(lambda value: _row_group(str(value)))
    frame["__value"] = pd.to_numeric(
        frame[measure].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    frame = frame.dropna(subset=["__group", "__value"])
    if frame.empty:
        return None

    requested = _question_groups(question)
    available = list(dict.fromkeys(frame["__group"].tolist()))
    if requested:
        if any(group not in available for group in requested):
            return None
        groups = requested
        selected = frame[frame["__group"].isin(groups)]
    else:
        groups = available
        selected = frame

    sums = selected.groupby("__group", sort=False)["__value"].sum()
    ordered = [
        {"name": group, "value": float(sums[group])}
        for group in groups
        if group in sums
    ]
    if not ordered:
        return None
    total = float(sum(item["value"] for item in ordered))
    declared = _declared_total(text, measure)
    unit = declared[1] if declared else "个"
    return {
        "doc_id": doc_id,
        "title": title,
        "measure": measure,
        "unit": unit,
        "groups": ordered,
        "total": total,
        "declared_total": declared[0] if declared else None,
        "group_suffix": _group_suffix(groups),
    }


def compose_scanned_answer(calculation: dict) -> str:
    measure = calculation["measure"]
    unit = calculation["unit"]
    groups = calculation["groups"]
    total = calculation["total"]
    suffix = calculation.get("group_suffix") or "项"
    group_text = "；".join(
        f"{item['name']} {_number(item['value'])} {unit}{measure}" for item in groups
    )
    total_label = (
        f"{_cn_count(len(groups))}个{suffix}"
        if len(groups) <= 10
        else "上述项目"
    )
    answer = (
        f"按明细行汇总：{group_text}。"
        f"{total_label}合计 {_number(total)} {unit}{measure}。"
    )
    declared = calculation.get("declared_total")
    if declared is not None:
        answer += f"表尾统计行写 {_number(declared)} {unit}"
        if abs(declared - total) > 1e-9:
            answer += (
                f"，与明细逐行合计 {_number(total)} {unit}不一致，"
                "应以明细逐行为准"
            )
        answer += "。"
    return f"{answer} [{calculation['doc_id']}]"
