from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

TEXT_TABLE_FORMATS = {"pdf", "png", "jpg", "jpeg", "md", "txt"}

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
_SUMMARY_ROW_RE = re.compile(r"^(?:合计|总计|小计|汇总|总额|共计|总成本|总费用)$")
_SUM_DEMAND = ("合计", "总计", "总共", "一共", "汇总", "小计", "共计")
_MEASURE_UNIT_RE = re.compile(r"[（(]\s*([^（）()]{1,8}?)\s*[）)]\s*$")
_UNIT_HINTS = (
    ("金额", "元"), ("费用", "元"), ("成本", "元"), ("收入", "元"),
    ("预算", "元"), ("税额", "元"), ("单价", "元"), ("价格", "元"),
)


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


def _next_nonblank(lines: list[str], start: int) -> int:
    while start < len(lines) and not lines[start].strip():
        start += 1
    return start


def _is_separator(line: str) -> bool:
    return "|" in line and bool(_MD_SEPARATOR.match(line.strip()))


def _is_md_header(lines: list[str], index: int) -> bool:
    if "|" not in lines[index] or _is_separator(lines[index]):
        return False
    following = _next_nonblank(lines, index + 1)
    return following < len(lines) and _is_separator(lines[following])


def parse_text_tables(doc_id: str, title: str, text: str) -> list[TextTable]:
    """从 OCR/PDF/md 文本中提取 Markdown 表格与续页 TSV 表格。

    md 文档常在表头、分隔行与数据行之间夹空行，表头也未必以竖线开头，
    因此按“表头 + 分隔行（跳过空行）”识别表格，数据行允许跨空行。
    """
    lines = text.splitlines()
    tables: list[TextTable] = []
    header: list[str] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_md_header(lines, index):
            separator = _next_nonblank(lines, index + 1)
            header = [_cell(cell) for cell in _split_md_row(line)]
            index = separator + 1
            rows = []
            while index < len(lines):
                current = lines[index]
                if not current.strip():
                    following = _next_nonblank(lines, index + 1)
                    if (
                        following < len(lines)
                        and "|" in lines[following]
                        and not _is_separator(lines[following])
                        and not _is_md_header(lines, following)
                    ):
                        index = following
                        continue
                    break
                if "|" not in current or _is_separator(current):
                    break
                if _is_md_header(lines, index):
                    break
                normalized = _normalize(_split_md_row(current), header)
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


def _measure_name(measure: str) -> str:
    """去掉表头里的单位说明：本月(元) → 本月。"""
    return re.sub(r"[（(][^（）()]*[）)]", "", measure).strip() or measure


def _measure_unit(measure: str) -> str | None:
    match = _MEASURE_UNIT_RE.search(measure)
    if match:
        return match.group(1)
    for keyword, unit in _UNIT_HINTS:
        if keyword in measure:
            return unit
    return None


def _row_group(value: str) -> str | None:
    """实体行（一号工厂）优先取实体名；普通表格退化为行标签，合计行返回 None。"""
    match = _ENTITY_RE.search(value)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    label = _cell(value)
    if not label or label in _BLANK or _SUMMARY_ROW_RE.match(label):
        return None
    if len(label) > 20 or re.search(r"[。；;，,%]", label) or label[0] in "+-":
        return None
    if _NUMBER.match(label.replace(",", "")):
        return None
    return label


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


def _numeric_candidates(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if (ratio := _numeric_ratio(frame, column))[0] >= 0.8 and ratio[1] >= 2
    ]


def _pick_measure(frame: pd.DataFrame, question: str) -> str | None:
    candidates = _numeric_candidates(frame)
    if not candidates:
        return None
    named = [
        column
        for column in candidates
        if column
        and (_measure_name(column) in question or column in question)
    ]
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
    """实体分组列优先（一号工厂）；没有实体时退回普通行标签（如成本项目）。"""
    entity_scores: list[tuple[int, str]] = []
    label_scores: list[tuple[int, str]] = []
    for column in frame.columns:
        if column == measure:
            continue
        entity_rows = 0
        groups: set[str] = set()
        for value in frame[column].tolist():
            if _ENTITY_RE.search(str(value)):
                entity_rows += 1
            label = _row_group(str(value))
            if label:
                groups.add(label)
        if entity_rows >= 2:
            entity_scores.append((entity_rows, column))
        if len(groups) >= 2:
            label_scores.append((len(groups), column))
    if entity_scores:
        return max(entity_scores)[1]
    return max(label_scores)[1] if label_scores else None


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
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


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
    # 同一文档可能含多张不同表（如发票的购买方表 + 明细表），
    # 按列结构分组，只合并同构的续页表，避免空列把数值比例拉低。
    grouped: dict[tuple[str, ...], list[TextTable]] = {}
    for table in tables:
        grouped.setdefault(tuple(table.columns), []).append(table)
    for group_tables in grouped.values():
        calculation = _compute_tables(group_tables, text, question)
        if calculation:
            return {"doc_id": doc_id, "title": title, **calculation}
    return None


def _compute_tables(
    tables: list[TextTable], text: str, question: str
) -> dict | None:
    frame = pd.concat(
        [pd.DataFrame(table.rows, columns=table.columns) for table in tables],
        ignore_index=True,
    )
    measure = _pick_measure(frame, question)
    if not measure:
        return None
    named_measures = [
        column
        for column in _numeric_candidates(frame)
        if _measure_name(column) in question
    ]
    group_column = _pick_group_column(frame, measure)
    if not group_column:
        return None

    frame = frame.copy()
    labels = frame[group_column].map(lambda value: _cell(str(value)))
    summary_mask = labels.map(lambda value: bool(_SUMMARY_ROW_RE.match(value)))
    frame["__group"] = labels.map(_row_group)
    frame["__value"] = pd.to_numeric(
        frame[measure].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    declared_rows = frame.loc[summary_mask, "__value"].dropna()
    declared_row_total = float(declared_rows.sum()) if len(declared_rows) else None

    detail = frame.dropna(subset=["__group", "__value"])
    if detail.empty:
        return None

    requested = _question_groups(question)
    available = list(dict.fromkeys(detail["__group"].tolist()))
    if not requested:
        requested = [group for group in available if group in question]
    if requested:
        if any(group not in available for group in requested):
            return None
        groups = requested
        selected = detail[detail["__group"].isin(groups)]
    else:
        groups = available
        selected = detail

    suffix = _group_suffix(available)
    if suffix is None:
        # 普通 md 表格没有“N号工厂”这类实体行，容易把多字段题误算成一次汇总：
        # 只处理“题目明确点名唯一数值列 + 表内存在合计行 + 明确问合计”的题。
        if (
            declared_row_total is None
            or named_measures != [measure]
            or not any(word in question for word in _SUM_DEMAND)
        ):
            return None

    sums = selected.groupby("__group", sort=False)["__value"].sum()
    ordered = [
        {"name": group, "value": float(sums[group])}
        for group in groups
        if group in sums
    ]
    if not ordered:
        return None
    total = float(sum(item["value"] for item in ordered))
    declared = _declared_total(text, _measure_name(measure))
    declared_total = (
        declared_row_total if declared_row_total is not None else (declared[0] if declared else None)
    )
    unit = (
        _measure_unit(measure)
        or (declared[1] if declared else None)
        or ("个" if suffix else "")
    )
    return {
        "measure": measure,
        "measure_name": _measure_name(measure),
        "unit": unit,
        "groups": ordered,
        "total": total,
        "declared_total": declared_total,
        "group_suffix": suffix,
    }


def compose_scanned_answer(calculation: dict) -> str:
    measure = calculation["measure"]
    unit = calculation["unit"]
    groups = calculation["groups"]
    total = calculation["total"]
    suffix = calculation.get("group_suffix")
    if suffix:
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
    else:
        unit_suffix = f" {unit}" if unit else ""
        group_text = "；".join(
            f"{item['name']} {_number(item['value'])}{unit_suffix}" for item in groups
        )
        total_label = (
            f"{_cn_count(len(groups))}项"
            if len(groups) <= 10
            else "上述各项"
        )
        answer = (
            f"按明细行汇总：{group_text}。"
            f"{total_label}合计 {_number(total)}{unit_suffix}。"
        )
    declared = calculation.get("declared_total")
    if declared is not None:
        declared_unit = f" {unit}" if unit else ""
        answer += f"表尾统计行写 {_number(declared)}{declared_unit}"
        if abs(declared - total) > 1e-9:
            answer += (
                f"，与明细逐行合计 {_number(total)}{declared_unit}不一致，"
                "应以明细逐行为准"
            )
        answer += "。"
    return f"{answer} [{calculation['doc_id']}]"
