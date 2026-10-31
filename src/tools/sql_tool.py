from __future__ import annotations

import re
import sqlite3
import sys
import time

from src.config import (
    AGG_KEYWORDS,
    CHAT_MODEL,
    TABLE_FORMATS,
    TABLE_MAX_DOCS,
    TABLE_MAX_ROWS,
    TABLE_RESULT_PROMPT_ROWS,
    TABLE_SAMPLE_ROWS,
    TABLE_SQL_MAX_TOKENS,
    TABLE_TOOL_ENABLED,
    get_client,
)
from src.security.policy import can_access
from src.tools.table_loader import LoadedTables, build_loaded_tables, select_records

ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
SYSTEM_TABLES = {"sqlite_master", "sqlite_schema", "sqlite_temp_master"}

SQL_SYSTEM_PROMPT = """你是 SQLite SQL 专家，负责把内部统计问题转成一条查询。
规则：
1. 只输出一条 SELECT 语句，不要解释、不要 markdown。
2. 只能使用给定的表和列；中文列名必须用双引号，例如 "金额(元)"。
3. 以 _summary 结尾的表存放合计/小计等汇总行：明细的 SUM/COUNT/MAX/MIN/排序必须查明细表；问“整体/合计/汇总”时才用 _summary 表。问题含多个小问或同时问整体与排名时，一条查询要返回足够的数据（例如明细排行 + 汇总值），不要只查汇总行。
4. 百分比列存的是百分点（77.5 表示 77.5%）；整体执行率优先用 _summary，或用 累计/预算 计算。
5. 金额列是数值，空值为 NULL；需要保留行明细时不要省略列。
6. 结果行数用 LIMIT 50 限制。
7. 问“最高/最低/最多/最少”时，查询结果必须包含该行对应的数值（如执行率），不能只返回名称。
8. 问“超过 N 天/大于 N”时，条件要覆盖所有更高档位：例如账龄超过 90 天应同时包含“90天以上”和“180天以上”，不能只等值匹配一档；“—”或空值不计入。"""

RESTATE_SYSTEM_PROMPT = """你负责把数据库查询结果转述成中文回答。
规则：
1. 只能使用查询结果中的数字和文字，原样引用，不得自行计算、四舍五入或补充。
2. 每个关键数字后标注来源 [文档ID]，文档ID见来源列表。
3. 简要说明统计口径（筛选条件、时间范围）。
4. 查询结果为空时回答“未查询到符合条件的记录”，不要猜测。
5. 简洁专业，不要提及 SQL、数据库、查询语句或 “t_” 开头的内部表名。"""


VERSION_MARKERS = (
    "以哪份文件为准", "现行版本", "哪一版", "最新版", "是否废止", "已经废止",
    "原来的", "还在吗", "是否还在",
)


def is_aggregation_question(question: str) -> bool:
    # 版本权威性问题需要结合状态/生效日期/废止声明判断，不能只做表内统计。
    if any(marker in question for marker in VERSION_MARKERS):
        return False
    return any(keyword in question for keyword in AGG_KEYWORDS)


def eligible_table_hits(hits: list[dict], role: str, max_docs: int = TABLE_MAX_DOCS) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        meta = hit["metadata"]
        doc_id = meta.get("doc_id", "")
        if not doc_id or doc_id in seen:
            continue
        if meta.get("format") not in TABLE_FORMATS:
            continue
        # 草稿/过期/被替代的表格不得作为统计口径（如过期的通讯录）
        if meta.get("status") != "active":
            continue
        if not can_access(role, meta):
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        if len(doc_ids) >= max_docs:
            break
    return doc_ids


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_connection(loaded: LoadedTables) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for table in loaded.tables:
        columns = ", ".join(
            f"{_ident(name)} {column_type}"
            for name, column_type in zip(table.columns, table.types)
        )
        conn.execute(f"CREATE TABLE {_ident(table.name)} ({columns})")
        placeholders = ", ".join("?" for _ in table.columns)
        conn.executemany(f"INSERT INTO {_ident(table.name)} VALUES ({placeholders})", table.rows)
    conn.commit()
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_authorizer)
    return conn


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action not in ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and str(arg1).lower() in SYSTEM_TABLES:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def sanitize_sql(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    if "```" in text:
        raise ValueError("SQL 中包含代码围栏")
    if "--" in text or "/*" in text:
        raise ValueError("SQL 中不允许注释")
    text = text.rstrip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    if ";" in text:
        raise ValueError("只允许单条语句")
    if not re.match(r"(?is)^(select|with)\s", text):
        raise ValueError("只允许 SELECT 查询")
    return text


def run_select(conn: sqlite3.Connection, sql: str, max_rows: int = TABLE_MAX_ROWS) -> dict:
    deadline = time.time() + 5.0
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        cursor = conn.execute(sql)
        columns = [description[0] for description in cursor.description or []]
        fetched = cursor.fetchmany(max_rows + 1)
    finally:
        conn.set_progress_handler(None, 0)
    truncated = len(fetched) > max_rows
    rows = [list(row) for row in fetched[:max_rows]]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


def _schema_text(loaded: LoadedTables) -> str:
    blocks: list[str] = []
    for table in loaded.tables:
        columns = ", ".join(
            f"{_ident(name)} {column_type}" + (f"（单位：{unit}）" if unit else "")
            for name, column_type, unit in zip(table.columns, table.types, table.units)
        )
        samples = "\n".join(
            " | ".join("NULL" if value is None else str(value) for value in row)
            for row in table.rows[:TABLE_SAMPLE_ROWS]
        )
        kind = "明细表" if table.kind == "detail" else "汇总表"
        blocks.append(
            f"表 {table.name}（doc_id={table.doc_id}，标题={table.title}，"
            f"sheet={table.sheet or '-'}，{kind}，{len(table.rows)} 行）\n"
            f"列：{columns}\n样例：\n{samples}"
        )
    return "\n\n".join(blocks)


def generate_sql(question: str, loaded: LoadedTables, conn: sqlite3.Connection) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"可用表结构：\n{_schema_text(loaded)}\n\n问题：{question}",
        },
    ]
    last_error: Exception | None = None
    for attempt in range(2):
        response = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=TABLE_SQL_MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()
        try:
            sql = sanitize_sql(raw)
            return sql, run_select(conn, sql)
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": f"上面的 SQL 无法执行：{exc}\n请修正后只输出一条 SELECT 语句。",
                    }
                )
    raise RuntimeError(f"SQL 生成失败：{last_error}")


def _tsv(columns: list[str], rows: list[list]) -> str:
    lines = ["\t".join(columns)]
    for row in rows[:TABLE_RESULT_PROMPT_ROWS]:
        lines.append("\t".join("NULL" if value is None else str(value) for value in row))
    return "\n".join(lines)


def _legend(loaded: LoadedTables) -> str:
    return "\n".join(
        f"{table.name} → [{table.doc_id}] {table.title}"
        + (f"（sheet: {table.sheet}）" if table.sheet else "")
        for table in loaded.tables
    )


def compose_answer(
    question: str,
    role: str,
    loaded: LoadedTables,
    sql: str,
    result: dict,
) -> str:
    notes = "\n".join(loaded.notes[:6]) or "（无）"
    if result["rows"]:
        result_text = _tsv(result["columns"], result["rows"])
    else:
        result_text = "（空结果）"
    prompt = (
        f"用户角色：{role}\n"
        f"问题：{question}\n\n"
        f"表与来源：\n{_legend(loaded)}\n\n"
        f"表外说明：\n{notes}\n\n"
        f"查询语句（仅用于理解统计口径，不要在回答中提及）：\n{sql}\n\n"
        f"查询结果（共 {result['row_count']} 行"
        f"{'，已截断' if result['truncated'] else ''}）：\n{result_text}"
    )
    response = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": RESTATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def _used_doc_ids(sql: str, answer: str, loaded: LoadedTables) -> list[str]:
    referenced = [
        table.doc_id for table in loaded.tables if table.name.lower() in sql.lower()
    ]
    for doc_id in loaded.doc_ids:
        if f"[{doc_id}]" in answer:
            referenced.append(doc_id)
    ordered = list(dict.fromkeys(referenced))
    return ordered or list(dict.fromkeys(table.doc_id for table in loaded.tables))


def answer_with_tables(question: str, role: str, hits: list[dict]) -> dict | None:
    if not TABLE_TOOL_ENABLED or not is_aggregation_question(question):
        return None
    doc_ids = eligible_table_hits(hits, role)
    if not doc_ids:
        return None
    try:
        records = select_records(doc_ids, role, max_docs=TABLE_MAX_DOCS)
        if not records:
            return None
        loaded = build_loaded_tables(records)
        if not loaded.tables:
            return None
        conn = build_connection(loaded)
        try:
            sql, result = generate_sql(question, loaded, conn)
        finally:
            conn.close()
        answer = compose_answer(question, role, loaded, sql, result)
    except Exception as exc:
        print(f"[table_tool] 回退到向量问答：{type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    used_doc_ids = _used_doc_ids(sql, answer, loaded)
    return {
        "answer": answer,
        "route": "sql",
        "sql": sql,
        "sql_result": result,
        "table_doc_ids": used_doc_ids,
    }
