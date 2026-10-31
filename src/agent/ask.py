from __future__ import annotations

import argparse
from datetime import date

from src.config import AGG_TOP_K, CHAT_MODEL, TOP_K, ROLES, TABLE_TOOL_ENABLED, get_client
from src.retrieval.search import hybrid_search
from src.security.policy import can_access, normalize_role
from src.tools.sql_tool import answer_with_tables, eligible_table_hits, is_aggregation_question

SYSTEM_PROMPT = """你是北辰智造科技的内部知识库助手。请严格依据提供的资料回答。

规则：
1. 只使用资料中的信息，不得编造；资料不足时明确说明无法回答。
2. 每个关键事实后用 [文档ID] 标注来源，例如 [FN-007]；没有来源的结论不要输出。
3. 同一问题存在多个版本时，优先采用状态为 active 且生效日期最新的正式文件。草稿（draft）、已废止（superseded/outdated）、批注稿不得作为既定事实；草稿只能说明“该文件为草稿、未对外承诺”，不能把草稿中的方案或排期当作已确认计划。资料矛盾时要说明以哪份文件为准及原因。
4. 表格数据可以计算，但要说明统计口径（例如仅统计“已付款”）；逐行核对，确保笔数与金额一致，不要凭印象计数。
5. 如果资料显示某个信息属于机密且当前角色无权查看，直接说明无权访问，不得透露内容。
6. 今天的日期是 {today}。判断文件是否已生效、是否为现行版本时，一律以该日期为准，不要假设当前年份。
7. 回答使用简洁、专业的中文。"""


def _system_prompt() -> str:
    return SYSTEM_PROMPT.format(today=date.today().isoformat())


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for index, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        blocks.append(
            f"[资料 {index}] doc_id={meta['doc_id']} | 标题={meta['title']} | "
            f"版本={meta['version']} | 状态={meta['status']} | "
            f"生效日期={meta['effective_date']} | 密级={meta['classification']} | "
            f"路径={meta['path']}\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def _sources(hits: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for hit in hits:
        meta = hit["metadata"]
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "title": meta["title"],
                "version": meta["version"],
                "status": meta["status"],
                "effective_date": meta["effective_date"],
                "path": meta["path"],
                "classification": meta["classification"],
            }
    return list(seen.values())


def answer(question: str, role: str = "普通员工", top_k: int = TOP_K) -> dict:
    role = normalize_role(role)
    hits = hybrid_search(question, role=role, top_k=top_k)
    # 聚合题的表格文档可能被叙述型文档挤出 top-K：没有任何可访问表格时再放宽候选
    if (
        TABLE_TOOL_ENABLED
        and is_aggregation_question(question)
        and not eligible_table_hits(hits, role)
    ):
        hits = hybrid_search(question, role=role, top_k=max(top_k, AGG_TOP_K))
    best_overall = hybrid_search(question, role=role, top_k=1, ignore_permissions=True)

    def _denied() -> dict:
        return {
            "answer": "该信息属于公司机密，当前角色无权访问。",
            "sources": [],
            "hits": [],
            "denied": True,
            "route": "rag",
        }

    if not hits:
        if best_overall and not can_access(role, best_overall[0]["metadata"]):
            return _denied()
        return {
            "answer": "知识库中没有找到与该问题相关且你有权访问的资料，无法回答。",
            "sources": [],
            "hits": [],
            "denied": False,
            "route": "rag",
        }

    # 最优匹配是外部门机密时直接拒答，避免模型用关联的公开资料拼凑敏感信息。
    if best_overall and not can_access(role, best_overall[0]["metadata"]):
        if str(best_overall[0]["metadata"].get("scope", "")).startswith("dept:"):
            return _denied()

    if TABLE_TOOL_ENABLED:
        tool_result = answer_with_tables(question, role, hits)
        if tool_result:
            used_doc_ids = set(tool_result.get("table_doc_ids", []))
            tool_result["hits"] = hits
            tool_result["sources"] = _sources(
                [hit for hit in hits if hit["metadata"]["doc_id"] in used_doc_ids]
            )
            tool_result["denied"] = False
            return tool_result

    context = _format_context(hits)
    messages = [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": f"当前用户角色：{role}\n\n参考资料：\n{context}\n\n问题：{question}",
        },
    ]
    response = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=900,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": _sources(hits),
        "hits": hits,
        "denied": False,
        "route": "rag",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="北辰智造知识库问答")
    parser.add_argument("question", nargs="+", help="要提问的问题")
    parser.add_argument("--role", default="普通员工", choices=ROLES)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--debug", action="store_true", help="打印路由与 SQL")
    args = parser.parse_args()
    result = answer(" ".join(args.question), role=args.role, top_k=args.top_k)
    print(result["answer"])
    if args.debug:
        print(f"\n[route] {result.get('route', '?')}")
        if result.get("sql"):
            print(f"[sql] {result['sql']}")
        if result.get("sql_result"):
            print(f"[rows] {result['sql_result']['row_count']}")
    if result["sources"]:
        print("\n来源：")
        for source in result["sources"]:
            print(
                f"- [{source['doc_id']}] {source['title']} "
                f"(v{source['version']}, {source['status']}, {source['path']})"
            )


if __name__ == "__main__":
    main()
