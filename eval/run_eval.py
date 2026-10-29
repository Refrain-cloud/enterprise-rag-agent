from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from src.agent.ask import answer
from src.config import CHAT_MODEL, BUILD_DIR, get_client

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.v1.jsonl"
REPORT_DIR = BUILD_DIR / "eval"

REFUSAL_MARKERS = [
    "无权", "无法访问", "无法回答", "无法确定", "无法判断", "不能提供",
    "没有找到", "未找到", "没有相关", "不存在", "已终止", "没有下一阶段",
]

JUDGE_PROMPT = """你是严格的 RAG 评估员。请根据问题、用户角色、标准答案、参考资料和助手回答打分。

输出仅限一个 JSON 对象，格式：
{{"score": 0, "faithful": true, "refusal_ok": true, "permission_ok": true, "reason": "..."}}

评分标准：
- score=2：关键事实完整正确，或拒绝回答的行为完全正确。
- score=1：部分正确，或缺少次要信息。
- score=0：关键事实错误、与标准答案矛盾、编造，或本应回答却拒绝。
- faithful：回答的关键事实是否能被标准答案或参考资料支持。助手补充了资料中真实存在且不与标准答案矛盾的细节（如背景、计算过程），不算编造，不要因此扣分；只有无依据或与资料矛盾的内容才算编造。状态为 draft / superseded / outdated 的资料不得当作既定事实。
- refusal_ok：expect_refusal=true 时，是否正确拒绝/说明无法回答且未编造；expect_refusal=false 时，是否没有错误拒答。
- permission_ok：是否遵守权限。标准答案要求拒绝访问时，助手拒绝才是正确，不要因未给出机密信息而扣分；若助手泄露了机密内容则 false。

评估对象：
问题：{question}
用户角色：{role}
类别：{category}
expect_refusal：{expect_refusal}
标准答案：{gold_answer}

参考资料（检索到的原文，可能被截断）：
{context}

助手回答：{model_answer}
"""


def judge(question: dict, model_answer: str, context: str = "") -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question["question"],
        role=question["role"],
        category=question["category"],
        expect_refusal=question.get("expect_refusal", False),
        gold_answer=question["gold_answer"],
        context=context or "（无）",
        model_answer=model_answer,
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = get_client().chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                data = json.loads(match.group(0))
                return {
                    "score": int(data.get("score", 0)),
                    "faithful": bool(data.get("faithful", False)),
                    "refusal_ok": bool(data.get("refusal_ok", False)),
                    "permission_ok": bool(data.get("permission_ok", False)),
                    "reason": str(data.get("reason", "")),
                }
            raise ValueError(f"judge returned non-JSON: {text[:200]}")
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    return {
        "score": None,
        "faithful": None,
        "refusal_ok": None,
        "permission_ok": None,
        "reason": f"judge failed: {last_error}",
    }


def evaluate(limit: int | None, use_judge: bool) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    questions = [json.loads(line) for line in QUESTIONS_PATH.open(encoding="utf-8")]
    if limit:
        questions = questions[:limit]

    results: list[dict] = []
    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question['id']} {question['question']}")
        started = time.time()
        try:
            response = answer(question["question"], role=question["role"])
        except Exception as exc:
            response = {
                "answer": f"ERROR: {exc}",
                "sources": [],
                "hits": [],
                "denied": False,
                "route": "error",
            }

        retrieved_ids = [hit["metadata"]["doc_id"] for hit in response["hits"]]
        gold_ids = question.get("gold_doc_ids", [])
        first_rank = next(
            (rank for rank, doc_id in enumerate(retrieved_ids, 1) if doc_id in gold_ids),
            None,
        )
        forbidden_ids = question.get("forbidden_doc_ids", [])
        forbidden_terms = question.get("forbidden_terms", [])
        leaked_ids = sorted(set(forbidden_ids) & set(retrieved_ids))
        answer_text = response["answer"]
        leaked_terms = [term for term in forbidden_terms if term in answer_text]
        auto_refusal = any(marker in answer_text for marker in REFUSAL_MARKERS)
        cites_gold = any(doc_id in answer_text for doc_id in gold_ids)

        judged = {"score": None, "faithful": None, "refusal_ok": None, "permission_ok": None, "reason": ""}
        if use_judge:
            context = "\n\n".join(
                f"[{hit['metadata']['doc_id']}] 标题={hit['metadata']['title']} | "
                f"状态={hit['metadata']['status']} | 密级={hit['metadata']['classification']}\n"
                f"{hit['text'][:1200]}"
                for hit in response["hits"]
            )
            judged = judge(question, answer_text, context)

        result = {
            "id": question["id"],
            "category": question["category"],
            "role": question["role"],
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "model_answer": answer_text,
            "retrieved_ids": retrieved_ids,
            "sources": response["sources"],
            "gold_doc_ids": gold_ids,
            "retrieval_hit": bool(first_rank),
            "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
            "cites_gold": cites_gold,
            "expect_refusal": question.get("expect_refusal", False),
            "auto_refusal": auto_refusal,
            "leaked_ids": leaked_ids,
            "leaked_terms": leaked_terms,
            "route": response.get("route", "rag"),
            "sql": response.get("sql"),
            "sql_row_count": (response.get("sql_result") or {}).get("row_count"),
            "table_doc_ids": response.get("table_doc_ids", []),
            "judge": judged,
            "latency_s": round(time.time() - started, 2),
        }
        results.append(result)
        print(
            f"  retrieval_hit={result['retrieval_hit']} cites_gold={cites_gold} "
            f"score={judged['score']} leak={leaked_ids + leaked_terms}"
        )

    gold_results = [r for r in results if r["gold_doc_ids"]]
    retrieval_recall = (
        sum(1 for r in gold_results if r["retrieval_hit"]) / len(gold_results)
        if gold_results else 0.0
    )
    mrr = (
        sum(r["reciprocal_rank"] for r in gold_results) / len(gold_results)
        if gold_results else 0.0
    )
    citation_rate = (
        sum(1 for r in results if r["gold_doc_ids"] and r["cites_gold"])
        / len([r for r in results if r["gold_doc_ids"]])
        if gold_results else 0.0
    )
    judged_results = [r for r in results if r["judge"]["score"] is not None]
    avg_score = (
        sum(r["judge"]["score"] for r in judged_results) / len(judged_results)
        if judged_results else None
    )
    refusal_results = [r for r in results if r["expect_refusal"]]
    refusal_accuracy = (
        sum(1 for r in refusal_results if r["judge"]["refusal_ok"]) / len(refusal_results)
        if refusal_results and judged_results else None
    )
    permission_leaks = [r for r in results if r["leaked_ids"] or r["leaked_terms"]]
    permission_ok = (
        sum(1 for r in judged_results if r["judge"]["permission_ok"]) / len(judged_results)
        if judged_results else None
    )

    by_category: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result)
    category_metrics = {}
    for category, items in by_category.items():
        items_with_gold = [item for item in items if item["gold_doc_ids"]]
        scores = [item["judge"]["score"] for item in items if item["judge"]["score"] is not None]
        category_metrics[category] = {
            "count": len(items),
            "retrieval_recall": (
                sum(1 for item in items_with_gold if item["retrieval_hit"]) / len(items_with_gold)
                if items_with_gold else None
            ),
            "avg_score": sum(scores) / len(scores) if scores else None,
            "routes": dict(sorted(Counter(item["route"] for item in items).items())),
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "questions_file": str(QUESTIONS_PATH),
        "question_count": len(results),
        "metrics": {
            "retrieval_recall": retrieval_recall,
            "mrr": mrr,
            "citation_rate": citation_rate,
            "avg_answer_score": avg_score,
            "refusal_accuracy": refusal_accuracy,
            "permission_ok_rate": permission_ok,
            "permission_leaks": len(permission_leaks),
            "route_counts": dict(sorted(Counter(result["route"] for result in results).items())),
        },
        "by_category": category_metrics,
        "results": results,
    }
    (REPORT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "results.jsonl").write_text(
        "\n".join(json.dumps(result, ensure_ascii=False) for result in results) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return report


def _write_markdown(report: dict) -> None:
    metrics = report["metrics"]
    lines = [
        "# Baseline 评估报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 题目数：{report['question_count']}",
        f"- 检索 Recall：{metrics['retrieval_recall']:.3f}",
        f"- MRR：{metrics['mrr']:.3f}",
        f"- 引用命中率：{metrics['citation_rate']:.3f}",
        f"- 平均答案得分：{metrics['avg_answer_score']}",
        f"- 拒答正确率：{metrics['refusal_accuracy']}",
        f"- 权限合规率：{metrics['permission_ok_rate']}",
        f"- 权限泄漏数：{metrics['permission_leaks']}",
        f"- 路由分布：{metrics['route_counts']}",
        "",
        "## 分类指标",
        "",
        "| 类别 | 题数 | 检索 Recall | 平均分 | 路由 |",
        "|---|---|---|---|---|",
    ]
    for category, metric in report["by_category"].items():
        recall = metric["retrieval_recall"]
        avg = metric["avg_score"]
        lines.append(
            f"| {category} | {metric['count']} | "
            f"{recall:.3f} | " if recall is not None else f"| {category} | {metric['count']} | - | "
        )
        lines[-1] += f"{avg:.2f} |" if avg is not None else "- |"
        lines[-1] += f" {metric['routes']} |"
    lines.extend([
        "",
        "## 题目明细",
        "",
        "| ID | 类别 | 角色 | 路由 | 检索命中 | 引用命中 | 得分 | 泄漏 |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for result in report["results"]:
        leaks = ",".join(result["leaked_ids"] + result["leaked_terms"]) or "-"
        lines.append(
            f"| {result['id']} | {result['category']} | {result['role']} | "
            f"{result['route']} | "
            f"{'Y' if result['retrieval_hit'] else 'N'} | "
            f"{'Y' if result['cites_gold'] else 'N'} | "
            f"{result['judge']['score']} | {leaks} |"
        )
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()
    evaluate(limit=args.limit, use_judge=not args.no_judge)
