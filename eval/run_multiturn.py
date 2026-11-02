from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.run_eval import (
    REFUSAL_MARKERS,
    _aggregate,
    _answer_fingerprint,
    _full_docs_context,
    _generation_stability,
    _judge_excerpt,
    _judge_fingerprint,
    _modal_judgement,
    judge,
)
from src.agent.ask import answer
from src.agent.session import SessionStore
from src.llm import (
    reset_usage,
    start_request_usage,
    summarize_records,
    usage_snapshot,
)
from src.retrieval.search import reset_search_stats, search_stats

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.multiturn.v1.jsonl"
REPORT_DIR = Path(__file__).resolve().parents[1] / "build" / "eval"


def load_groups(path: Path) -> list[dict]:
    groups: list[dict] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        lines = list(enumerate(handle, 1))
    for line_no, line in lines:
        if not line.strip():
            continue
        try:
            group = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc
        group_id = group.get("id")
        if not group_id:
            raise ValueError(f"{path}:{line_no} 缺少 id")
        if group_id in seen:
            raise ValueError(f"{path}:{line_no} id 重复：{group_id}")
        seen.add(group_id)
        if not group.get("role"):
            raise ValueError(f"{group_id} 缺少 role")
        turns = group.get("turns") or []
        if not turns:
            raise ValueError(f"{group_id} 没有 turns")
        for index, turn in enumerate(turns, 1):
            if not turn.get("question") or not turn.get("gold_answer"):
                raise ValueError(f"{group_id} 第 {index} 轮缺少 question/gold_answer")
        groups.append(group)
    return groups


def _followup_rewrite(response: dict) -> dict:
    for event in response.get("trace") or []:
        if event.get("node") == "prepare":
            return (event.get("summary") or {}).get("followup_rewrite") or {}
    return {}


def _allowed_citation_ids(response: dict, retrieved_ids: list[str]) -> set[str]:
    allowed = set(retrieved_ids)
    for source in response.get("sources") or []:
        if source.get("doc_id"):
            allowed.add(source["doc_id"])
    allowed.update(response.get("table_doc_ids") or [])
    return allowed


def run_group(
    group: dict,
    *,
    store: SessionStore,
    answer_fn=answer,
    judge_fn=judge,
    use_judge: bool = True,
    judge_repeats: int = 1,
) -> list[dict]:
    role = group["role"]
    session_id = store.create(role).session_id
    rows: list[dict] = []

    for turn_index, turn in enumerate(group["turns"], 1):
        view = store.begin_turn(session_id, role)
        history = view.history_payload() or None
        question_text = turn["question"]
        print(
            f"  [{group['id']} {turn_index}/{len(group['turns'])}] {question_text}",
            flush=True,
        )

        started = time.time()
        records = start_request_usage()
        reset_search_stats()
        try:
            response = answer_fn(
                question_text,
                role=role,
                history=history,
                session_id=session_id,
                turn_index=view.turn_count,
            )
        except Exception as exc:  # 评估不能因单轮异常中断整组
            response = {
                "answer": f"ERROR: {exc}",
                "sources": [],
                "hits": [],
                "denied": False,
                "route": "error",
                "citations": [],
                "rewritten_question": question_text,
            }
        generation_usage = summarize_records(records)
        retrieval_stats = search_stats()
        answer_text = response.get("answer", "") or ""
        rewritten_question = (
            response.get("rewritten_question") or question_text
        )
        store.finish_turn(
            session_id,
            question=question_text,
            answer=answer_text,
            rewritten_question=rewritten_question,
            usage=generation_usage,
            token=view.turn_token,
        )

        gold_ids = turn.get("gold_doc_ids", [])
        forbidden_ids = turn.get("forbidden_doc_ids", [])
        forbidden_terms = turn.get("forbidden_terms", [])
        forbidden_citations = turn.get("forbidden_citation_doc_ids", [])
        retrieved_ids = [
            hit["metadata"]["doc_id"] for hit in response.get("hits") or []
        ]
        first_rank = next(
            (rank for rank, doc_id in enumerate(retrieved_ids, 1) if doc_id in gold_ids),
            None,
        )
        leaked_ids = sorted(set(forbidden_ids) & set(retrieved_ids))
        leaked_terms = [term for term in forbidden_terms if term in answer_text]
        citations = response.get("citations")
        if citations is None:
            citations = []
        allowed_citations = _allowed_citation_ids(response, retrieved_ids)
        citations_outside = sorted(set(citations) - allowed_citations)
        forbidden_citation_hits = sorted(set(citations) & set(forbidden_citations))
        followup_rewrite = _followup_rewrite(response)

        question = {
            "id": f"{group['id']}_t{turn_index}",
            "category": group.get("category", "多轮追问"),
            "role": role,
            "question": question_text,
            "gold_answer": turn["gold_answer"],
            "gold_doc_ids": gold_ids,
            "key_facts": turn.get("key_facts") or [],
            "expect_refusal": turn.get("expect_refusal", False),
        }
        judged = {
            "score": None,
            "faithful": None,
            "refusal_ok": None,
            "permission_ok": None,
            "reason": "",
            "facts": [],
            "contradictions": [],
            "unsupported_claims": [],
            "refused": None,
        }
        runs: list[dict] = []
        judge_usage: dict = {}
        if use_judge:
            reset_usage()
            leaked = bool(leaked_ids or leaked_terms)
            runs = [
                judge_fn(
                    question,
                    answer_text,
                    context=_judge_excerpt(response.get("hits") or []),
                    full_docs=_full_docs_context(answer_text, gold_ids),
                    leaked=leaked,
                )
                for _ in range(max(1, judge_repeats))
            ]
            judged = _modal_judgement(runs) if len(runs) > 1 else runs[0]
            judge_usage = usage_snapshot()
        latency_s = round(time.time() - started, 2)

        row = {
            "id": question["id"],
            "group_id": group["id"],
            "turn_index": turn_index,
            "history_turns": len(history or []),
            "category": question["category"],
            "role": role,
            "question": question_text,
            "rewritten_question": rewritten_question,
            "followup_rewrite": followup_rewrite,
            "gold_answer": question["gold_answer"],
            "key_facts": question["key_facts"],
            "model_answer": answer_text,
            "retrieved_ids": retrieved_ids,
            "sources": response.get("sources", []),
            "gold_doc_ids": gold_ids,
            "retrieval_hit": bool(first_rank),
            "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
            "cites_gold": any(doc_id in answer_text for doc_id in gold_ids),
            "citations": citations,
            "citations_outside_retrieved": citations_outside,
            "forbidden_citation_hits": forbidden_citation_hits,
            "expect_refusal": question["expect_refusal"],
            "auto_refusal": any(marker in answer_text for marker in REFUSAL_MARKERS),
            "leaked_ids": leaked_ids,
            "leaked_terms": leaked_terms,
            "route": response.get("route", "rag"),
            "sql": response.get("sql"),
            "sql_row_count": (response.get("sql_result") or {}).get("row_count"),
            "table_doc_ids": response.get("table_doc_ids", []),
            "table_calc": response.get("table_calc"),
            "judge": judged,
            "answer_fingerprint": _answer_fingerprint(),
            "judge_fingerprint": _judge_fingerprint(question),
            "latency_s": latency_s,
            "retrieval_calls": retrieval_stats["calls"],
            "retrieval_latency_s": retrieval_stats["elapsed_s"],
            "generation_usage": generation_usage,
            "judge_usage": judge_usage,
            "generation_stability": _generation_stability(
                [response], [judged["score"]]
            ),
            "format_check": response.get("format_check"),
            "generation_check": response.get("generation_check"),
            "self_check": response.get("self_check"),
            "trace": response.get("trace"),
            "trace_meta": response.get("trace_meta"),
            "rewrite": response.get("rewrite", {}),
            "retrieval_queries": (response.get("retrieval") or {}).get("queries"),
        }
        if len(runs) > 1:
            row["judge_repeat_scores"] = [run["score"] for run in runs]
        rows.append(row)

        rewrite = followup_rewrite or {}
        rewrite_note = (
            f" rewrite={'ok' if rewrite.get('succeeded') else 'no'}"
            f"({rewrite.get('query')})"
            if rewrite.get("triggered")
            else ""
        )
        print(
            f"    route={row['route']} hit={row['retrieval_hit']} "
            f"score={judged['score']} citations={citations}"
            f" outside={citations_outside}"
            f" forbidden_cites={forbidden_citation_hits}{rewrite_note}",
            flush=True,
        )
    return rows


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_multiturn(rows: list[dict], groups: list[dict]) -> dict:
    followups = [row for row in rows if row["turn_index"] > 1]
    first_turns = [row for row in rows if row["turn_index"] == 1]
    triggered = [
        row for row in followups if (row.get("followup_rewrite") or {}).get("triggered")
    ]
    succeeded = [
        row for row in triggered if (row.get("followup_rewrite") or {}).get("succeeded")
    ]

    def scores(items: list[dict]) -> list[float]:
        return [
            row["judge"]["score"]
            for row in items
            if row["judge"]["score"] is not None
        ]

    def group_summary(group: dict) -> dict:
        items = [row for row in rows if row["group_id"] == group["id"]]
        group_triggered = [
            row
            for row in items
            if (row.get("followup_rewrite") or {}).get("triggered")
        ]
        group_succeeded = [
            row
            for row in group_triggered
            if (row.get("followup_rewrite") or {}).get("succeeded")
        ]
        return {
            "id": group["id"],
            "role": group["role"],
            "turns": len(items),
            "scores": [row["judge"]["score"] for row in items],
            "avg_score": _avg(scores(items)),
            "rewrite_triggered": len(group_triggered),
            "rewrite_succeeded": len(group_succeeded),
            "citations_outside_retrieved": sum(
                1 for row in items if row["citations_outside_retrieved"]
            ),
            "forbidden_citation_hits": sum(
                1 for row in items if row["forbidden_citation_hits"]
            ),
            "permission_leaks": sum(
                1 for row in items if row["leaked_ids"] or row["leaked_terms"]
            ),
        }

    return {
        "groups": len(groups),
        "turns": len(rows),
        "first_turn_avg_score": _avg(scores(first_turns)),
        "followup_avg_score": _avg(scores(followups)),
        "followup_turns": len(followups),
        "rewrite_triggered": len(triggered),
        "rewrite_trigger_rate": (
            len(triggered) / len(followups) if followups else None
        ),
        "rewrite_succeeded": len(succeeded),
        "rewrite_success_rate": (
            len(succeeded) / len(triggered) if triggered else None
        ),
        "rewrite_noop_turns": [
            row["id"]
            for row in triggered
            if not (row.get("followup_rewrite") or {}).get("succeeded")
        ],
        "citations_outside_retrieved_turns": [
            row["id"] for row in rows if row["citations_outside_retrieved"]
        ],
        "forbidden_citation_turns": [
            row["id"] for row in rows if row["forbidden_citation_hits"]
        ],
        "history_isolation_turns": len(rows) - sum(
            1 for row in rows if row["citations_outside_retrieved"]
        ),
        "by_group": [group_summary(group) for group in groups],
        "rewrites": [
            {
                "id": row["id"],
                "question": row["question"],
                "rewritten_question": row["rewritten_question"],
                "triggered": (row.get("followup_rewrite") or {}).get("triggered"),
                "succeeded": (row.get("followup_rewrite") or {}).get("succeeded"),
            }
            for row in followups
        ],
    }


def _write_outputs(report: dict, stem: str, results_name: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / results_name).write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False) for row in report["results"]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, REPORT_DIR / f"{stem}.md")
    print(
        json.dumps(
            {"metrics": report["metrics"], "multiturn": report["multiturn"]},
            ensure_ascii=False,
            indent=2,
        )
    )


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def _write_markdown(report: dict, path: Path) -> None:
    metrics = report["metrics"]
    multi = report["multiturn"]
    lines = [
        "# 多轮评估报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 用例文件：`{report['questions_file']}`",
        f"- 组数 / 轮数：{multi['groups']} / {multi['turns']}",
        f"- 平均分：{_fmt(metrics['avg_answer_score'])}（首轮 {_fmt(multi['first_turn_avg_score'])}，"
        f"追问 {_fmt(multi['followup_avg_score'])}）",
        f"- 检索 Recall / MRR：{_fmt(metrics['retrieval_recall'])} / {_fmt(metrics['mrr'])}",
        f"- 引用命中率 / 忠实率：{_fmt(metrics['citation_rate'])} / {_fmt(metrics['faithful_rate'])}",
        f"- 拒答正确率：{_fmt(metrics['refusal_accuracy'])}",
        f"- 权限泄漏：{metrics['permission_leaks']}",
        "",
        "## 多轮专项",
        "",
        f"- 追问改写触发：{multi['rewrite_triggered']}/{multi['followup_turns']}"
        f"（{_fmt(multi['rewrite_trigger_rate'])}）",
        f"- 改写成功：{multi['rewrite_succeeded']}/{multi['rewrite_triggered']}"
        f"（{_fmt(multi['rewrite_success_rate'])}）",
        f"- 改写空转：{multi['rewrite_noop_turns']}",
        f"- 引用越出本轮检索：{multi['citations_outside_retrieved_turns']}",
        f"- 禁用引用命中（历史文档被当来源）：{multi['forbidden_citation_turns']}",
        "",
        "## 分组",
        "",
        "| 组 | 角色 | 轮数 | 均分 | 改写触发/成功 | 引用越界 | 禁用引用 | 泄漏 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group in multi["by_group"]:
        lines.append(
            f"| {group['id']} | {group['role']} | {group['turns']} "
            f"| {_fmt(group['avg_score'])} "
            f"| {group['rewrite_triggered']}/{group['rewrite_succeeded']} "
            f"| {group['citations_outside_retrieved']} "
            f"| {group['forbidden_citation_hits']} "
            f"| {group['permission_leaks']} |"
        )
    lines.extend(["", "## 非满分轮次", ""])
    below = [row for row in report["results"] if row["judge"]["score"] != 2]
    if below:
        for row in below:
            lines.append(
                f"- `{row['id']}` score={row['judge']['score']}：{row['judge']['reason']}"
            )
    else:
        lines.append("- 无")
    lines.extend(["", "## 改写明细", ""])
    for item in multi["rewrites"]:
        lines.append(
            f"- `{item['id']}` triggered={item['triggered']} "
            f"succeeded={item['succeeded']}：{item['question']} → "
            f"{item['rewritten_question']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮会话评估")
    parser.add_argument(
        "--questions", default=str(QUESTIONS_PATH), help="多轮用例 JSONL"
    )
    parser.add_argument("--groups", default="", help="只跑指定组，逗号分隔")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 组")
    parser.add_argument("--no-judge", action="store_true", help="只生成不判分")
    parser.add_argument(
        "--judge-repeats", type=int, default=1, help="每轮重复判分次数"
    )
    parser.add_argument(
        "--report-stem", default="report_multiturn", help="报告文件名前缀"
    )
    parser.add_argument(
        "--results-name", default="results_multiturn.jsonl", help="结果文件名"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只校验用例并打印调用规模，不调用 API"
    )
    args = parser.parse_args()

    groups = load_groups(Path(args.questions))
    wanted = {item.strip() for item in args.groups.split(",") if item.strip()}
    if wanted:
        groups = [group for group in groups if group["id"] in wanted]
        missing = wanted - {group["id"] for group in groups}
        if missing:
            parser.error(f"未找到组：{', '.join(sorted(missing))}")
    if args.limit:
        groups = groups[: args.limit]

    turns = sum(len(group["turns"]) for group in groups)
    followups = sum(max(0, len(group["turns"]) - 1) for group in groups)
    if args.dry_run:
        print(
            f"groups={len(groups)} turns={turns} followups={followups} "
            f"judge_calls≈{0 if args.no_judge else turns * max(1, args.judge_repeats)} "
            f"rewrite_calls≈{followups if not args.no_judge else 0}+"
        )
        return

    store = SessionStore()
    rows: list[dict] = []
    for group in groups:
        print(f"[{group['id']}] role={group['role']} turns={len(group['turns'])}")
        rows.extend(
            run_group(
                group,
                store=store,
                use_judge=not args.no_judge,
                judge_repeats=args.judge_repeats,
            )
        )

    report = _aggregate(rows, str(Path(args.questions)))
    report["multiturn"] = summarize_multiturn(rows, groups)
    report["judge_repeats"] = args.judge_repeats
    _write_outputs(report, args.report_stem, args.results_name)


if __name__ == "__main__":
    main()
