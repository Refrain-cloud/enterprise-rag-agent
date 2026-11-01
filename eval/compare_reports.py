from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import BUILD_DIR

REPORT_DIR = BUILD_DIR / "eval"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_rank(row: dict) -> int | None:
    gold = set(row.get("gold_doc_ids") or [])
    for rank, doc_id in enumerate(row.get("retrieved_ids") or [], 1):
        if doc_id in gold:
            return rank
    return None


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _display_path(value: str | None) -> str | None:
    if not value:
        return value
    try:
        return str(Path(value).resolve().relative_to(BUILD_DIR.parent))
    except ValueError:
        return value


def _metric(report: dict, key: str) -> float | None:
    return (report.get("metrics") or {}).get(key)


def _latency(report: dict, key: str) -> float | None:
    cost = (report.get("cost") or {}).get("per_question") or {}
    return cost.get(key)


def _cost(report: dict, phase: str = "total") -> float | None:
    return ((report.get("cost") or {}).get(phase) or {}).get("cost_cny")


def _rerank_cost(report: dict) -> float | None:
    generation = (report.get("cost") or {}).get("generation") or {}
    rerank = (generation.get("by_kind") or {}).get("rerank") or {}
    return rerank.get("cost_cny")


def _key_fact_statuses(row: dict) -> list[str] | None:
    """只对齐题面给定的必答点（K1..Kn），避免 judge 自拆点的措辞漂移。"""
    facts = [str(item).strip() for item in (row.get("key_facts") or []) if str(item).strip()]
    if not facts:
        return None
    by_id = {
        str(fact.get("id", "")): str(fact.get("status", "absent"))
        for fact in (row.get("judge") or {}).get("facts") or []
    }
    return [by_id.get(f"K{index}", "absent") for index in range(1, len(facts) + 1)]


def compare(base: dict, other: dict) -> dict:
    base_rows = {row["id"]: row for row in base["results"]}
    other_rows = {row["id"]: row for row in other["results"]}
    common = sorted(set(base_rows) & set(other_rows))

    improvements: list[dict] = []
    regressions: list[dict] = []
    changed: list[dict] = []
    fact_changes: list[dict] = []
    for qid in common:
        before = base_rows[qid]
        after = other_rows[qid]
        before_rank = _first_rank(before)
        after_rank = _first_rank(after)
        before_mrr = before.get("reciprocal_rank", 0.0)
        after_mrr = after.get("reciprocal_rank", 0.0)
        row = {
            "id": qid,
            "category": before.get("category"),
            "before_rank": before_rank,
            "after_rank": after_rank,
            "mrr_delta": round(after_mrr - before_mrr, 4),
            "route_before": before.get("route"),
            "route_after": after.get("route"),
            "cite_before": before.get("cites_gold"),
            "cite_after": after.get("cites_gold"),
            "score_before": (before.get("judge") or {}).get("score"),
            "score_after": (after.get("judge") or {}).get("score"),
        }
        if row["mrr_delta"] > 0:
            improvements.append(row)
        elif row["mrr_delta"] < 0:
            regressions.append(row)
        if (
            row["mrr_delta"]
            or row["route_before"] != row["route_after"]
            or row["cite_before"] != row["cite_after"]
            or (row["score_before"] != row["score_after"] and row["score_after"] is not None)
        ):
            changed.append(row)
        facts_before = _key_fact_statuses(before)
        facts_after = _key_fact_statuses(after)
        judged_both = row["score_before"] is not None and row["score_after"] is not None
        if judged_both and facts_before is not None and facts_after is not None and facts_before != facts_after:
            fact_changes.append(
                {
                    "id": qid,
                    "category": before.get("category"),
                    "before": facts_before,
                    "after": facts_after,
                    "score_before": row["score_before"],
                    "score_after": row["score_after"],
                }
            )
    improvements.sort(key=lambda row: row["mrr_delta"], reverse=True)
    regressions.sort(key=lambda row: row["mrr_delta"])

    metric_keys = (
        "retrieval_recall",
        "gold_doc_coverage",
        "gold_doc_full_rate",
        "mrr",
        "citation_rate",
        "avg_answer_score",
        "faithful_rate",
        "refusal_accuracy",
        "permission_ok_rate",
        "permission_leaks",
        "unsupported_claims_count",
    )
    metrics = {}
    for key in metric_keys:
        before = _metric(base, key)
        after = _metric(other, key)
        metrics[key] = {
            "before": before,
            "after": after,
            "delta": (
                round(after - before, 4)
                if isinstance(before, (int, float)) and isinstance(after, (int, float))
                else None
            ),
        }
    return {
        "baseline": _display_path(base.get("questions_file")),
        "comparison": _display_path(other.get("questions_file")),
        "question_count": len(common),
        "metrics": metrics,
        "latency": {
            "avg_latency_s": {
                "before": _latency(base, "avg_latency_s"),
                "after": _latency(other, "avg_latency_s"),
            },
            "avg_retrieval_latency_s": {
                "before": _latency(base, "avg_retrieval_latency_s"),
                "after": _latency(other, "avg_retrieval_latency_s"),
            },
            "avg_rerank_latency_s": {
                "before": _latency(base, "avg_rerank_latency_s"),
                "after": _latency(other, "avg_rerank_latency_s"),
            },
        },
        "cost_cny": {
            "total": {"before": _cost(base), "after": _cost(other)},
            "generation": {
                "before": _cost(base, "generation"),
                "after": _cost(other, "generation"),
            },
            "judge": {"before": _cost(base, "judge"), "after": _cost(other, "judge")},
            "rerank": {"before": _rerank_cost(base), "after": _rerank_cost(other)},
        },
        "judge_agreement": {
            "repeats": {
                "before": base.get("judge_repeats"),
                "after": other.get("judge_repeats"),
            },
            "score_agreement": {
                "before": base.get("judge_score_agreement"),
                "after": other.get("judge_score_agreement"),
            },
        },
        "rank_changes": {
            "improved": len(improvements),
            "regressed": len(regressions),
            "unchanged": len(common) - len(improvements) - len(regressions),
        },
        "improvements": improvements,
        "regressions": regressions,
        "changed": changed,
        "fact_changes": fact_changes,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Rerank 对比报告",
        "",
        f"- 基线：{report['baseline']}",
        f"- 对比：{report['comparison']}",
        f"- 共同题数：{report['question_count']}",
        "",
        "## 指标对比",
        "",
        "| 指标 | 基线 | 对比 | 变化 |",
        "|---|---|---|---|",
    ]
    for key, item in report["metrics"].items():
        lines.append(
            f"| {key} | {_fmt(item['before'])} | {_fmt(item['after'])} | "
            f"{_fmt(item['delta'], 4)} |"
        )
    latency = report["latency"]
    cost = report["cost_cny"]
    lines.extend([
        "",
        "## 延迟与费用",
        "",
        "| 项 | 基线 | 对比 |",
        "|---|---|---|",
        f"| 平均端到端延迟 (s) | {_fmt(latency['avg_latency_s']['before'])} | "
        f"{_fmt(latency['avg_latency_s']['after'])} |",
        f"| 平均检索延迟 (s) | {_fmt(latency['avg_retrieval_latency_s']['before'])} | "
        f"{_fmt(latency['avg_retrieval_latency_s']['after'])} |",
        f"| 平均 rerank 延迟 (s/题) | {_fmt(latency['avg_rerank_latency_s']['before'])} | "
        f"{_fmt(latency['avg_rerank_latency_s']['after'])} |",
        f"| 生成费用 (元) | {_fmt(cost['generation']['before'], 4)} | "
        f"{_fmt(cost['generation']['after'], 4)} |",
        f"| 其中 rerank 费用 (元) | {_fmt(cost['rerank']['before'], 4)} | "
        f"{_fmt(cost['rerank']['after'], 4)} |",
        f"| judge 费用 (元) | {_fmt(cost['judge']['before'], 4)} | "
        f"{_fmt(cost['judge']['after'], 4)} |",
        f"| 合计 (元) | {_fmt(cost['total']['before'], 4)} | "
        f"{_fmt(cost['total']['after'], 4)} |",
        "",
        "> 合计费用与端到端延迟受是否判分影响，跨报告比较以检索延迟与生成费用为准。",
        "",
        "## 判分一致性",
        "",
        "| 项 | 基线 | 对比 |",
        "|---|---|---|",
        f"| 每题判分次数 | {report['judge_agreement']['repeats']['before'] or '-'} | "
        f"{report['judge_agreement']['repeats']['after'] or '-'} |",
        f"| 三轮分数一致率 | {_fmt(report['judge_agreement']['score_agreement']['before'])} | "
        f"{_fmt(report['judge_agreement']['score_agreement']['after'])} |",
        "",
        "## 排名与判分变化",
        "",
        f"- 提升：{report['rank_changes']['improved']} 题；"
        f"下降：{report['rank_changes']['regressed']} 题；"
        f"不变：{report['rank_changes']['unchanged']} 题",
        "",
        "| ID | 类别 | 基线排名 | rerank 排名 | MRR 变化 | 路由 | 引用 | 分数 |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for row in sorted(report["changed"], key=lambda item: -abs(item["mrr_delta"])):
        score = (
            f"{row['score_before']}→{row['score_after']}"
            if row["score_after"] is not None
            else "-"
        )
        cites = f"{'Y' if row['cite_before'] else 'N'}→{'Y' if row['cite_after'] else 'N'}"
        lines.append(
            f"| {row['id']} | {row['category']} | {row['before_rank'] or '-'} | "
            f"{row['after_rank'] or '-'} | {row['mrr_delta']:+.4f} | "
            f"{row['route_before']}→{row['route_after']} | {cites} | {score} |"
        )
    if not report["changed"]:
        lines.append("| - | - | - | - | - | - | - | - |")
    lines.extend([
        "",
        "## 必答点状态变化",
        "",
        "| ID | 类别 | 基线状态 | rerank 状态 | 分数 |",
        "|---|---|---|---|---|",
    ])
    for row in report["fact_changes"]:
        score = (
            f"{row['score_before']}→{row['score_after']}"
            if row["score_after"] is not None
            else "-"
        )
        lines.append(
            f"| {row['id']} | {row['category']} | {','.join(row['before'])} | "
            f"{','.join(row['after'])} | {score} |"
        )
    if not report["fact_changes"]:
        lines.append("| - | - | - | - | - |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两份评估报告的检索与成本指标")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--stem", default="report_compare")
    args = parser.parse_args()

    report = compare(_load(args.baseline), _load(args.comparison))
    stem = args.stem
    (REPORT_DIR / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(report, REPORT_DIR / f"{stem}.md")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["rank_changes"], ensure_ascii=False))


if __name__ == "__main__":
    main()
