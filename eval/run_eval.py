from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from src.agent.ask import answer
from src.config import CHAT_MODEL, BUILD_DIR, TOP_K, get_client
from src.retrieval.search import hybrid_search

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.v1.jsonl"
REPORT_DIR = BUILD_DIR / "eval"
CORPUS_PATH = BUILD_DIR / "corpus" / "docs.jsonl"
CITE_PATTERN = re.compile(r"\[([A-Z]+-\d+[A-Z]?)\]")
FULL_DOC_CHARS = 12000

REFUSAL_MARKERS = [
    "无权", "无法访问", "无法回答", "无法确定", "无法判断", "不能提供",
    "没有找到", "未找到", "没有相关", "不存在", "已终止", "没有下一阶段",
]
STRICT_REFUSAL_MARKERS = [
    "无权", "无法访问", "无法回答", "无法确定", "无法判断",
    "不能提供", "没有找到", "未找到", "没有相关", "无法获取",
]
ABSENCE_REFUSAL_MARKERS = [
    "不存在", "未提及", "已终止", "没有下一阶段", "无下一阶段",
    "没有相关信息", "无相关信息",
]

JUDGE_PROMPT = """你是 RAG 评估中的事实提取员，只做核对，不评分，不写分析。
只输出一个 JSON 对象：
{{"facts": [{{"id": "F1", "claim": "事实点", "status": "covered", "quote": "从助手回答逐字复制的支撑片段"}}],
  "contradictions": [{{"claim": "回答中的陈述", "evidence": "资料中与之冲突的原文"}}],
  "refused": false}}

规则：
1. 拆事实点：把标准答案拆成必答事实点（每个数字、日期、版本号、名称、结论、条件各算一个）。括号内的计算过程、来源说明、补充说明不算独立事实点。
   - 标准答案“9,530 元（4,280 + 3,150 + 2,100）”只算一个事实点，括号内分项不是独立事实点。
   - 标准答案“缺口 4 人（编制 45 人，在岗 41 人）”只算一个事实点，括号内为计算依据。
   - 标准答案“现场差旅增加（恒远精工、华中二期现场周期延长）”只算一个事实点，括号内为补充说明。
2. status 只能是：
   - covered：助手在回答正文中明确写出该事实（仅标注引用来源不算）；
   - missing：回答正文没有写；
   - wrong：回答写了，但与标准答案或资料矛盾。
   - covered 和 wrong 必须给出 quote：只从最后的“助手回答”中逐字复制支撑片段，不要改写、不要加引号，也不要从检索片段或文件全文中复制。quote 无法在回答中逐字匹配时，按 missing 处理。
   - 如果回答只是复述资料原文、没有针对该事实作出表述，也不能算 covered。
3. contradictions 只记录回答与检索片段或文件全文直接冲突的陈述；evidence 必须逐字引用冲突原文。找不到直接冲突原文时，contradictions 必须为空。禁止根据制度效力层级、法律解释、“一般条款隐含”等推理判定冲突。
   回答中每条关于资料内容的否定性陈述（“未规定/未重新规定/未提及/没有/不存在/不适用”），都必须回到文件全文（含表格）核查相关章节；只要文件存在相反内容，就必须记冲突，不能因为回答总体结论正确就忽略局部错误陈述。
   对状态/密级/版本号等元数据字段的解读、效力推断，不算直接冲突；evidence 必须来自文件正文或表格，不能只是片段开头的元数据行。
   矛盾必须是两个陈述相互对立。如果回答与 evidence 表达同一结论（例如两者都说某文件已废止），这不是矛盾，不要记录。
   示例：回答称“某制度未重新规定住宿标准”，而该制度全文表格中列有该标准 → 记冲突，evidence 逐字引用该表格行。
   示例：回答称“某文件未对某情形作特别规定”，而全文确实没有该情形的专条 → 不记冲突。
4. refused：回答是否明确表示无法回答、无权访问或信息不存在，而不是给出实质答案。
5. 只核对，不要输出分数、理由或建议。

评估对象：
问题：{question}
用户角色：{role}
类别：{category}
expect_refusal：{expect_refusal}
标准答案：{gold_answer}

检索片段（助手回答时看到的原文，可能被截断）：
{context}

文件全文（用于核对回答是否与资料冲突；助手当时未看到全文，不要求它使用）：
{full_docs}

助手回答：{model_answer}
"""


CONTRADICTION_CHECK_PROMPT = """判断“资料原文”是否直接否定“回答陈述”。只输出 yes 或 no。
- 资料原文明确说明回答中的同一对象、同一数值或同一结论不成立 → yes
- 资料原文没有提到该对象、两者可以同时成立、或资料支持回答 → no
不要考虑制度层级、法律解释或隐含含义。
示例：
回答：X 已被废止。资料：原 X 同时废止。→ no
回答：某制度未规定住宿标准。资料：表格列出该住宿标准。→ yes
回答：X 已废止。资料：本手册与专项制度不一致时以专项制度为准。→ no

回答陈述：{claim}
资料原文：{evidence}
"""


def _is_direct_contradiction(claim: str, evidence: str) -> bool:
    prompt = CONTRADICTION_CHECK_PROMPT.format(claim=claim, evidence=evidence[:1500])
    for attempt in range(2):
        try:
            response = get_client().chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )
            text = response.choices[0].message.content.strip().lower()
            return text.startswith("yes") or text.startswith("是")
        except Exception:
            time.sleep(2**attempt)
    return False


@lru_cache(maxsize=1)
def _load_doc_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            texts[doc["doc_id"]] = doc.get("text", "")
    return texts


def _judge_excerpt(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{hit['metadata']['doc_id']}] 标题={hit['metadata']['title']} | "
        f"状态={hit['metadata']['status']} | 密级={hit['metadata']['classification']}\n"
        f"{hit['text'][:1200]}"
        for hit in hits
    ) or "（无）"


def _full_docs_context(model_answer: str, gold_ids: list[str]) -> str:
    texts = _load_doc_texts()
    cited = set(CITE_PATTERN.findall(model_answer))
    blocks: list[str] = []
    used = 0
    for doc_id in sorted(cited | set(gold_ids)):
        text = texts.get(doc_id)
        if not text:
            continue
        block = f"===== {doc_id} 全文 =====\n{text}"
        remaining = FULL_DOC_CHARS - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining > 500:
                blocks.append(block[:remaining] + "\n（全文过长，已截断）")
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) or "（无）"


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _canon(text: str) -> str:
    normalized = _norm(text)
    for variant, canonical in (("不可以", "不能"), ("不可", "不能"), ("不予", "不能")):
        normalized = normalized.replace(variant, canonical)
    return normalized


def _quote_supports_claim(claim: str, quote: str, answer: str = "") -> bool:
    quote_norm = _canon(quote)
    claim_norm = _canon(claim)
    if not quote_norm or not claim_norm:
        return False
    tokens = re.findall(r"[A-Za-z0-9]+", claim)
    if tokens:
        scope = _norm(answer) if answer else quote_norm
        return all(_norm(token) in scope for token in tokens)
    if len(claim_norm) <= 2:
        return claim_norm in quote_norm
    if claim_norm in quote_norm:
        return True
    bigrams = [claim_norm[i : i + 2] for i in range(len(claim_norm) - 1)]
    return sum(1 for bigram in bigrams if bigram in quote_norm) >= 2


def _looks_like_contradiction(claim: str, evidence: str) -> bool:
    if re.search(r"(未|没有|不存在|无).{0,6}(规定|提及|包含|列出|说明|明确)", claim):
        return True
    claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim))
    evidence_nums = set(re.findall(r"\d+(?:\.\d+)?", evidence))
    if claim_nums and evidence_nums and claim_nums - evidence_nums:
        return True
    if re.search(r"(废止|失效|终止|停止|无效)", claim) and re.search(
        r"(有效|现行|active|仍在|仍在进行)", evidence
    ):
        return True
    if re.search(r"(有效|现行|active|仍在|仍在进行)", claim) and re.search(
        r"(废止|失效|终止|停止|无效)", evidence
    ):
        return True
    return False


def _agrees_on_discontinuation(claim: str, evidence: str) -> bool:
    for term in ("废止", "失效", "终止"):
        if term in claim and term in evidence:
            negated = r"(未|没有|不|无法).{0,3}" + term
            if not re.search(negated, claim) and not re.search(negated, evidence):
                return True
    return False


def _doc_blocks(context: str, full_docs: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    excerpt = r"\[([A-Z]+-\d+[A-Z]?)\][^\n]*\n(.*?)(?=\n\n\[[A-Z]+-\d+[A-Z]?\][^\n]*\n|\Z)"
    for match in re.finditer(excerpt, context, re.S):
        blocks.append((match.group(1), match.group(2)))
    full = r"===== ([A-Z]+-\d+[A-Z]?) 全文 =====\n(.*?)(?=\n\n===== [A-Z]+-\d+[A-Z]? 全文 =====|\Z)"
    for match in re.finditer(full, full_docs, re.S):
        blocks.append((match.group(1), match.group(2)))
    return blocks


def _assemble_judgement(
    question: dict,
    data: dict,
    context: str,
    full_docs: str,
    leaked: bool,
    model_answer: str = "",
) -> dict:
    answer_norm = _norm(model_answer)
    facts: list[dict] = []
    raw_facts = data.get("facts")
    for index, item in enumerate(raw_facts if isinstance(raw_facts, list) else [], 1):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        status = item.get("status")
        if not claim or status not in {"covered", "missing", "wrong"}:
            continue
        quote = str(item.get("quote", "")).strip()
        if status == "missing":
            base = re.split(r"[（(]", claim, maxsplit=1)[0].strip()
            if len(_norm(base)) >= 3 and _norm(base) in answer_norm:
                claim = base
                quote = base
                status = "covered"
        if status in {"covered", "wrong"}:
            if (
                not _norm(quote)
                or _norm(quote) not in answer_norm
                or not _quote_supports_claim(claim, quote, model_answer)
            ):
                status = "missing"
                quote = ""
        facts.append(
            {
                "id": str(item.get("id") or f"F{index}"),
                "claim": claim,
                "status": status,
                "quote": quote,
            }
        )

    source = _norm(f"{context}\n{full_docs}")
    blocks = _doc_blocks(context, full_docs)
    contradictions: list[dict] = []
    raw_contradictions = data.get("contradictions")
    for item in raw_contradictions if isinstance(raw_contradictions, list) else []:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        evidence_norm = _norm(evidence)
        if not claim or len(evidence_norm) < 4 or evidence_norm not in source:
            continue
        if re.match(r"^\[[A-Z]+-\d+[A-Z]?\]\s*标题=", evidence):
            continue
        claim_ids = set(CITE_PATTERN.findall(claim))
        if claim_ids:
            evidence_ids = {
                doc_id for doc_id, text in blocks if evidence_norm in _norm(text)
            }
            if evidence_ids and evidence_ids.isdisjoint(claim_ids):
                continue
        if _agrees_on_discontinuation(claim, evidence):
            continue
        if not _looks_like_contradiction(claim, evidence):
            continue
        if not _is_direct_contradiction(claim, evidence):
            continue
        contradictions.append({"claim": claim, "evidence": evidence})

    expect_refusal = bool(question.get("expect_refusal"))
    refused = bool(data.get("refused")) or any(
        marker in model_answer for marker in STRICT_REFUSAL_MARKERS
    )
    if not refused and expect_refusal:
        refused = any(marker in model_answer for marker in ABSENCE_REFUSAL_MARKERS)
    statuses = [fact["status"] for fact in facts]
    covered = any(status == "covered" for status in statuses)
    wrong = any(status == "wrong" for status in statuses)

    if expect_refusal:
        score = 2 if refused else 0
    elif refused:
        score = 0
    elif not facts:
        score = 1 if contradictions else 2
    elif all(status == "covered" for status in statuses) and not contradictions:
        score = 2
    elif covered:
        score = 1
    else:
        score = 0

    missing_claims = [f["claim"] for f in facts if f["status"] == "missing"]
    wrong_claims = [f["claim"] for f in facts if f["status"] == "wrong"]
    parts: list[str] = []
    if missing_claims:
        parts.append("缺失：" + "；".join(missing_claims[:6]))
    if wrong_claims:
        parts.append("错误：" + "；".join(wrong_claims[:6]))
    if contradictions:
        parts.append(
            "矛盾：" + "；".join(
                f"{item['claim']}（证据：{item['evidence'][:80]}）" for item in contradictions[:4]
            )
        )
    parts.append(f"refused={refused}")
    if leaked:
        parts.append("检测到权限泄漏")

    return {
        "score": score,
        "faithful": (not wrong) and (not contradictions),
        "refusal_ok": refused == expect_refusal,
        "permission_ok": not leaked,
        "reason": "；".join(parts),
        "facts": facts,
        "contradictions": contradictions,
        "unsupported_claims": contradictions,
        "refused": refused,
    }


def judge(
    question: dict,
    model_answer: str,
    context: str = "",
    full_docs: str = "",
    leaked: bool = False,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question["question"],
        role=question["role"],
        category=question["category"],
        expect_refusal=question.get("expect_refusal", False),
        gold_answer=question["gold_answer"],
        context=context or "（无）",
        full_docs=full_docs or "（无）",
        model_answer=model_answer,
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = get_client().chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1200,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                data = json.loads(match.group(0))
                return _assemble_judgement(
                    question, data, context or "", full_docs or "", leaked, model_answer
                )
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
        "facts": [],
        "contradictions": [],
        "unsupported_claims": [],
        "refused": None,
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

        judged = {
            "score": None, "faithful": None, "refusal_ok": None,
            "permission_ok": None, "reason": "", "facts": [],
            "contradictions": [], "unsupported_claims": [], "refused": None,
        }
        if use_judge:
            judged = judge(
                question,
                answer_text,
                context=_judge_excerpt(response["hits"]),
                full_docs=_full_docs_context(answer_text, gold_ids),
                leaked=bool(leaked_ids or leaked_terms),
            )

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
            f"score={judged['score']} faithful={judged['faithful']} "
            f"unsupported={len(judged['unsupported_claims'])} "
            f"leak={leaked_ids + leaked_terms}"
        )

    report = _aggregate(results, str(QUESTIONS_PATH))
    _write_outputs(report, REPORT_DIR, "report", "results.jsonl")
    return report


def rejudge(results_path: Path, limit: int | None, ids: set[str] | None = None) -> dict:
    rows = [json.loads(line) for line in results_path.open(encoding="utf-8")]
    if ids:
        rows = [row for row in rows if row["id"] in ids]
    if limit:
        rows = rows[:limit]

    results: list[dict] = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] rejudge {row['id']}")
        if row["retrieved_ids"]:
            hits = hybrid_search(row["question"], role=row["role"], top_k=TOP_K)
            got = [hit["metadata"]["doc_id"] for hit in hits]
            if got != row["retrieved_ids"]:
                print(f"  WARN: retrieved ids differ\n    now={got}\n    old={row['retrieved_ids']}")
        else:
            hits = []
        judged = judge(
            row,
            row["model_answer"],
            context=_judge_excerpt(hits),
            full_docs=_full_docs_context(row["model_answer"], row["gold_doc_ids"]),
            leaked=bool(row.get("leaked_ids") or row.get("leaked_terms")),
        )
        results.append({**row, "judge": judged})
        print(
            f"  score={judged['score']} faithful={judged['faithful']} "
            f"unsupported={len(judged['unsupported_claims'])}"
        )

    report = _aggregate(results, str(results_path))
    report["rejudged_from"] = str(results_path)
    _write_outputs(report, REPORT_DIR, "report_rejudge", "results_rejudge.jsonl")
    return report


def _aggregate(results: list[dict], source: str) -> dict:
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
        sum(1 for r in gold_results if r["cites_gold"]) / len(gold_results)
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
    faithful_rate = (
        sum(1 for r in judged_results if r["judge"]["faithful"]) / len(judged_results)
        if judged_results else None
    )
    unsupported_count = sum(
        len(r["judge"].get("unsupported_claims") or []) for r in judged_results
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

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "questions_file": source,
        "question_count": len(results),
        "metrics": {
            "retrieval_recall": retrieval_recall,
            "mrr": mrr,
            "citation_rate": citation_rate,
            "avg_answer_score": avg_score,
            "faithful_rate": faithful_rate,
            "unsupported_claims_count": unsupported_count,
            "refusal_accuracy": refusal_accuracy,
            "permission_ok_rate": permission_ok,
            "permission_leaks": len(permission_leaks),
            "score_distribution": dict(sorted(Counter(r["judge"]["score"] for r in judged_results).items())),
            "route_counts": dict(sorted(Counter(result["route"] for result in results).items())),
        },
        "by_category": category_metrics,
        "results": results,
    }


def _write_outputs(report: dict, directory: Path, stem: str, results_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / results_name).write_text(
        "\n".join(json.dumps(result, ensure_ascii=False) for result in report["results"]) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, directory / f"{stem}.md")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


def _write_markdown(report: dict, path: Path) -> None:
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
        f"- 忠实率：{metrics['faithful_rate']}",
        f"- 无依据断言数：{metrics['unsupported_claims_count']}",
        f"- 得分分布：{metrics['score_distribution']}",
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
        recall_text = f"{recall:.3f}" if recall is not None else "-"
        avg_text = f"{avg:.2f}" if avg is not None else "-"
        lines.append(
            f"| {category} | {metric['count']} | {recall_text} | {avg_text} | {metric['routes']} |"
        )
    lines.extend([
        "",
        "## 题目明细",
        "",
        "| ID | 类别 | 角色 | 路由 | 检索命中 | 引用命中 | 得分 | 忠实 | 无依据断言 | 泄漏 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for result in report["results"]:
        leaks = ",".join(result["leaked_ids"] + result["leaked_terms"]) or "-"
        unsupported = len(result["judge"].get("unsupported_claims") or [])
        lines.append(
            f"| {result['id']} | {result['category']} | {result['role']} | "
            f"{result['route']} | "
            f"{'Y' if result['retrieval_hit'] else 'N'} | "
            f"{'Y' if result['cites_gold'] else 'N'} | "
            f"{result['judge']['score']} | "
            f"{'Y' if result['judge']['faithful'] else 'N'} | "
            f"{unsupported} | {leaks} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="复用已存回答离线复判，不调用问答模型",
    )
    parser.add_argument(
        "--results",
        default=str(REPORT_DIR / "results.jsonl"),
        help="--rejudge 读取的结果文件",
    )
    parser.add_argument(
        "--ids",
        default="",
        help="只复判指定题号，逗号分隔（配合 --rejudge）",
    )
    args = parser.parse_args()
    if args.rejudge:
        ids = {item.strip() for item in args.ids.split(",") if item.strip()} or None
        rejudge(Path(args.results), args.limit, ids)
    else:
        evaluate(limit=args.limit, use_judge=not args.no_judge)
