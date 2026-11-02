from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from src.config import (
    CHAT_MODEL,
    GENERATION_MAX_TOKENS,
    GENERATION_MAX_TOKENS_CEILING,
    GENERATION_MAX_TOKENS_HEADER,
    GENERATION_SEED,
    SESSION_HISTORY_CHARS,
    SESSION_HISTORY_TURNS,
    STRUCTURED_FIELD_RETRY,
)
from src.llm import chat_completion, stream_chat_completion

from .streaming import emit_delta, emit_reset

SYSTEM_PROMPT = """你是北辰智造科技的内部知识库助手。请严格依据提供的资料回答。

规则：
1. 只使用资料中的信息，不得编造；资料不足或资料中没有明确给出的信息时，必须明确说明无法回答，不得用“无”“暂无”等占位值代替答案。
2. 每个关键事实后用方括号标注文档 ID，例如 [FN-007]；必须使用上下文中的 doc_id，不得用 [资料 N] 或文档标题代替；没有来源的结论不要输出；引用前须核对引文确实出自所标 doc_id 对应的文档，多个群聊、会议纪要或同名文件并存时须按群名、日期或标题对应，不得把 A 文档的原话标成 B 文档。
3. 同一问题存在多个版本时，优先采用状态为 active 且生效日期最新的正式文件。草稿（draft）、已废止（superseded/outdated）、批注稿不得作为既定事实；草稿只能说明“该文件为草稿、未对外承诺”，不能把草稿中的方案或排期当作已确认计划。资料矛盾时要说明以哪份文件为准及原因；若资料显示旧版本已废止或被替代，须写明旧版本（版本号或年份）及其废止依据，不能只给现行结论；资料标题行、修订说明或附则中出现“替代/废止/取代”声明时，即使问题只问现行数值，也须逐一列出被替代的每个旧版本（版本号或年份）、其对应数值或结论及失效依据，不能只列其中一个；若同一事实存在旧版与现行值，须在同一句中以“旧版为 X，现行为 Y”的形式对照说明，旧版本须写明版本号或年份。
4. 表格数据可以计算，但要说明统计口径（例如仅统计“已付款”）；逐行核对，确保笔数与金额一致，不要凭印象计数。
5. 如果资料显示某个信息属于机密且当前角色无权查看，直接说明无权访问，不得透露内容。
6. 今天的日期是 {today}。判断文件是否已生效、是否为现行版本时，一律以该日期为准，不要假设当前年份。
7. 对“能否/可不可以/能不能/是否允许”类问题，除给出允许或禁止的结论外，必须写出资料中的例外条件或合规替代做法（例如“改用公司内部部署的模型或先脱敏”）；资料没有时不要臆造。
8. 引用表格或结构化字段时，保留字段名与取值的对应关系（如“目标职级：P4”“薪资：18K 以内”），不要把不同字段合并改写。当回答需要给出结构化字段（如姓名、职级、薪资、结论、日期等）时，必须逐行输出“字段名：取值”，冒号前直接写资料表格中的字段名（如“目标职级：P4”），不要写“字段名”“取值”这类字面标签；字段名须与资料表格完全一致；表格中同一对象、同一行的字段是一个整体，必须完整逐行列出（对象、日期、职级、结论、数值等），不能只列问题字面问到的字段；表格与正文简写不一致时以表格为准，不得沿用“建议”句里的简写，不得把多个字段合并成一句。
9. 区分“客户确认/正式承诺”与“内部计划”：会议纪要、拜访记录、销售预测中的日期、方案或目标，若未标明客户确认或未形成正式合同，只能表述为内部计划/建议，并写出前提条件；不得写成已确定的重启时间或交付承诺。
10. 回答使用专业中文；须完整覆盖资料中与问题相关的功能、条件、限制和版本状态，完整性优先于简洁。
11. 问“表头/正文金额与明细合计是否一致”时，逐份文档用两三行写出表头（或正文）金额、明细合计、差额和是否一致，不要逐项重算明细加总，也不要展开验算过程；不得把汇总行或折扣后的合计当作表头金额，不一致时说明不能直接采信表头金额。
12. 若两份现行文件都声明对方优先（A 规定与 B 冲突以 B 为准，B 规定与 A 冲突以 A 为准），即构成循环引用，必须明确指出双方互相指向、无法据此确定优先级，不得自行选择其中一方，也不得以“某文件不属于该制度类别”等推断绕开循环。"""

CITE_PATTERN = re.compile(r"\[([A-Z]{2,}-\d+[A-Z]?)\]")


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(today=date.today().isoformat())


def status_label(status: str) -> str:
    if status == "draft":
        return "draft（草稿，未对外承诺，不得当作已确认计划）"
    if status == "superseded":
        return "superseded（已废止，不得作为现行依据）"
    if status == "outdated":
        return "outdated（旧版/已过期，不得作为现行依据）"
    return status


def format_context(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        meta = hit["metadata"]
        blocks.append(
            f"[{meta['doc_id']}] 标题={meta['title']} | "
            f"版本={meta['version']} | 状态={status_label(meta['status'])} | "
            f"生效日期={meta['effective_date']} | 密级={meta['classification']} | "
            f"路径={meta['path']}\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def format_history(
    history: list[dict] | None,
    *,
    max_turns: int = SESSION_HISTORY_TURNS,
    max_chars: int = SESSION_HISTORY_CHARS,
) -> str:
    if not history:
        return ""
    selected = history[-max_turns:] if max_turns > 0 else []
    start = len(history) - len(selected) + 1
    blocks: list[str] = []
    used = 0
    for offset, turn in enumerate(selected):
        question = CITE_PATTERN.sub("", str(turn.get("question", ""))).strip()
        answer = CITE_PATTERN.sub("", str(turn.get("answer", ""))).strip()
        if not question and not answer:
            continue
        block = f"第{start + offset}轮 用户：{question}\n第{start + offset}轮 助手：{answer}"
        separator = 2 if blocks else 0
        remaining = max_chars - used - separator
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining]
        blocks.append(block)
        used += len(block) + separator
    return "\n\n".join(blocks)


def build_generation_messages(
    question: str, role: str, hits: list[dict], history: list[dict] | None = None
) -> tuple[list[dict], int]:
    context = format_context(hits)
    history_text = format_history(history)
    if history_text:
        user_content = (
            f"当前用户角色：{role}\n\n"
            f"对话历史（仅用于理解指代，不得作为引用来源）：\n{history_text}\n\n"
            f"参考资料：\n{context}\n\n问题：{question}"
        )
    else:
        user_content = f"当前用户角色：{role}\n\n参考资料：\n{context}\n\n问题：{question}"
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_content},
    ]
    max_tokens = (
        GENERATION_MAX_TOKENS_HEADER
        if ("表头" in question and "明细" in question)
        else GENERATION_MAX_TOKENS
    )
    return messages, max_tokens


def generate_complete(
    kind: str, messages: list[dict], max_tokens: int
) -> tuple[object, list[dict]]:
    """检测 finish_reason=length 并放宽预算重答；最多升到配置上限。"""
    attempts: list[dict] = []
    while True:
        response = chat_completion(
            kind if not attempts else f"{kind}_truncated_retry",
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            seed=GENERATION_SEED,
        )
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        attempts.append({"max_tokens": max_tokens, "finish_reason": finish_reason})
        if finish_reason != "length" or max_tokens >= GENERATION_MAX_TOKENS_CEILING:
            break
        max_tokens = min(max_tokens * 2, GENERATION_MAX_TOKENS_CEILING)
    return response, attempts


# 结构化字段校验：从“字段：值”或两列表格行中取字段名，若回答里只出现取值而缺字段名，
# 判为字段合并并重试一次。只使用检索到的原文，不读取题目 key_facts。
STRUCTURED_LINE_RE = re.compile(r"^\s*([^|：:\n]{1,24}?)\s*[：:]\s*(.{1,40}?)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*([^|\n]{1,24}?)\s*\|\s*([^|\n]{1,40}?)\s*$")
VALUE_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|[A-Za-z]\d{1,4}|\d{1,4}[A-Za-z]|\d{3,}")
GENERIC_FIELD_LABELS = {"项", "内容", "维度", "评分", "字段名", "取值", "备注", "说明"}
# 模型有时把提示词里的“字段名”当成字面标签输出，属于确定性问题，直接去掉前缀即可
LITERAL_FIELD_PREFIX_RE = re.compile(r"^[ \t]*字段名\s*[：:]\s*", re.MULTILINE)


def _norm_field_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _strip_literal_field_prefix(text: str) -> str:
    return LITERAL_FIELD_PREFIX_RE.sub("", text)


def _structured_candidates(hits: list[dict], cited_ids: set[str]) -> dict[str, list[str]]:
    """收集已引用文档中的（字段名 → 可能的取值序列），用于确定性格式校验。"""
    candidates: dict[str, list[str]] = defaultdict(list)
    for hit in hits:
        meta = hit["metadata"]
        if meta["doc_id"] not in cited_ids:
            continue
        for line in hit["text"].splitlines():
            if set(line.strip()) <= {"|", "-", " ", ":"}:
                continue
            match = TABLE_ROW_RE.match(line) or STRUCTURED_LINE_RE.match(line)
            if not match:
                continue
            label, value = match.group(1).strip(), match.group(2).strip()
            # 字段名应为短标签：含数字的标题行（如“版本 2026.1 生效日期”）不是字段
            if (
                not label
                or not value
                or label in GENERIC_FIELD_LABELS
                or len(label) > 12
                or re.search(r"[\d=＝]", label)
            ):
                continue
            candidates[label].append(value)
    return candidates


def structured_field_violations(text: str, hits: list[dict]) -> list[str]:
    cited_ids = set(CITE_PATTERN.findall(text))
    if not cited_ids:
        return []
    answer_norm = _norm_field_text(text)
    violations: list[str] = []
    for label, values in _structured_candidates(hits, cited_ids).items():
        if _norm_field_text(label) in answer_norm:
            continue
        if any(
            token
            and re.search(
                rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text
            )
            for value in values
            for token in VALUE_TOKEN_RE.findall(value)
        ):
            violations.append(label)
    return sorted(set(violations))


def ensure_draft_disclosure(text: str, hits: list[dict]) -> str:
    """回答引用了草稿却没说明状态时，按元数据补一条声明，避免把草稿内容当作已确认计划。"""
    meta_by_id: dict[str, dict] = {}
    for hit in hits:
        meta = hit["metadata"]
        meta_by_id.setdefault(meta["doc_id"], meta)
    drafts = [
        (doc_id, meta_by_id[doc_id])
        for doc_id in sorted(set(CITE_PATTERN.findall(text)) & set(meta_by_id))
        if meta_by_id[doc_id].get("status") == "draft"
    ]
    if not drafts or "草稿" in text or "draft" in text.lower() or "未对外承诺" in text:
        return text
    notes = "；".join(
        f"[{doc_id}]《{meta.get('title', '')}》为草稿版本，未对外承诺"
        for doc_id, meta in drafts
    )
    return f"{text.rstrip()}\n\n注：{notes}，其中的排期与方案仅用于内部讨论。"


def sources(hits: list[dict]) -> list[dict]:
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


def _postprocess_text(text: str, hits: list[dict]) -> tuple[str, dict]:
    text = text.strip()
    stripped = _strip_literal_field_prefix(text)
    literal_prefix_stripped = stripped != text
    text = stripped
    violations = structured_field_violations(text, hits)
    format_check = {
        "violations": violations,
        "retry_enabled": STRUCTURED_FIELD_RETRY,
        "retried": False,
        "remaining": [],
        "literal_prefix_stripped": literal_prefix_stripped,
    }
    return text, format_check


def _complete_blocking(kind: str, messages: list[dict], max_tokens: int):
    response, attempts = generate_complete(kind, messages, max_tokens)
    return response.choices[0].message.content, attempts


def _complete_streaming(kind: str, messages: list[dict], max_tokens: int):
    attempts: list[dict] = []
    while True:
        if attempts:
            emit_reset(
                "truncation_retry",
                attempt=len(attempts) + 1,
                max_tokens=max_tokens,
            )
        text, finish_reason, _ = stream_chat_completion(
            kind if not attempts else f"{kind}_truncated_retry",
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            seed=GENERATION_SEED,
            on_delta=emit_delta,
        )
        attempts.append({"max_tokens": max_tokens, "finish_reason": finish_reason})
        if finish_reason != "length" or max_tokens >= GENERATION_MAX_TOKENS_CEILING:
            break
        max_tokens = min(max_tokens * 2, GENERATION_MAX_TOKENS_CEILING)
    return text, attempts


def _generate_answer(
    question: str,
    role: str,
    hits: list[dict],
    history: list[dict] | None,
    complete,
    *,
    on_reset=None,
) -> dict:
    messages, max_tokens = build_generation_messages(question, role, hits, history)
    raw_text, generation_attempts = complete("generate", messages, max_tokens)
    max_tokens = generation_attempts[-1]["max_tokens"]
    text, format_check = _postprocess_text(raw_text, hits)
    violations = format_check["violations"]
    if violations and STRUCTURED_FIELD_RETRY:
        correction = (
            "上一版回答缺少以下字段名或把字段与其它内容合并："
            f"{'、'.join(violations)}。请保留原有事实与引用，按“字段名：取值”逐行重写，"
            "字段名须与资料表格一致，不得把多个字段合并成一句。"
        )
        if on_reset is not None:
            on_reset("field_retry", max_tokens=max_tokens)
        retry_raw, retry_attempts = complete(
            "generate_retry",
            [
                *messages,
                {"role": "assistant", "content": text},
                {"role": "user", "content": correction},
            ],
            max_tokens,
        )
        retry_text, retry_check = _postprocess_text(retry_raw, hits)
        remaining = retry_check["violations"]
        format_check.update(
            {
                "retried": True,
                "remaining": remaining,
                "fixed": not remaining,
                "retry_truncation": retry_attempts,
            }
        )
        if len(remaining) <= len(violations):
            text = retry_text
    return {
        "text": text,
        "format_check": format_check,
        "generation_check": {
            "truncation": generation_attempts,
            "retried_on_truncation": len(generation_attempts) > 1,
        },
    }


def generate_rag_answer(
    question: str, role: str, hits: list[dict], history: list[dict] | None = None
) -> dict:
    """与重构前（无历史时）的 RAG 生成路径逐字一致，供图节点直接复用。"""
    return _generate_answer(question, role, hits, history, _complete_blocking)


def generate_rag_answer_stream(
    question: str, role: str, hits: list[dict], history: list[dict] | None = None
) -> dict:
    return _generate_answer(
        question, role, hits, history, _complete_streaming, on_reset=emit_reset
    )
