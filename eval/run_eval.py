from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from src.agent.ask import answer
from src.config import CHAT_MODEL, BUILD_DIR, TOP_K
from src.llm import chat_completion, merge_usage, reset_usage, usage_snapshot
from src.retrieval.search import hybrid_search, reset_search_stats, search_stats

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.v3.jsonl"
REPORT_DIR = BUILD_DIR / "eval"
CORPUS_PATH = BUILD_DIR / "corpus" / "docs.jsonl"
CITE_PATTERN = re.compile(r"\[([A-Z]+-\d+[A-Z]?)\]")
FULL_DOC_CHARS = 12000

ROOT = Path(__file__).resolve().parents[1]
# 只覆盖运行期影响回答的代码与派生索引；改 ingest/api/ui 不会让已存回答失效
ANSWER_SOURCE_DIRS = ("agent", "retrieval", "security", "tools")
ANSWER_SOURCE_FILES = (
    *sorted(
        path
        for name in ANSWER_SOURCE_DIRS
        for path in (ROOT / "src" / name).rglob("*.py")
    ),
    ROOT / "src" / "config.py",
    ROOT / "src" / "llm.py",
    BUILD_DIR / "corpus" / "docs.jsonl",
    BUILD_DIR / "index" / "chunks.jsonl",
    BUILD_DIR / "index" / "bm25.pkl",
    BUILD_DIR / "index" / "build_summary.json",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def _answer_fingerprint() -> str:
    """回答指纹：代码、索引产物或 RAG_* 环境变量任一变化，已存回答即失效。"""
    parts = [
        f"{path.relative_to(ROOT)}:{_hash_file(path)}"
        for path in ANSWER_SOURCE_FILES
        if path.exists()
    ]
    env = {key: value for key, value in os.environ.items() if key.startswith("RAG_")}
    if env:
        parts.append("env:" + json.dumps(env, ensure_ascii=False, sort_keys=True))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


# judge 指纹只覆盖判定语义；报告、CLI、稳定性统计的改动不应污染判定凭据。
JUDGE_LOGIC_CONSTANTS = (
    "JUDGE_PROMPT",
    "CONTRADICTION_CHECK_PROMPT",
    "REFUSAL_MARKERS",
    "STRICT_REFUSAL_MARKERS",
    "ABSENCE_REFUSAL_MARKERS",
    "CITE_PATTERN",
    "FULL_DOC_CHARS",
    "_ENUM_PREDICATE_MARKERS",
    "_ENUM_ACTION_SYNONYMS",
    "_NEGATED_CLAIM",
    "_REJECTION_MARKERS",
    "_IDENTIFIER_TOKEN",
    "_WORD_STOPWORDS",
    "_NEGATION",
)

JUDGE_LOGIC_FUNCTIONS = (
    "judge",
    "_assemble_judgement",
    "_build_fact_rows",
    "_fact_row",
    "_recovery_window",
    "_word_level_recovery",
    "_compound_parts",
    "_compound_recovery",
    "_enumeration_parts",
    "_enumeration_recovery",
    "_predicate_norm",
    "_predicate_present",
    "_negation_recovery",
    "_quote_supports_claim",
    "_parse_judge_json",
    "_escape_inner_quotes",
    "_key_facts_block",
    "_is_direct_contradiction",
    "_looks_like_contradiction",
    "_agrees_on_discontinuation",
    "_doc_blocks",
    "_norm",
    "_canon",
    "_number_in",
    "_claim_numbers",
    "_content_words",
    "_load_doc_texts",
    "_judge_excerpt",
    "_full_docs_context",
)


def _constant_source(value: object) -> str:
    # 集合/正则的默认 repr 可能受哈希随机化或对象地址影响，指纹需稳定
    if isinstance(value, re.Pattern):
        return f"{value.pattern}|{value.flags}"
    if isinstance(value, (set, frozenset)):
        return ",".join(sorted(map(str, value)))
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))
    return str(value)


@lru_cache(maxsize=1)
def _judge_code_fingerprint() -> str:
    parts = [
        f"{name}:{_constant_source(globals().get(name))}"
        for name in JUDGE_LOGIC_CONSTANTS
    ]
    for name in JUDGE_LOGIC_FUNCTIONS:
        function = globals().get(name)
        source = inspect.getsource(function) if callable(function) else ""
        parts.append(f"{name}:\n{source}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _judge_fingerprint(question: dict) -> str:
    """判定指纹：judge 代码 + 该题必答点/标准答案，key_facts 改动可被发现。"""
    payload = {
        "code": _judge_code_fingerprint(),
        "key_facts": question.get("key_facts") or [],
        "gold_answer": question.get("gold_answer", ""),
        "category": question.get("category", ""),
        "expect_refusal": question.get("expect_refusal", False),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

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
0. 如果下方给出了“必答事实点”，必须逐条核对，不得自行增删或改写：facts 的 claim 必须逐字使用给定事实点，id 依次为 K1、K2……；每条给定事实点都必须输出一行（没答到就记 missing）。只有在未给出必答事实点时，才按第 1 条自行拆分。
1. 拆事实点：把标准答案拆成必答事实点（每个数字、日期、版本号、名称、结论、条件各算一个）。括号内的计算过程、来源说明、补充说明不算独立事实点。
   - 标准答案“9,530 元（4,280 + 3,150 + 2,100）”只算一个事实点，括号内分项不是独立事实点。
   - 标准答案“缺口 4 人（编制 45 人，在岗 41 人）”只算一个事实点，括号内为计算依据。
   - 标准答案“现场差旅增加（恒远精工、华中二期现场周期延长）”只算一个事实点，括号内为补充说明。
2. status 只能是：
   - covered：助手在回答正文中明确写出该事实（仅标注引用来源不算）；
   - missing：回答正文没有写；
   - wrong：回答写了，但与标准答案或资料矛盾。
   - 判定以语义为准：允许同义改写、语序调整，也允许同一事实拆到多句或与相邻句合并；只要回答正文表达了该事实的全部要素（主体、数字、日期、结论、条件）即记 covered；只覆盖部分要素才记 missing。
   - 若一条必答事实点包含多个独立要素（由“且/并/以及/；/同时”连接），须逐要素核对：全部要素都在回答中出现才记 covered，缺少任一要素记 missing。回答把这些要素分散在不同句子时，quote 可用“……”连接多个逐字片段，每个片段都必须能在回答中逐字找到。
   - covered 和 wrong 必须给出 quote：只从最后的“助手回答”中逐字复制支撑片段，不要改写、不要加引号，也不要从检索片段或文件全文中复制。quote 无法在回答中逐字匹配时，按 missing 处理。
   - 如果回答只是复述资料原文、没有针对该事实作出表述，也不能算 covered。
3. contradictions 只记录回答与检索片段或文件全文直接冲突的陈述；evidence 必须逐字引用冲突原文。找不到直接冲突原文时，contradictions 必须为空。禁止根据制度效力层级、法律解释、“一般条款隐含”等推理判定冲突。
   回答中每条关于资料内容的否定性陈述（“未规定/未重新规定/未提及/没有/不存在/不适用”），都必须回到文件全文（含表格）核查相关章节；只要文件存在相反内容，就必须记冲突，不能因为回答总体结论正确就忽略局部错误陈述。
   对状态/密级/版本号等元数据字段的解读、效力推断，不算直接冲突；evidence 必须来自文件正文或表格，不能只是片段开头的元数据行。
   矛盾必须是两个陈述相互对立。如果回答与 evidence 表达同一结论（例如两者都说某文件已废止），这不是矛盾，不要记录。
   示例：回答称“某制度未重新规定住宿标准”，而该制度全文表格中列有该标准 → 记冲突，evidence 逐字引用该表格行。
   示例：回答称“某文件未对某情形作特别规定”，而全文确实没有该情形的专条 → 不记冲突。
4. refused：回答是否明确表示无法回答、无权访问或信息不存在，而不是给出实质答案。明确说明所问信息不存在、数量为零或公司不具备该属性（如“公司无股票代码”“未设立海外分支机构”），且没有编造具体值，也属于 refused。
5. 只核对，不要输出分数、理由或建议。

评估对象：
问题：{question}
用户角色：{role}
类别：{category}
expect_refusal：{expect_refusal}
标准答案：{gold_answer}

必答事实点：
{key_facts}

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
            response = chat_completion(
                "contradiction",
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
    for variant, canonical in (
        ("不可以", "不能"),
        ("不可", "不能"),
        ("不予", "不能"),
        ("没有", "无"),
        ("失效", "废止"),
        ("作废", "废止"),
        ("过时", "废止"),
        ("当前", "现行"),
        ("目前", "现行"),
    ):
        normalized = normalized.replace(variant, canonical)
    # 统一配置/单位的常见写法：8C ≡ 8 核，32G ≡ 32GB，2T ≡ 2TB；草稿中英混用
    for variant, canonical in (("GB", "G"), ("TB", "T"), ("核", "C"), ("draft", "草稿")):
        normalized = normalized.replace(variant, canonical)
    return normalized


def _quote_supports_claim(claim: str, quote: str, answer: str = "") -> bool:
    quote_norm = _canon(quote)
    claim_norm = _canon(claim)
    if not quote_norm or not claim_norm:
        return False
    # 括号内是计算过程/补充说明，按提示词规则不参与 token 校验
    claim_for_tokens = re.sub(r"[（(][^（）()]*[）)]", "", claim).strip() or claim
    tokens = re.findall(r"[A-Za-z0-9]+", claim_for_tokens)
    if tokens:
        scope = _canon(answer) if answer else quote_norm
        return all(_canon(token) in scope for token in tokens)
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


def _key_facts_block(key_facts: list[str]) -> str:
    if not key_facts:
        return "（未指定；请按标准答案自行拆分事实点）"
    return "\n".join(f"K{index}. {fact}" for index, fact in enumerate(key_facts, 1))


def _number_in(number: str, sentence_norm: str) -> bool:
    compact = _norm(number.replace(",", ""))
    if compact in sentence_norm:
        return True
    parts = [part for part in number.replace(",", "").split(".") if part]
    return len(parts) > 1 and all(_norm(part) in sentence_norm for part in parts)


_IDENTIFIER_TOKEN = re.compile(
    r"[A-Za-z]+[-/]?\d+(?:\.\d+)*[A-Za-z]?|[A-Za-z]+\d+(?:\.\d+)*"
)


def _claim_numbers(text: str) -> list[str]:
    """只取独立数量：V1.0、V3.2.2、HR-002、S1 等标识符里的数字不算事实数量。"""
    return re.findall(r"\d+(?:[.,]\d+)?", _IDENTIFIER_TOKEN.sub(" ", text))


_WORD_STOPWORDS = frozenset("的了和与及为在约而就等都或对将把被于该本属已其中")
_NEGATION = re.compile(r"(不|未|无|没有|非|禁止|拒绝|否定)")


@lru_cache(maxsize=8192)
def _content_words(text: str) -> tuple[str, ...]:
    import jieba

    jieba.setLogLevel(20)
    normalized = _canon(text).replace("万元", "万")
    return tuple(
        word
        for word in jieba.cut_for_search(normalized)
        if len(word) >= 2 and word not in _WORD_STOPWORDS and not word.isdigit()
    )


def _word_level_recovery(claim: str, model_answer: str) -> str | None:
    """实词覆盖兜底：吸收“转入/并入”“投入约11人月/累计投入11个人月”这类改写。

    含否定词的事实不走这条路径，避免“不愿承担/愿意承担”因实词重合被判覆盖。
    """
    if _NEGATION.search(claim):
        return None
    claim_words = _content_words(claim)
    if len(claim_words) < 3:
        return None
    answer_norm = _canon(model_answer)
    numbers = _claim_numbers(claim)
    if any(not _number_in(number, answer_norm) for number in numbers):
        return None
    answer_words = set(_content_words(model_answer))
    hits = sum(1 for word in claim_words if word in answer_words)
    if hits * 10 < len(claim_words) * 8:
        return None
    sentences = [part.strip(" -*") for part in re.split(r"[。；;\n]", model_answer) if part.strip()]
    if not sentences:
        return model_answer.strip()
    candidates = sentences
    if numbers:
        windows = sentences + [
            f"{sentences[i]} {sentences[i + 1]}" for i in range(len(sentences) - 1)
        ]
        candidates = [
            window
            for window in windows
            if all(_number_in(number, _canon(window)) for number in numbers)
        ] or sentences
    return max(
        candidates,
        key=lambda sentence: sum(
            1 for word in claim_words if word in set(_content_words(sentence))
        ),
    ).strip()


def _recovery_window(
    claim: str, model_answer: str, allow_word_fallback: bool = True
) -> str | None:
    """事实点被同义改写或拆到相邻句时，按“数字全部命中 + 二字词覆盖率”找回。"""
    fallback = (
        _word_level_recovery(claim, model_answer) if allow_word_fallback else None
    )
    numbers = _claim_numbers(claim)
    claim_norm = _canon(claim)
    bigrams = [claim_norm[i : i + 2] for i in range(len(claim_norm) - 1)]
    if not bigrams:
        return None
    sentences = [part.strip(" -*") for part in re.split(r"[。；;\n]", model_answer) if part.strip()]
    if not numbers:
        # 无数字事实常被拆到多句，按整篇回答算覆盖率；60% 可吸收“须待/需视”等同义改写，
        # 同时仍要求主体词（如客户、预算）多数命中，避免只沾一个词就判覆盖。
        if len(claim_norm) < 4:
            return fallback
        whole = _canon(model_answer)
        need = min(len(bigrams), max(2, -(-len(bigrams) * 60 // 100)))
        if sum(1 for bigram in bigrams if bigram in whole) < need:
            return fallback
        if not sentences:
            return model_answer.strip()
        return max(
            sentences,
            key=lambda sentence: sum(1 for bigram in bigrams if bigram in _canon(sentence)),
        ).strip()
    need = min(len(bigrams), max(2, (len(bigrams) + 1) // 2))
    windows = sentences + [
        f"{sentences[i]} {sentences[i + 1]}" for i in range(len(sentences) - 1)
    ]
    for window in windows:
        window_norm = _canon(window)
        if not all(_number_in(number, window_norm) for number in numbers):
            continue
        if sum(1 for bigram in bigrams if bigram in window_norm) >= need:
            return window.strip()
    # 数字被拆到间隔较远的多句时（如“2024 立项”“2025-10 终止”），按整篇覆盖率兜底
    whole = _canon(model_answer)
    if all(_number_in(number, whole) for number in numbers):
        whole_need = min(len(bigrams), max(2, -(-len(bigrams) * 75 // 100)))
        if sum(1 for bigram in bigrams if bigram in whole) >= whole_need:
            if not sentences:
                return model_answer.strip()
            return max(
                sentences,
                key=lambda sentence: sum(1 for bigram in bigrams if bigram in _canon(sentence)),
            ).strip()
    return fallback


def _compound_parts(claim: str) -> list[str]:
    """把“A，且B”“A；B”这类复合事实拆成独立子句；括号内的计算/补充说明不参与拆分。"""
    text = re.sub(r"[（(][^（）()]*[）)]", "", claim)
    parts = [
        part.strip(" 。，,；;")
        for part in re.split(r"[，,；;]\s*(?:且|并且|并|同时|以及)?|(?:并且|同时|以及)", text)
    ]
    parts = [part for part in parts if part]
    if len(parts) < 2 or any(len(part) < 4 for part in parts):
        return []
    return parts


_ENUM_PREDICATE_MARKERS = (
    "一律", "均", "必须", "应当", "不得", "不予", "禁止", "需要", "须", "应",
)
_ENUM_ACTION_SYNONYMS = (
    ("一律退回", "不能受理"),
    ("不予接收", "不能受理"),
    ("不予报销", "不能受理"),
    ("不得报销", "不能受理"),
    ("不能接收", "不能受理"),
    ("不能报销", "不能受理"),
    ("拒收", "不能受理"),
    ("拒付", "不能受理"),
    ("不接收", "不能受理"),
    ("退回", "不能受理"),
)


def _enumeration_parts(claim: str) -> tuple[list[str], str] | None:
    """拆“定额发票、手撕票一律不予受理”这类顿号列举 + 共享谓语的事实点。"""
    text = re.sub(r"[（(][^（）()]*[）)]", "", claim)
    if "、" not in text:
        return None
    segments = [part.strip(" 。，,；;") for part in text.split("、") if part.strip()]
    if len(segments) < 2:
        return None
    last = segments[-1]
    predicate = head = ""
    for marker in _ENUM_PREDICATE_MARKERS:
        index = last.find(marker)
        if index > 0:
            predicate, head = last[index:], last[:index]
            break
    if not predicate or len(head) < 2:
        return None
    items = segments[:-1] + [head]
    if any(len(item) < 2 for item in items):
        return None
    return items, predicate


def _predicate_norm(text: str) -> str:
    normalized = _canon(text)
    for variant, canonical in _ENUM_ACTION_SYNONYMS:
        normalized = normalized.replace(_canon(variant), _canon(canonical))
    return normalized


def _predicate_present(predicate: str, answer: str) -> bool:
    expected = _predicate_norm(predicate)
    actual = _predicate_norm(answer)
    if expected in actual:
        return True
    bigrams = [expected[index : index + 2] for index in range(len(expected) - 1)]
    if not bigrams:
        return False
    need = max(2, -(-len(bigrams) * 50 // 100))
    return sum(1 for bigram in bigrams if bigram in actual) >= need


def _enumeration_recovery(claim: str, model_answer: str) -> str | None:
    """逐项确认列举项，并单独确认共享谓语；避免要求谓语在每个子句窗口内逐字出现。"""
    parsed = _enumeration_parts(claim)
    if not parsed:
        return None
    items, predicate = parsed
    if not _predicate_present(predicate, model_answer):
        return None
    sentences = [
        part.strip(" -*") for part in re.split(r"[。；;\n]", model_answer) if part.strip()
    ]
    quotes = []
    for item in items:
        item_norm = _canon(item)
        quote = next(
            (sentence for sentence in sentences if item_norm in _canon(sentence)),
            None,
        )
        if not quote:
            return None
        quotes.append(quote)
    return "……".join(quotes)


_NEGATED_CLAIM = re.compile(r"^(?:不|未|无|没有|非)")
_REJECTION_MARKERS = (
    "拒绝", "否决", "反对", "禁止", "不得", "不能", "不予", "没有",
    "不再", "不支持", "不接收", "不谎称", "驳回", "取消",
)


def _negation_recovery(claim: str, model_answer: str) -> str | None:
    """否定式事实被改写成“拒绝/否决/不予……”时找回，避免误判 missing。"""
    if not _NEGATED_CLAIM.match(claim.strip()):
        return None
    words = [
        word for word in _content_words(claim)
        if word not in {"没有", "不是", "不能", "不得"}
    ]
    if len(words) < 2:
        return None
    answer_norm = _canon(model_answer)
    if sum(1 for word in words if _canon(word) in answer_norm) * 10 < len(words) * 6:
        return None
    sentences = [
        part.strip(" -*") for part in re.split(r"[。；;\n]", model_answer) if part.strip()
    ]
    candidates = [
        sentence
        for sentence in sentences
        if any(marker in sentence for marker in _REJECTION_MARKERS)
        and any(_canon(word) in _canon(sentence) for word in words)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda sentence: sum(
            1 for word in words if _canon(word) in _canon(sentence)
        ),
    )


def _compound_recovery(claim: str, model_answer: str) -> str | None:
    """逐子句找回：全部子句都在回答中有证据才算覆盖，引文用“……”连接。"""
    parts = _compound_parts(claim)
    if not parts:
        return _enumeration_recovery(claim, model_answer)
    quotes = []
    for part in parts:
        quote = _recovery_window(part, model_answer, allow_word_fallback=False)
        if not quote:
            return None
        quotes.append(quote)
    return "……".join(quotes)


def _fact_row(
    item: dict,
    fallback_id: str,
    answer_norm: str,
    model_answer: str,
    allow_recovery: bool = False,
) -> dict | None:
    claim = str(item.get("claim", "")).strip()
    status = item.get("status")
    if not claim or status not in {"covered", "missing", "wrong"}:
        return None
    quote = str(item.get("quote", "")).strip()
    if status in {"covered", "wrong"}:
        # 复合事实可被回答拆到多句：允许用“……”连接多个逐字片段，每段都须能在回答中逐字找到
        fragments = [part.strip() for part in re.split(r"…{2,}", quote) if part.strip()]
        if (
            not fragments
            or any(_norm(fragment) not in answer_norm for fragment in fragments)
            or not _quote_supports_claim(claim, "".join(fragments), model_answer)
        ):
            status, quote = "missing", ""
    if status != "covered" and allow_recovery:
        base = re.split(r"[（(]", claim, maxsplit=1)[0].strip()
        if len(_norm(base)) >= 3 and _norm(base) in answer_norm:
            claim, quote, status = base, base, "covered"
        else:
            recovered = _recovery_window(
                claim, model_answer, allow_word_fallback=(status == "missing")
            )
            if not recovered and status == "missing":
                recovered = _negation_recovery(claim, model_answer)
            if recovered:
                quote, status = recovered, "covered"
    return {"id": str(item.get("id") or fallback_id), "claim": claim, "status": status, "quote": quote}


def _build_fact_rows(question: dict, data: dict, model_answer: str) -> list[dict]:
    answer_norm = _norm(model_answer)
    raw = data.get("facts")
    raw = raw if isinstance(raw, list) else []
    key_facts = [str(f).strip() for f in (question.get("key_facts") or []) if str(f).strip()]
    if not key_facts:
        rows = []
        for index, item in enumerate(raw, 1):
            if isinstance(item, dict):
                row = _fact_row(item, f"F{index}", answer_norm, model_answer)
                if row:
                    rows.append(row)
        return rows

    matched: dict[int, dict] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        pos = None
        id_match = re.fullmatch(r"K(\d+)", str(item.get("id", "")).strip())
        if id_match and 1 <= int(id_match.group(1)) <= len(key_facts):
            pos = int(id_match.group(1))
        elif claim:
            claim_norm = _norm(claim)
            for index, key_fact in enumerate(key_facts, 1):
                key_norm = _norm(key_fact)
                if key_norm and (
                    claim_norm == key_norm
                    or claim_norm in key_norm
                    or key_norm in claim_norm
                ):
                    pos = index
                    break
        if pos and pos not in matched:
            matched[pos] = item

    rows = []
    for index, key_fact in enumerate(key_facts, 1):
        item = matched.get(index) or {
            "id": f"K{index}", "claim": key_fact, "status": "missing", "quote": ""
        }
        row = _fact_row(item, f"K{index}", answer_norm, model_answer, allow_recovery=True)
        if row:
            if row["status"] == "missing":
                compound_quote = _compound_recovery(key_fact, model_answer)
                if compound_quote:
                    row["status"] = "covered"
                    row["quote"] = compound_quote
            row["id"] = f"K{index}"
            row["claim"] = key_fact
            rows.append(row)
    return rows


def _assemble_judgement(
    question: dict,
    data: dict,
    context: str,
    full_docs: str,
    leaked: bool,
    model_answer: str = "",
) -> dict:
    facts = _build_fact_rows(question, data, model_answer)

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
    # 预期拒答题里，字段式“X：无/暂无/不存在”也是明确说明信息不存在，不当作实质回答
    if (
        not refused
        and expect_refusal
        and re.search(r"[:：]\s*(?:无|暂无|没有|不存在|未设立|未发行)[。；;，,\s]", model_answer)
    ):
        refused = True
    statuses = [fact["status"] for fact in facts]
    covered = any(status == "covered" for status in statuses)
    wrong = any(status == "wrong" for status in statuses)
    # 非拒答题中顺带一句“无权访问某关联资料”不等于拒答：必答事实点全部覆盖且无矛盾时按实质回答计分
    if (
        not expect_refusal
        and refused
        and facts
        and all(status == "covered" for status in statuses)
        and not contradictions
    ):
        refused = False

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

    faithful = (not wrong) and (not contradictions)
    has_key_facts = any(str(f).strip() for f in (question.get("key_facts") or []))
    # 未指定必答点的预期拒答题正确拒答时，标准答案本身就是“无法回答”；
    # 回答写成“未提及…数量为 0 家”等更具体表述，不再因自动拆分的事实点被判 wrong 而算不忠实。
    # 真实矛盾仍由 contradictions 检出，且带必答点的拒答题不适用此兜底。
    if expect_refusal and refused and not has_key_facts and not contradictions:
        faithful = True

    return {
        "score": score,
        "faithful": faithful,
        "refusal_ok": refused == expect_refusal,
        "permission_ok": not leaked,
        "reason": "；".join(parts),
        "facts": facts,
        "contradictions": contradictions,
        "unsupported_claims": contradictions,
        "refused": refused,
    }


def _escape_inner_quotes(raw: str) -> str:
    """模型把原文引号直接塞进 JSON 字符串时，把非结构性的 " 转义掉。"""
    out: list[str] = []
    in_str = False
    escaped = False
    for index, char in enumerate(raw):
        if not in_str:
            out.append(char)
            if char == '"':
                in_str = True
            continue
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char != '"':
            out.append(char)
            continue
        probe = index + 1
        while probe < len(raw) and raw[probe] in " \t\r\n":
            probe += 1
        if probe >= len(raw) or raw[probe] in ":}],":
            out.append(char)
            in_str = False
        else:
            out.append('\\"')
    return "".join(out)


def _parse_judge_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"judge returned non-JSON: {text[:200]}")
    raw = match.group(0)
    # 模型偶尔把字符串结尾的英文引号写成中文引号（如 `。“}`），先按结构位置修回
    repaired = re.sub(r"“(\s*[}\]])", r'"\1', raw)
    candidates = [raw] if repaired == raw else [raw, repaired]
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        for attempt in (candidate, _escape_inner_quotes(candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_error = exc
    raise last_error or ValueError("judge returned unparsable JSON")


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
        key_facts=_key_facts_block(
            [str(f).strip() for f in (question.get("key_facts") or []) if str(f).strip()]
        ),
        context=context or "（无）",
        full_docs=full_docs or "（无）",
        model_answer=model_answer,
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = chat_completion(
                "judge",
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1200,
            )
            text = response.choices[0].message.content.strip()
            data = _parse_judge_json(text)
            return _assemble_judgement(
                question, data, context or "", full_docs or "", leaked, model_answer
            )
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


def evaluate(
    limit: int | None,
    use_judge: bool,
    questions_path: Path | None = None,
    ids: set[str] | None = None,
    output_stem: str | None = None,
    results_name: str | None = None,
    write: bool = True,
    judge_repeats: int = 1,
) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_path = questions_path or QUESTIONS_PATH
    questions = [json.loads(line) for line in source_path.open(encoding="utf-8")]
    if ids:
        questions = [question for question in questions if question["id"] in ids]
    if limit:
        questions = questions[:limit]

    answer_fp = _answer_fingerprint()
    results: list[dict] = []
    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question['id']} {question['question']}")
        started = time.time()
        reset_usage()
        reset_search_stats()
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
        generation_usage = usage_snapshot()
        retrieval_stats = search_stats()

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
        runs: list[dict] = []
        reset_usage()
        if use_judge:
            runs = [
                judge(
                    question,
                    answer_text,
                    context=_judge_excerpt(response["hits"]),
                    full_docs=_full_docs_context(answer_text, gold_ids),
                    leaked=bool(leaked_ids or leaked_terms),
                )
                for _ in range(max(1, judge_repeats))
            ]
            judged = _modal_judgement(runs) if len(runs) > 1 else runs[0]
        judge_usage = usage_snapshot()

        result = {
            "id": question["id"],
            "category": question["category"],
            "role": question["role"],
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "key_facts": question.get("key_facts") or [],
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
            "table_calc": response.get("table_calc"),
            "judge": judged,
            "answer_fingerprint": answer_fp,
            "judge_fingerprint": _judge_fingerprint(question),
            "latency_s": round(time.time() - started, 2),
            "retrieval_calls": retrieval_stats["calls"],
            "retrieval_latency_s": retrieval_stats["elapsed_s"],
            "generation_usage": generation_usage,
            "judge_usage": judge_usage,
        }
        if len(runs) > 1:
            result["judge_repeat_scores"] = [run["score"] for run in runs]
        results.append(result)
        suffix = (
            f" repeats={result['judge_repeat_scores']}"
            if len(runs) > 1
            else ""
        )
        print(
            f"  retrieval_hit={result['retrieval_hit']} cites_gold={cites_gold} "
            f"score={judged['score']} faithful={judged['faithful']} "
            f"unsupported={len(judged['unsupported_claims'])} "
            f"leak={leaked_ids + leaked_terms}{suffix}"
        )

    report = _aggregate(results, str(source_path))
    if judge_repeats > 1:
        report["judge_repeats"] = judge_repeats
        report["judge_score_agreement"] = (
            sum(
                1 for result in results
                if len(set(result.get("judge_repeat_scores", []))) == 1
            ) / len(results)
            if results else None
        )
    if write:
        stem = output_stem or ("report_partial" if ids else "report")
        name = results_name or ("results_partial.jsonl" if ids else "results.jsonl")
        _write_outputs(report, REPORT_DIR, stem, name)
    return report


def _rows_for_judging(
    results_path: Path,
    limit: int | None,
    ids: set[str] | None,
    allow_stale: bool,
) -> list[dict]:
    rows = [json.loads(line) for line in results_path.open(encoding="utf-8")]
    if ids:
        rows = [row for row in rows if row["id"] in ids]
    if limit:
        rows = rows[:limit]

    current_fp = _answer_fingerprint()
    unverified = [
        row["id"] for row in rows if row.get("answer_fingerprint") != current_fp
    ]
    if unverified:
        preview = ", ".join(sorted(unverified)[:5])
        suffix = "…" if len(unverified) > 5 else ""
        print(
            f"  WARN: {len(unverified)} 条回答无法证明由当前代码/索引生成"
            f"（{preview}{suffix}）",
            flush=True,
        )
        if not allow_stale:
            raise SystemExit(
                "已存回答已过期或缺少指纹：先全量重生成；"
                "确认只验证 judge 时再加 --allow-stale。"
            )
    return rows


def _hits_for_row(row: dict) -> list[dict]:
    if not row["retrieved_ids"]:
        return []
    # 聚合题答案生成时可能放宽了候选集，复判时按实际检索条数还原上下文
    top_k = max(len(row["retrieved_ids"]), TOP_K)
    hits = hybrid_search(row["question"], role=row["role"], top_k=top_k)
    got = [hit["metadata"]["doc_id"] for hit in hits]
    if got != row["retrieved_ids"]:
        print(f"  WARN: retrieved ids differ\n    now={got}\n    old={row['retrieved_ids']}")
    return hits


def _fact_signature(judged: dict) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(fact.get("id", "")), str(fact.get("status", "")))
        for fact in judged.get("facts") or []
    )


def _fact_points(
    runs: list[dict], key_facts: list[str] | None = None
) -> list[dict]:
    """逐事实点核对多轮判分状态；无必答点时按规范化 claim 保守对齐。"""
    facts = [str(item).strip() for item in (key_facts or []) if str(item).strip()]
    points: list[dict] = []
    if facts:
        for index, claim in enumerate(facts, 1):
            points.append({"id": f"K{index}", "claim": claim, "statuses": []})
        for run in runs:
            by_id = {
                str(fact.get("id", "")): fact
                for fact in run.get("facts") or []
            }
            for point in points:
                fact = by_id.get(point["id"])
                point["statuses"].append(
                    str(fact.get("status", "absent")) if fact else "absent"
                )
    else:
        claims: dict[str, str] = {}
        for run in runs:
            for fact in run.get("facts") or []:
                key = _canon(str(fact.get("claim", "")))
                if key:
                    claims.setdefault(key, str(fact.get("claim", "")))
        for key, claim in claims.items():
            statuses = []
            for run in runs:
                match = next(
                    (
                        fact for fact in run.get("facts") or []
                        if _canon(str(fact.get("claim", ""))) == key
                    ),
                    None,
                )
                statuses.append(
                    str(match.get("status", "absent")) if match else "absent"
                )
            points.append({"id": "", "claim": claim, "statuses": statuses})
    for point in points:
        point["stable"] = len(set(point["statuses"])) == 1
    return points


def _modal_judgement(runs: list[dict]) -> dict:
    """重复判分时取众数：先按分数，再按事实点状态组合，避免单次抖动定分。"""
    valid = [run for run in runs if run.get("score") is not None]
    if not valid:
        return runs[0]
    score = Counter(run["score"] for run in valid).most_common(1)[0][0]
    candidates = [run for run in valid if run["score"] == score]
    signature = Counter(_fact_signature(run) for run in candidates).most_common(1)[0][0]
    return next(run for run in candidates if _fact_signature(run) == signature)


def rejudge(
    results_path: Path,
    limit: int | None,
    ids: set[str] | None = None,
    allow_stale: bool = False,
    judge_repeats: int = 1,
) -> dict:
    rows = _rows_for_judging(results_path, limit, ids, allow_stale)
    judge_repeats = max(1, judge_repeats)

    results: list[dict] = []
    reset_usage()
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] rejudge {row['id']}")
        hits = _hits_for_row(row)
        context = _judge_excerpt(hits)
        full_docs = _full_docs_context(row["model_answer"], row["gold_doc_ids"])
        leaked = bool(row.get("leaked_ids") or row.get("leaked_terms"))
        runs = [
            judge(row, row["model_answer"], context=context, full_docs=full_docs, leaked=leaked)
            for _ in range(judge_repeats)
        ]
        judged = _modal_judgement(runs) if len(runs) > 1 else runs[0]
        record = {**row, "judge": judged, "judge_fingerprint": _judge_fingerprint(row)}
        if len(runs) > 1:
            record["judge_repeat_scores"] = [run["score"] for run in runs]
        results.append(record)
        scores = [run["score"] for run in runs]
        suffix = f" repeats={scores}" if len(runs) > 1 else ""
        print(
            f"  score={judged['score']} faithful={judged['faithful']} "
            f"unsupported={len(judged['unsupported_claims'])}{suffix}"
        )

    report = _aggregate(results, str(results_path))
    report["rejudged_from"] = str(results_path)
    report["cost"] = _judge_cost_summary(usage_snapshot(), len(results))
    if judge_repeats > 1:
        report["judge_repeats"] = judge_repeats
        report["judge_score_agreement"] = (
            sum(
                1 for row in results
                if len(set(row.get("judge_repeat_scores", []))) == 1
            ) / len(results)
            if results else None
        )
    _write_outputs(report, REPORT_DIR, "report_rejudge", "results_rejudge.jsonl")
    return report


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _cost_summary(results: list[dict]) -> dict | None:
    generations = [
        result["generation_usage"]
        for result in results
        if result.get("generation_usage")
    ]
    judges = [result["judge_usage"] for result in results if result.get("judge_usage")]
    if not generations and not judges:
        return None
    generation = merge_usage(*generations)
    judge_usage = merge_usage(*judges)
    total = merge_usage(generation, judge_usage)
    count = len(results)
    rerank = (generation.get("by_kind") or {}).get("rerank") or {}
    return {
        "currency": "CNY",
        "generation": generation,
        "judge": judge_usage,
        "total": total,
        "per_question": {
            "count": count,
            "avg_total_cost_cny": round(total["cost_cny"] / count, 6) if count else None,
            "avg_generation_cost_cny": (
                round(generation["cost_cny"] / count, 6) if count else None
            ),
            "avg_judge_cost_cny": round(judge_usage["cost_cny"] / count, 6) if count else None,
            "avg_latency_s": _avg(
                [result["latency_s"] for result in results if result.get("latency_s") is not None]
            ),
            "avg_retrieval_latency_s": _avg(
                [
                    result["retrieval_latency_s"]
                    for result in results
                    if result.get("retrieval_latency_s") is not None
                ]
            ),
            "avg_retrieval_calls": _avg(
                [
                    result["retrieval_calls"]
                    for result in results
                    if result.get("retrieval_calls") is not None
                ]
            ),
            "avg_rerank_latency_s": (
                round(rerank.get("elapsed_s", 0.0) / count, 4) if count else None
            ),
            "rerank_calls_per_question": (
                round(rerank.get("calls", 0) / count, 3) if count else None
            ),
        },
    }


def _phase_usage(usage: dict, kind: str) -> dict:
    return (usage.get("by_kind") or {}).get(kind) or {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_cny": 0.0,
        "elapsed_s": 0.0,
    }


def _judge_cost_summary(usage: dict, count: int) -> dict:
    # 复判稳定性时会重新检索上下文，embedding 费用单列，避免混入 judge 单价
    judge = _phase_usage(usage, "judge")
    return {
        "currency": "CNY",
        "judge": judge,
        "embedding": _phase_usage(usage, "embedding"),
        "total": usage,
        "per_question": {
            "count": count,
            "avg_total_cost_cny": round(usage["cost_cny"] / count, 6) if count else None,
            "avg_judge_cost_cny": round(judge["cost_cny"] / count, 6) if count else None,
        },
    }


def _run_metrics(judged_rows: list[dict]) -> dict:
    judged = [row for row in judged_rows if row["score"] is not None]
    return {
        "avg_answer_score": (
            sum(row["score"] for row in judged) / len(judged) if judged else None
        ),
        "faithful_rate": (
            sum(1 for row in judged if row["faithful"]) / len(judged)
            if judged else None
        ),
        "refusal_accuracy": (
            sum(1 for row in judged if row["refusal_ok"]) / len(judged)
            if judged else None
        ),
        "score_distribution": dict(sorted(Counter(row["score"] for row in judged).items())),
    }


def judge_stability(
    results_path: Path,
    repeats: int = 3,
    ids: set[str] | None = None,
    limit: int | None = None,
    allow_stale: bool = False,
    output_stem: str = "report_stability",
) -> dict:
    """对同一批已存回答重复判分，量化 judge 的逐题一致性。"""
    if repeats < 2:
        raise SystemExit("--judge-stability 至少需要 2 次重复判分")
    rows = _rows_for_judging(results_path, limit, ids, allow_stale)

    records: list[dict] = []
    reset_usage()
    for index, row in enumerate(rows, 1):
        hits = _hits_for_row(row)
        context = _judge_excerpt(hits)
        full_docs = _full_docs_context(row["model_answer"], row["gold_doc_ids"])
        leaked = bool(row.get("leaked_ids") or row.get("leaked_terms"))
        runs = [
            judge(row, row["model_answer"], context=context, full_docs=full_docs, leaked=leaked)
            for _ in range(repeats)
        ]
        scores = [run["score"] for run in runs]
        faithfuls = [run["faithful"] for run in runs]
        refusals = [run["refusal_ok"] for run in runs]
        fact_signatures = {_fact_signature(run) for run in runs}
        fact_points = _fact_points(runs, row.get("key_facts"))
        majority = Counter(scores).most_common(1)[0][0]
        record = {
            "id": row["id"],
            "category": row["category"],
            "stored_score": (row.get("judge") or {}).get("score"),
            "repeat_scores": scores,
            "majority_score": majority,
            "score_stable": len(set(scores)) == 1,
            "facts_stable": len(fact_signatures) == 1,
            "fact_points_stable": all(point["stable"] for point in fact_points),
            "has_key_facts": bool(row.get("key_facts")),
            "faithful_stable": len(set(faithfuls)) == 1,
            "refusal_stable": len(set(refusals)) == 1,
            "fact_points": fact_points,
            "repeat_reasons": [run["reason"] for run in runs],
            "runs": runs,
        }
        records.append(record)
        print(
            f"[{index}/{len(rows)}] {row['id']} stored={record['stored_score']} "
            f"repeats={scores} facts_stable={record['facts_stable']} "
            f"fact_points_stable={record['fact_points_stable']}",
            flush=True,
        )

    def _rate(key: str) -> float | None:
        return sum(1 for record in records if record[key]) / len(records) if records else None

    all_points = [
        (record["id"], point)
        for record in records
        for point in record["fact_points"]
    ]
    stable_points = sum(1 for _, point in all_points if point["stable"])
    fact_point_agreement = stable_points / len(all_points) if all_points else None

    def _point_agreement(subset: list[dict]) -> dict:
        points = [point for record in subset for point in record["fact_points"]]
        stable = sum(1 for point in points if point["stable"])
        return {
            "questions": len(subset),
            "stable": stable,
            "total": len(points),
            "rate": stable / len(points) if points else None,
        }

    score_stable_facts_unstable = [
        record["id"]
        for record in records
        if record["score_stable"] and not record["fact_points_stable"]
    ]

    run_metrics = [
        _run_metrics([record["runs"][run_index] for record in records])
        for run_index in range(repeats)
    ]
    majority_metrics = _run_metrics(
        [
            {**record["runs"][0], "score": record["majority_score"]}
            for record in records
        ]
    )
    unstable = [
        record for record in records
        if not (
            record["score_stable"]
            and record["facts_stable"]
            and record["faithful_stable"]
            and record["refusal_stable"]
        )
    ]
    changed = [
        record for record in records
        if record["stored_score"] != record["majority_score"]
    ]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_file": str(results_path),
        "question_count": len(records),
        "repeats": repeats,
        "agreement": {
            "score": _rate("score_stable"),
            "facts": _rate("facts_stable"),
            "fact_points": fact_point_agreement,
            "faithful": _rate("faithful_stable"),
            "refusal": _rate("refusal_stable"),
            "all": (
                sum(
                    1 for record in records
                    if record["score_stable"]
                    and record["facts_stable"]
                    and record["faithful_stable"]
                    and record["refusal_stable"]
                ) / len(records)
                if records else None
            ),
        },
        "fact_point_counts": {
            "stable": stable_points,
            "total": len(all_points),
        },
        "fact_point_by_type": {
            "with_key_facts": _point_agreement(
                [record for record in records if record["has_key_facts"]]
            ),
            "auto_split": _point_agreement(
                [record for record in records if not record["has_key_facts"]]
            ),
        },
        "score_stable_facts_unstable": score_stable_facts_unstable,
        "unstable_fact_points": [
            {
                "id": record_id,
                "fact_id": point["id"],
                "claim": point["claim"],
                "statuses": point["statuses"],
            }
            for record_id, point in all_points
            if not point["stable"]
        ],
        "run_metrics": run_metrics,
        "majority_metrics": majority_metrics,
        "stored_metrics": _run_metrics(
            [row.get("judge") or {"score": None} for row in rows]
        ),
        "unstable": [
            {
                key: record[key]
                for key in (
                    "id", "category", "stored_score", "repeat_scores",
                    "score_stable", "facts_stable", "faithful_stable", "refusal_stable",
                )
            }
            for record in unstable
        ],
        "changed_vs_stored": [
            {
                "id": record["id"],
                "stored": record["stored_score"],
                "majority": record["majority_score"],
                "repeats": record["repeat_scores"],
            }
            for record in changed
        ],
        "cost": _judge_cost_summary(usage_snapshot(), len(records)),
        "records": records,
    }
    _write_stability_outputs(report, output_stem)
    return report


def _fmt_rate(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def _write_stability_outputs(report: dict, stem: str = "report_stability") -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    agreement = report["agreement"]
    lines = [
        "# Judge 稳定性报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 结果文件：{report['results_file']}",
        f"- 题数：{report['question_count']}，每题重复判分：{report['repeats']} 次",
        f"- 分数完全一致：{_fmt_rate(agreement['score'])}",
        f"- 整题事实点签名完全一致：{_fmt_rate(agreement['facts'])}",
        f"- 逐事实点状态一致率：{_fmt_rate(agreement['fact_points'])}"
        f"（{report['fact_point_counts']['stable']}/{report['fact_point_counts']['total']}）",
        f"- 逐事实点一致率（有必答点）：{_fmt_rate(report['fact_point_by_type']['with_key_facts']['rate'])}"
        f"（{report['fact_point_by_type']['with_key_facts']['stable']}/"
        f"{report['fact_point_by_type']['with_key_facts']['total']}）",
        f"- 逐事实点一致率（judge 自拆点）：{_fmt_rate(report['fact_point_by_type']['auto_split']['rate'])}"
        f"（{report['fact_point_by_type']['auto_split']['stable']}/"
        f"{report['fact_point_by_type']['auto_split']['total']}，措辞差异保守计入不一致）",
        f"- 忠实判定完全一致：{_fmt_rate(agreement['faithful'])}",
        f"- 拒答判定完全一致：{_fmt_rate(agreement['refusal'])}",
        f"- 四项全部一致：{_fmt_rate(agreement['all'])}",
        *(
            [
                f"- judge 费用：约 {report['cost']['judge']['cost_cny']:.4f} 元"
                f"（{report['cost']['judge']['calls']} 次调用，"
                f"{report['cost']['judge']['total_tokens']} tokens，"
                f"均摊 {report['cost']['per_question']['avg_judge_cost_cny']:.6f} 元/题）",
                f"- 复判检索 embedding：约 {report['cost']['embedding']['cost_cny']:.4f} 元"
                f"（{report['cost']['embedding']['calls']} 次调用）",
            ]
            if report.get("cost")
            else []
        ),
        "",
        "## 各轮聚合指标",
        "",
        "| 轮次 | 平均分 | 忠实率 | 拒答正确率 | 得分分布 |",
        "|---|---|---|---|---|",
    ]
    for index, metrics in enumerate(report["run_metrics"], 1):
        lines.append(
            f"| 第 {index} 轮 | {_fmt_float(metrics['avg_answer_score'])} | "
            f"{_fmt_float(metrics['faithful_rate'])} | "
            f"{_fmt_float(metrics['refusal_accuracy'])} | "
            f"{metrics['score_distribution']} |"
        )
    majority = report["majority_metrics"]
    lines.append(
        f"| 多数票 | {_fmt_float(majority['avg_answer_score'])} | "
        f"{_fmt_float(majority['faithful_rate'])} | "
        f"{_fmt_float(majority['refusal_accuracy'])} | "
        f"{majority['score_distribution']} |"
    )
    stored = report["stored_metrics"]
    lines.append(
        f"| 已存判定 | {_fmt_float(stored['avg_answer_score'])} | "
        f"{_fmt_float(stored['faithful_rate'])} | "
        f"{_fmt_float(stored['refusal_accuracy'])} | "
        f"{stored['score_distribution']} |"
    )
    lines.extend([
        "",
        "## 不一致题目",
        "",
        "| ID | 类别 | 已存分 | 重复分 | 分数稳定 | 事实点稳定 | 忠实稳定 | 拒答稳定 |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for record in report["unstable"]:
        lines.append(
            f"| {record['id']} | {record['category']} | {record['stored_score']} | "
            f"{record['repeat_scores']} | "
            f"{'Y' if record['score_stable'] else 'N'} | "
            f"{'Y' if record['facts_stable'] else 'N'} | "
            f"{'Y' if record['faithful_stable'] else 'N'} | "
            f"{'Y' if record['refusal_stable'] else 'N'} |"
        )
    if not report["unstable"]:
        lines.append("| - | - | - | - | - | - | - | - |")
    lines.extend([
        "",
        "## 分数稳定但事实点归因不一致",
        "",
        f"- 题目：{', '.join(report['score_stable_facts_unstable']) or '无'}",
        "",
        "## 事实点级不一致",
        "",
        "| 题目 | 事实点 | 各轮状态 |",
        "|---|---|---|",
    ])
    for point in report["unstable_fact_points"]:
        lines.append(
            f"| {point['id']} | {point['claim']} | {' / '.join(point['statuses'])} |"
        )
    if not report["unstable_fact_points"]:
        lines.append("| - | - | - |")
    (REPORT_DIR / f"{stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {key: report[key] for key in ("repeats", "agreement", "majority_metrics")},
        ensure_ascii=False,
        indent=2,
    ))


def _gold_coverage(result: dict) -> tuple[int, int]:
    """返回（命中的 gold 文档数，gold 文档总数）；无 gold 时返回 (0, 0)。"""
    gold = set(result.get("gold_doc_ids") or [])
    if not gold:
        return 0, 0
    return len(set(result.get("retrieved_ids") or []) & gold), len(gold)


def _coverage_metrics(results: list[dict]) -> dict:
    pairs = [_gold_coverage(result) for result in results]
    pairs = [(hit, total) for hit, total in pairs if total]
    gold_total = sum(total for _, total in pairs)
    hit_total = sum(hit for hit, _ in pairs)
    return {
        # micro：全部 gold 文档中被召回的比例，多跳题的部分缺口不会被“任一命中”掩盖
        "gold_doc_coverage": hit_total / gold_total if gold_total else None,
        "gold_doc_full_rate": (
            sum(1 for hit, total in pairs if hit == total) / len(pairs) if pairs else None
        ),
    }


def _aggregate(results: list[dict], source: str) -> dict:
    gold_results = [r for r in results if r["gold_doc_ids"]]
    retrieval_recall = (
        sum(1 for r in gold_results if r["retrieval_hit"]) / len(gold_results)
        if gold_results else 0.0
    )
    coverage = _coverage_metrics(results)
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
            **_coverage_metrics(items),
            "avg_score": sum(scores) / len(scores) if scores else None,
            "routes": dict(sorted(Counter(item["route"] for item in items).items())),
        }

    answer_fingerprints = sorted(
        {
            result["answer_fingerprint"]
            for result in results
            if result.get("answer_fingerprint")
        }
    )
    judge_fingerprints = sorted(
        {
            result["judge_fingerprint"]
            for result in results
            if result.get("judge_fingerprint")
        }
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "questions_file": source,
        "question_count": len(results),
        "fingerprints": {
            "answer": answer_fingerprints,
            "judge": judge_fingerprints,
            "missing_answer": sum(
                1 for result in results if not result.get("answer_fingerprint")
            ),
            "missing_judge": sum(
                1 for result in results if not result.get("judge_fingerprint")
            ),
        },
        "metrics": {
            "retrieval_recall": retrieval_recall,
            **coverage,
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
        "cost": _cost_summary(results),
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


def _fmt_float(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def _write_markdown(report: dict, path: Path) -> None:
    metrics = report["metrics"]
    fingerprints = report.get("fingerprints") or {}
    lines = [
        "# 评估报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 题目数：{report['question_count']}",
        f"- 检索 Recall：{metrics['retrieval_recall']:.3f}",
        f"- Gold 文档覆盖率（micro）：{_fmt_float(metrics.get('gold_doc_coverage'))}",
        f"- Gold 文档全命中率：{_fmt_float(metrics.get('gold_doc_full_rate'))}",
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
        *(
            [
                f"- 生成 token：{cost['generation']['total_tokens']}（{cost['generation']['calls']} 次调用，"
                f"约 {cost['generation']['cost_cny']:.4f} 元）",
                f"- judge token：{cost['judge']['total_tokens']}（{cost['judge']['calls']} 次调用，"
                f"约 {cost['judge']['cost_cny']:.4f} 元）",
                f"- 预估总费用：约 {cost['total']['cost_cny']:.4f} 元"
                f"（均摊 {cost['per_question']['avg_total_cost_cny']:.6f} 元/题）",
                f"- 平均延迟：{_fmt_float(cost['per_question']['avg_latency_s'])} s"
                f"（检索 {_fmt_float(cost['per_question']['avg_retrieval_latency_s'])} s，"
                f"rerank {_fmt_float(cost['per_question']['avg_rerank_latency_s'])} s/题）",
            ]
            if (cost := report.get("cost"))
            else []
        ),
        f"- 回答指纹：{', '.join(fingerprints.get('answer') or []) or '-'}",
        f"- 判定指纹：{', '.join(fingerprints.get('judge') or []) or '-'}",
        *(
            ["- **注意**：结果包含多代回答或缺少生成指纹，不是纯净基线。"]
            if len(fingerprints.get("answer") or []) > 1
            or fingerprints.get("missing_answer")
            else []
        ),
        "",
        "## 分类指标",
        "",
        "| 类别 | 题数 | 检索 Recall | Gold 覆盖率 | 平均分 | 路由 |",
        "|---|---|---|---|---|---|",
    ]
    for category, metric in report["by_category"].items():
        recall = metric["retrieval_recall"]
        avg = metric["avg_score"]
        recall_text = f"{recall:.3f}" if recall is not None else "-"
        coverage_text = _fmt_float(metric.get("gold_doc_coverage"))
        avg_text = f"{avg:.2f}" if avg is not None else "-"
        lines.append(
            f"| {category} | {metric['count']} | {recall_text} | {coverage_text} | "
            f"{avg_text} | {metric['routes']} |"
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


def merge_partial(partial_path: Path, baseline_path: Path, ids: set[str]) -> None:
    if not ids:
        raise SystemExit("--merge-partial 需要 --ids 明确指定要并入的题号")
    if not partial_path.exists():
        raise SystemExit(f"找不到 partial 文件：{partial_path}")
    partial = {
        row["id"]: row
        for row in (
            json.loads(line) for line in partial_path.open(encoding="utf-8")
        )
    }
    missing = sorted(ids - partial.keys())
    if missing:
        raise SystemExit(f"partial 文件缺少题号：{missing}")

    baseline = [json.loads(line) for line in baseline_path.open(encoding="utf-8")]
    baseline_by_id = {row["id"]: row for row in baseline}
    unknown = sorted(ids - baseline_by_id.keys())
    if unknown:
        raise SystemExit(f"基线中不存在题号：{unknown}")

    current_fp = _answer_fingerprint()
    merged_at = datetime.now().isoformat(timespec="seconds")
    for qid in sorted(ids):
        row = partial[qid]
        if row.get("answer_fingerprint") != current_fp:
            print(
                f"  WARN: {qid} 的回答也不是当前代码生成"
                f"（{row.get('answer_fingerprint') or '无指纹'} != {current_fp}）"
            )
        baseline_by_id[qid] = {
            **row,
            "provenance": "partial_regenerated",
            "merged_at": merged_at,
        }

    baseline_path.write_text(
        "".join(
            json.dumps(baseline_by_id[row["id"]], ensure_ascii=False) + "\n"
            for row in baseline
        ),
        encoding="utf-8",
    )
    fingerprints = sorted(
        {row.get("answer_fingerprint") or "unknown" for row in baseline_by_id.values()}
    )
    print(f"已并入 {len(ids)} 题到 {baseline_path}：{sorted(ids)}")
    print(f"基线现有 {len(fingerprints)} 种回答指纹：{fingerprints}")
    if len(fingerprints) > 1 or "unknown" in fingerprints:
        print("  WARN: 合并后是混合基线，不能当作全量重生成结果引用。")


def _select_drift_sample(rows: list[dict], size: int) -> list[str]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["id"]):
        by_category[row["category"]].append(row["id"])
    selected: list[str] = []
    categories = sorted(by_category)
    while len(selected) < size and any(by_category.values()):
        for category in categories:
            if by_category[category] and len(selected) < size:
                selected.append(by_category[category].pop(0))
    return selected


def drift_sample(baseline_path: Path, size: int) -> dict:
    baseline = [json.loads(line) for line in baseline_path.open(encoding="utf-8")]
    sample = _select_drift_sample(baseline, size)
    if not sample:
        raise SystemExit("基线为空，无法抽样")
    print(f"漂移抽样 {len(sample)} 题：{', '.join(sample)}")

    regenerated = evaluate(limit=None, use_judge=True, ids=set(sample), write=False)
    before = {row["id"]: row for row in baseline}
    after = {row["id"]: row for row in regenerated["results"]}
    rows = []
    for qid in sample:
        old = before[qid]
        new = after[qid]
        old_judge = old.get("judge") or {}
        new_judge = new.get("judge") or {}
        old_facts = {
            fact.get("id"): fact.get("status")
            for fact in old_judge.get("facts") or []
        }
        new_facts = {
            fact.get("id"): fact.get("status")
            for fact in new_judge.get("facts") or []
        }
        rows.append({
            "id": qid,
            "category": old["category"],
            "retrieval_changed": old.get("retrieved_ids") != new.get("retrieved_ids"),
            "route_changed": old.get("route") != new.get("route"),
            "answer_changed": _norm(old.get("model_answer", ""))
            != _norm(new.get("model_answer", "")),
            "score_before": old_judge.get("score"),
            "score_after": new_judge.get("score"),
            "facts_changed": old_facts != new_facts,
        })

    current_fp = _answer_fingerprint()
    baseline_fps = {
        row.get("answer_fingerprint")
        for row in baseline
        if row.get("answer_fingerprint")
    }
    fingerprint_changed = bool(baseline_fps) and current_fp not in baseline_fps
    signals = {
        "fingerprint_changed": fingerprint_changed,
        "retrieval_changed": sum(1 for row in rows if row["retrieval_changed"]),
        "route_changed": sum(1 for row in rows if row["route_changed"]),
        "answer_changed": sum(1 for row in rows if row["answer_changed"]),
        "score_changed": sum(
            1 for row in rows if row["score_before"] != row["score_after"]
        ),
        "facts_changed": sum(1 for row in rows if row["facts_changed"]),
    }
    if fingerprint_changed:
        verdict = "full_generation"
        reason = "基线回答指纹与当前代码/索引不一致"
    elif signals["retrieval_changed"] or signals["route_changed"]:
        verdict = "full_generation"
        reason = "检索或路由已变化，旧回答不能继续作为基线"
    elif signals["score_changed"] or signals["facts_changed"]:
        verdict = "inspect"
        reason = "检索稳定但判定变化，先确认是回答变化还是 judge 方差"
    elif signals["answer_changed"]:
        verdict = "rejudge_ok"
        reason = "仅措辞变化，检索与判定稳定，可全量复判"
    else:
        verdict = "rejudge_ok"
        reason = "样本无变化，可全量复判"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": str(baseline_path),
        "sample_size": len(sample),
        "ids": sample,
        "answer_fingerprint": current_fp,
        "signals": signals,
        "verdict": verdict,
        "reason": reason,
        "rows": rows,
    }
    _write_drift_outputs(report)
    return report


def _write_drift_outputs(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report_drift.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 漂移抽样报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 基线：{report['baseline']}",
        f"- 抽样题数：{report['sample_size']}（{', '.join(report['ids'])}）",
        f"- 当前回答指纹：{report['answer_fingerprint']}",
        f"- 结论：**{report['verdict']}** — {report['reason']}",
        f"- 信号：{report['signals']}",
        "",
        "| ID | 类别 | 检索变化 | 路由变化 | 回答变化 | 分数 | 事实点变化 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['id']} | {row['category']} | "
            f"{'Y' if row['retrieval_changed'] else 'N'} | "
            f"{'Y' if row['route_changed'] else 'N'} | "
            f"{'Y' if row['answer_changed'] else 'N'} | "
            f"{row['score_before']}→{row['score_after']} | "
            f"{'Y' if row['facts_changed'] else 'N'} |"
        )
    (REPORT_DIR / "report_drift.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["signals"], ensure_ascii=False, indent=2))
    print(f"结论：{report['verdict']} — {report['reason']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--questions",
        default=str(QUESTIONS_PATH),
        help="题目文件（默认 v3）",
    )
    parser.add_argument(
        "--report-stem",
        default=None,
        help="报告文件名前缀，默认 report（全量）或 report_partial（--ids）",
    )
    parser.add_argument(
        "--results-name",
        default=None,
        help="结果文件名，默认 results.jsonl 或 results_partial.jsonl",
    )
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
        help="只处理指定题号，逗号分隔（配合 --rejudge / --merge-partial）",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="复判时允许使用缺少当前回答指纹的旧结果",
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=1,
        help="每题重复判分次数，取众数写入基线（建议 3；默认 1）",
    )
    parser.add_argument(
        "--merge-partial",
        action="store_true",
        help="把 --partial 中 --ids 指定的题号显式并入 --results 基线",
    )
    parser.add_argument(
        "--partial",
        default=str(REPORT_DIR / "results_partial.jsonl"),
        help="--merge-partial 读取的 partial 结果文件",
    )
    parser.add_argument(
        "--judge-stability",
        nargs="?",
        const=3,
        type=int,
        default=None,
        help="复用已存回答重复判分 N 次，量化 judge 稳定性（默认 3）",
    )
    parser.add_argument(
        "--stability-stem",
        default=None,
        help="稳定性报告前缀，默认 report_stability；--ids 子集默认 report_stability_partial",
    )
    parser.add_argument(
        "--drift-sample",
        nargs="?",
        const=12,
        type=int,
        default=None,
        help="分层抽样重生成并比较漂移，默认 12 题",
    )
    args = parser.parse_args()
    ids = {item.strip() for item in args.ids.split(",") if item.strip()} or None
    if args.merge_partial:
        if not ids:
            parser.error("--merge-partial 需要 --ids 指定题号")
        merge_partial(Path(args.partial), Path(args.results), ids)
    elif args.drift_sample is not None:
        if args.rejudge:
            parser.error("--drift-sample 不能与 --rejudge 同时使用")
        drift_sample(Path(args.results), args.drift_sample)
    elif args.judge_stability:
        judge_stability(
            Path(args.results),
            args.judge_stability,
            ids,
            args.limit,
            allow_stale=args.allow_stale,
            output_stem=(
                args.stability_stem
                or ("report_stability_partial" if ids else "report_stability")
            ),
        )
    elif args.rejudge:
        rejudge(
            Path(args.results),
            args.limit,
            ids,
            allow_stale=args.allow_stale,
            judge_repeats=args.judge_repeats,
        )
    else:
        evaluate(
            limit=args.limit,
            use_judge=not args.no_judge,
            questions_path=Path(args.questions),
            ids=ids,
            output_stem=args.report_stem,
            results_name=args.results_name,
            judge_repeats=args.judge_repeats,
        )
