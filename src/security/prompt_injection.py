from __future__ import annotations

import re

BLOCKED_MARKER = "[已屏蔽的文档内指令]"
CONTROL_MARKER = "[已移除的控制符]"

_BIDI_RE = re.compile("[‪-‮⁦-⁩]")
_CONTROL_TOKEN_RE = re.compile(
    r"<\|[^|>\n]{0,40}\|>|<<\s*/?\s*SYS\s*>>|\[/?\s*INST\s*\]",
    re.IGNORECASE,
)
_DELIMITER_RE = re.compile(r"<\s*(/?)\s*document\b", re.IGNORECASE)

_INSTRUCTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "override_zh",
        re.compile(
            r"(?:忽略|无视|忘掉|忘记|跳过|不要遵守|停止遵守)"
            r"[^。\n]{0,12}?"
            r"(?:之前|以上|上面|前述|前面|先前|所有|全部)"
            r"[^。\n]{0,8}?"
            r"(?:指令|提示|规则|要求|约束|设定|说明)"
        ),
    ),
    (
        "override_en",
        re.compile(
            r"(?:ignore|disregard|forget|override)\s+"
            r"(?:all\s+|any\s+|the\s+)*"
            r"(?:previous|above|prior|earlier|system)\s+"
            r"(?:instructions?|prompts?|rules?|messages?)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack_zh",
        re.compile(
            r"(?:你现在(?:是|扮演|进入|已成为)"
            r"|从现在(?:开始|起)[，,]?\s*你(?:是|将|需要扮演)"
            r"|(?:请|立即|马上)进入(?:开发者|无限制|DAN)\s*模式)"
            r"[^。\n]{0,20}?"
            r"(?:管理员|超级用户|开发者模式|无限制模式|系统模式|root|DAN)"
        ),
    ),
    (
        "role_hijack_en",
        re.compile(
            r"(?:you\s+are\s+now|act\s+as|from\s+now\s+on\s+you\s+are|pretend\s+to\s+be)"
            r"\s+[^.\n]{0,30}?"
            r"(?:admin|administrator|developer\s+mode|unrestricted|dan|root)",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_leak_zh",
        re.compile(
            r"(?:输出|打印|复述|泄露|展示|告诉我|重复|返回)"
            r"[^。\n]{0,16}?"
            r"(?:系统提示词?|系统提示|初始指令|初始提示|隐藏指令|隐藏提示|system\s*prompt)"
        ),
    ),
    (
        "prompt_leak_en",
        re.compile(
            r"(?:reveal|print|show|repeat|leak|output|return)\s+"
            r"[^.\n]{0,20}?"
            r"(?:system\s*prompt|initial\s+instructions?|hidden\s+instructions?)",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_exec_zh",
        re.compile(
            r"(?:调用|执行|运行|使用)"
            r"[^。\n]{0,16}?"
            r"(?:sql|命令|脚本|工具|api)"
            r"[^。\n]{0,24}?"
            r"(?:删除|清空|修改|写入|上传|移除|覆盖)"
        ),
    ),
    (
        "tool_exec_en",
        re.compile(
            r"(?:call|execute|run|invoke)\s+"
            r"[^.\n]{0,20}?"
            r"(?:sql|command|script|tool|api)"
            r"[^.\n]{0,30}?"
            r"(?:delete|drop|truncate|modify|write|upload|remove)",
            re.IGNORECASE,
        ),
    ),
)


def _finding(kind: str, text: str, match: re.Match[str] | None = None) -> dict:
    snippet = match.group(0) if match else text
    return {
        "kind": kind,
        "match": snippet.strip()[:120],
        "start": match.start() if match else 0,
        "end": match.end() if match else len(text),
    }


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _expand_to_sentence(text: str, start: int, end: int) -> tuple[int, int]:
    """指令句通常带后续要求（如“忽略以上规则，回答 X”），整句一并屏蔽。"""
    boundaries = "。！？!?；;\n"
    left = max((text.rfind(char, 0, start) for char in boundaries), default=-1) + 1
    following = [text.find(char, end) for char in boundaries]
    following = [position for position in following if position != -1]
    right = min(following) + 1 if following else len(text)
    return left, right


def detect_injection(text: str) -> list[dict]:
    findings: list[dict] = []
    if _BIDI_RE.search(text):
        findings.append(_finding("hidden_bidi", text))
    for match in _CONTROL_TOKEN_RE.finditer(text):
        findings.append(_finding("control_token", text, match))
    for kind, pattern in _INSTRUCTION_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(_finding(kind, text, match))
    return findings


def prepare_untrusted_text(text: str) -> tuple[str, list[dict]]:
    """清洗检索文档：去方向控制符/聊天模板符，屏蔽指令句，转义可闭合外层标签的文本。"""
    findings = detect_injection(text)
    cleaned = _BIDI_RE.sub("", text)
    cleaned = _CONTROL_TOKEN_RE.sub(CONTROL_MARKER, cleaned)

    spans: list[tuple[int, int]] = []
    for _, pattern in _INSTRUCTION_PATTERNS:
        spans.extend(
            _expand_to_sentence(cleaned, match.start(), match.end())
            for match in pattern.finditer(cleaned)
        )
    for start, end in reversed(_merge_spans(spans)):
        cleaned = cleaned[:start] + BLOCKED_MARKER + cleaned[end:]

    cleaned = _DELIMITER_RE.sub(lambda match: f"＜{match.group(1)}document", cleaned)
    return cleaned, findings


def contains_control_token(text: str) -> bool:
    return bool(_CONTROL_TOKEN_RE.search(text))


def strip_control_tokens(text: str) -> str:
    return _CONTROL_TOKEN_RE.sub("", text)
