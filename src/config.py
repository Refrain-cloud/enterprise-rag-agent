from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "kb"
MANIFEST_PATH = DATA_DIR / "_manifest.json"
BUILD_DIR = ROOT / "build"
CORPUS_DIR = BUILD_DIR / "corpus"
INDEX_DIR = BUILD_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
OCR_DIR = BUILD_DIR / "ocr"

DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-v3")
OCR_ENABLED = os.getenv("RAG_OCR", "1") not in {"0", "false", "False"}
OCR_MODEL = os.getenv("RAG_OCR_MODEL", "qwen3-vl-plus")
GENERATION_SEED = int(os.getenv("RAG_SEED", "42"))
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1024"))
EMBED_BATCH = 10

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("RAG_TOP_K", "12"))
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "40"))
# 聚合题需要更宽的候选集，避免表格文档被叙述型文档挤出 top-K
AGG_TOP_K = int(os.getenv("RAG_AGG_TOP_K", "16"))
# 受限最优命中的余弦距离比可访问最优命中近多少才判定为“问的就是机密内容”
GUARD_DISTANCE_GAP = float(os.getenv("RAG_GUARD_DISTANCE_GAP", "0.03"))

RERANK_ENABLED = os.getenv("RAG_RERANK", "0") not in {"0", "false", "False"}
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "gte-rerank-v2")
# 只重排已召回的 top-k，避免把表格文档挤进上下文后改变工具路由
RERANK_CANDIDATE_K = int(os.getenv("RAG_RERANK_CANDIDATE_K", str(TOP_K)))
RERANK_BASE_URL = os.getenv(
    "RAG_RERANK_BASE_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
)

ROLES = ("普通员工", "HR", "财务", "销售", "IT", "产品", "管理层")

# 结构化字段重写默认关闭：v5 全量验证显示重写会把多跳题改写成无关表单字段
STRUCTURED_FIELD_RETRY = os.getenv("RAG_FIELD_RETRY", "0") not in {"0", "false", "False"}
TABLE_TOOL_ENABLED = os.getenv("RAG_TABLE_TOOL", "1") not in {"0", "false", "False"}
TABLE_FORMATS = {"csv", "xlsx"}
AGG_KEYWORDS = (
    "合计", "总计", "总共", "一共", "几笔", "几单",
    "最高", "最低", "最多", "最少", "平均", "统计", "汇总", "占比",
    "工时", "进度",
)
TABLE_MAX_DOCS = int(os.getenv("RAG_TABLE_MAX_DOCS", "3"))
TABLE_MAX_ROWS = int(os.getenv("RAG_TABLE_MAX_ROWS", "200"))
TABLE_HEADER_SCAN_ROWS = 10
TABLE_SAMPLE_ROWS = 3
TABLE_RESULT_PROMPT_ROWS = 20
TABLE_SQL_MAX_TOKENS = 400

# 生成预算：截断检测命中后逐级放宽，避免长答案停在半句
GENERATION_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "900"))
GENERATION_MAX_TOKENS_HEADER = int(os.getenv("RAG_MAX_TOKENS_HEADER", "1200"))
GENERATION_MAX_TOKENS_CEILING = int(os.getenv("RAG_MAX_TOKENS_CEILING", "2400"))

# 零命中时允许一次查询改写后重检索；正常题不触发额外模型调用
REWRITE_ON_EMPTY = os.getenv("RAG_REWRITE_ON_EMPTY", "1") not in {"0", "false", "False"}
REWRITE_MAX_TOKENS = int(os.getenv("RAG_REWRITE_MAX_TOKENS", "80"))

# 服务层并发与超时
API_MAX_CONCURRENCY = int(os.getenv("RAG_API_MAX_CONCURRENCY", "4"))
API_QUEUE_TIMEOUT_S = float(os.getenv("RAG_API_QUEUE_TIMEOUT_S", "5"))
API_REQUEST_TIMEOUT_S = float(os.getenv("RAG_API_REQUEST_TIMEOUT_S", "120"))

# 多轮会话：内存存储，服务重启即清空
SESSION_TTL_S = float(os.getenv("RAG_SESSION_TTL_S", "1800"))
SESSION_MAX_TURNS = int(os.getenv("RAG_SESSION_MAX_TURNS", "10"))
SESSION_MAX_TOKENS = int(os.getenv("RAG_SESSION_MAX_TOKENS", "100000"))
SESSION_MAX_COST_CNY = float(os.getenv("RAG_SESSION_MAX_COST_CNY", "0.5"))
SESSION_MAX_COUNT = int(os.getenv("RAG_SESSION_MAX_COUNT", "500"))
SESSION_HISTORY_TURNS = int(os.getenv("RAG_SESSION_HISTORY_TURNS", "3"))
SESSION_HISTORY_CHARS = int(os.getenv("RAG_SESSION_HISTORY_CHARS", "1200"))
SESSION_ANSWER_CHARS = int(os.getenv("RAG_SESSION_ANSWER_CHARS", "400"))
FOLLOWUP_REWRITE = os.getenv("RAG_FOLLOWUP_REWRITE", "1") not in {"0", "false", "False"}
FOLLOWUP_MAX_TOKENS = int(os.getenv("RAG_FOLLOWUP_MAX_TOKENS", "80"))

# 流式输出
STREAM_INCLUDE_USAGE = os.getenv("RAG_STREAM_INCLUDE_USAGE", "1") not in {"0", "false", "False"}
STREAM_HEARTBEAT_S = float(os.getenv("RAG_STREAM_HEARTBEAT_S", "15"))


def get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
    return key


def get_client():
    from openai import OpenAI

    return OpenAI(api_key=get_api_key(), base_url=DASHSCOPE_BASE_URL, timeout=60)


def ensure_build_dirs() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
