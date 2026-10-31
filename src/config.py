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

ROLES = ("普通员工", "HR", "财务", "销售", "IT", "产品", "管理层")

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
