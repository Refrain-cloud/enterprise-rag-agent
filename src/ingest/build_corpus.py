from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.config import CORPUS_DIR, DATA_DIR, MANIFEST_PATH, OCR_ENABLED, ensure_build_dirs
from src.ingest.parsers import parse_file

FOLDER_DEPTS = {
    "finance": "FN",
    "hr": "HR",
    "it": "IT",
    "policy": "PO",
    "product": "PD",
    "project": "IM",
    "sales": "SL",
}


def _load_manifest() -> dict[str, dict]:
    records = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {item["path"]: item for item in records}


def _doc_record(path: Path, meta: dict, use_ocr: bool = True) -> dict:
    rel = path.relative_to(DATA_DIR).as_posix()
    doc_id = meta.get("doc_id") or path.stem.split("_", 1)[0]
    result = parse_file(path, doc_id=doc_id, use_ocr=use_ocr)
    folder = rel.split("/", 1)[0]
    if result.text.strip():
        status = "ok"
    elif result.scanned:
        status = "scanned_no_text"
    elif result.error:
        status = "error"
    else:
        status = "empty_text"
    return {
        "doc_id": doc_id,
        "title": meta.get("title") or path.stem,
        "path": rel,
        "format": path.suffix.lower().lstrip("."),
        "dept": meta.get("dept") or FOLDER_DEPTS.get(folder, ""),
        "scope": meta.get("scope") or "company",
        "classification": meta.get("classification") or "internal",
        "version": meta.get("version") or "",
        "effective_date": meta.get("effective_date") or "",
        "status": meta.get("status") or "active",
        "owner": meta.get("owner") or "",
        "note": meta.get("note") or "",
        "facts": meta.get("facts") or "",
        "text": result.text,
        "chars": len(result.text),
        "parse_status": status,
        "parse_error": result.error,
        "ocr": result.ocr,
    }


def build_corpus(use_ocr: bool = True) -> dict:
    ensure_build_dirs()
    manifest = _load_manifest()
    files = sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    files = [path for path in files if not path.name.startswith("_")]

    docs: list[dict] = []
    failures: list[dict] = []
    scanned: list[str] = []
    empty: list[str] = []
    missing_manifest: list[str] = []

    for path in files:
        rel = path.relative_to(DATA_DIR).as_posix()
        meta = manifest.get(rel, {})
        if not meta:
            missing_manifest.append(rel)
        record = _doc_record(path, meta, use_ocr=use_ocr)
        docs.append(record)
        if record["parse_status"] == "scanned_no_text":
            scanned.append(rel)
        elif record["parse_status"] == "error":
            failures.append({"path": rel, "error": record["parse_error"]})
        elif record["parse_status"] == "empty_text":
            empty.append(rel)

    docs_path = CORPUS_DIR / "docs.jsonl"
    with docs_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    listed = set(manifest)
    actual = {path.relative_to(DATA_DIR).as_posix() for path in files}
    coverage = {
        "total_files": len(files),
        "parsed_ok": sum(1 for d in docs if d["parse_status"] == "ok"),
        "scanned_no_text": len(scanned),
        "empty_text": len(empty),
        "errors": len(failures),
        "total_chars": sum(d["chars"] for d in docs),
        "by_format": dict(Counter(d["format"] for d in docs)),
        "by_classification": dict(Counter(d["classification"] for d in docs)),
        "by_status": dict(Counter(d["status"] for d in docs)),
        "by_dept": dict(Counter(d["dept"] for d in docs)),
        "scanned_files": scanned,
        "ocr_files": [d["path"] for d in docs if d["ocr"] and d["chars"] > 0],
        "ocr_errors": [
            {"path": d["path"], "error": d["parse_error"]}
            for d in docs
            if d["ocr"] and d["parse_error"]
        ],
        "empty_files": empty,
        "failures": failures,
        "missing_manifest": missing_manifest,
        "manifest_only": sorted(listed - actual),
    }
    (CORPUS_DIR / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ocr", action="store_true", help="跳过扫描件 OCR")
    args = parser.parse_args()
    build_corpus(use_ocr=OCR_ENABLED and not args.no_ocr)
