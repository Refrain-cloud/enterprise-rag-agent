from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def summarize(results: list[dict], reference_path: Path | None = None) -> dict:
    node_calls: Counter[str] = Counter()
    node_elapsed: dict[str, list[float]] = {}
    structured_attempts = 0
    structured_fallbacks = 0

    for result in results:
        trace = result.get("trace") or []
        if not trace:
            continue
        nodes = [event.get("node", "") for event in trace]
        if "structured" in nodes:
            structured_attempts += 1
        if any(
            event.get("node") == "structured" and event.get("status") == "fallback"
            for event in trace
        ):
            structured_fallbacks += 1
        for event in trace:
            node = event.get("node", "")
            node_calls[node] += 1
            node_elapsed.setdefault(node, []).append(float(event.get("elapsed_ms") or 0.0))

    rewrites = [result.get("rewrite") or {} for result in results]
    self_checks = [
        result.get("self_check") or {}
        for result in results
        if result.get("self_check")
    ]
    finding_types = Counter(
        finding.get("type", "")
        for check in self_checks
        for finding in check.get("findings") or []
    )

    report = {
        "routes": dict(sorted(Counter(result.get("route") for result in results).items())),
        "route_consistency": None,
        "route_reference_size": 0,
        "node_calls": dict(sorted(node_calls.items())),
        "node_avg_ms": {
            node: round(sum(values) / len(values), 1)
            for node, values in sorted(node_elapsed.items())
        },
        "structured_attempts": structured_attempts,
        "structured_fallbacks": structured_fallbacks,
        "rewrite_triggered": sum(1 for item in rewrites if item.get("triggered")),
        "rewrite_succeeded": sum(1 for item in rewrites if item.get("succeeded")),
        "self_check_questions": len(self_checks),
        "self_check_warnings": sum(1 for check in self_checks if not check.get("ok", True)),
        "finding_types": dict(sorted(finding_types.items())),
        "avg_total_ms": _avg_total_ms(results),
    }

    if reference_path and reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        common = [
            result["id"]
            for result in results
            if result["id"] in reference and reference[result["id"]].get("route")
        ]
        if common:
            matches = sum(
                1
                for result in results
                if result["id"] in common
                and result.get("route") == reference[result["id"]]["route"]
            )
            report["route_consistency"] = round(matches / len(common), 4)
            report["route_reference_size"] = len(common)
    return report


def _avg_total_ms(results: list[dict]) -> float | None:
    values = [
        float((result.get("trace_meta") or {}).get("total_ms"))
        for result in results
        if (result.get("trace_meta") or {}).get("total_ms") is not None
    ]
    return round(sum(values) / len(values), 1) if values else None
