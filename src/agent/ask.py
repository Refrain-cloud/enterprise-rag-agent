from __future__ import annotations

import argparse

from src.agent.graph import run_agent
from src.config import ROLES, TOP_K


def answer(
    question: str,
    role: str = "普通员工",
    top_k: int = TOP_K,
    request_id: str | None = None,
) -> dict:
    return run_agent(question, role=role, top_k=top_k, request_id=request_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="北辰智造知识库问答")
    parser.add_argument("question", nargs="+", help="要提问的问题")
    parser.add_argument("--role", default="普通员工", choices=ROLES)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--debug", action="store_true", help="打印路由、SQL 与节点轨迹")
    args = parser.parse_args()
    result = answer(" ".join(args.question), role=args.role, top_k=args.top_k)
    print(result["answer"])
    if args.debug:
        print(f"\n[route] {result.get('route', '?')}")
        if result.get("sql"):
            print(f"[sql] {result['sql']}")
        if result.get("sql_result"):
            print(f"[rows] {result['sql_result']['row_count']}")
        trace = result.get("trace", [])
        if trace:
            print(f"[trace] {' → '.join(event['node'] for event in trace)}")
        self_check = result.get("self_check") or {}
        if self_check:
            print(f"[self_check] ok={self_check.get('ok')} findings={self_check.get('findings')}")
    if result["sources"]:
        print("\n来源：")
        for source in result["sources"]:
            print(
                f"- [{source['doc_id']}] {source['title']} "
                f"(v{source['version']}, {source['status']}, {source['path']})"
            )


if __name__ == "__main__":
    main()
