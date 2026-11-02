from __future__ import annotations

import unittest

from src.agent.graph import get_graph
from src.agent.nodes import guard_node, route_node, self_check_node
from src.agent.state import initial_state


def hit(classification: str = "internal", scope: str = "company") -> dict:
    return {
        "id": "chunk-1",
        "text": "内部资料",
        "metadata": {
            "doc_id": "IT-001",
            "title": "IT 制度",
            "path": "it/IT-001.md",
            "format": "md",
            "dept": "IT",
            "scope": scope,
            "classification": classification,
            "version": "1.0",
            "effective_date": "2026-01-01",
            "status": "active",
            "chunk_index": 0,
        },
    }


def state(**overrides) -> dict:
    base = initial_state("员工年假几天？", "普通员工", 12)
    base.update(overrides)
    return base


class GuardTest(unittest.TestCase):
    def test_no_hits_and_unauthorized_best_match_denies(self) -> None:
        result = guard_node(
            state(hits=[], best_overall=[hit("confidential", "dept:IT")])
        )
        self.assertEqual(result["terminal"], "denied")
        self.assertEqual(result["hits"], [])
        self.assertTrue(result["answer"])

    def test_no_hits_without_best_match_is_not_found(self) -> None:
        result = guard_node(state(hits=[], best_overall=[]))
        self.assertEqual(result["terminal"], "not_found")
        self.assertEqual(result["hits"], [])

    def test_authorized_hits_pass(self) -> None:
        result = guard_node(state(hits=[hit()], best_overall=[hit()]))
        self.assertNotIn("terminal", result)
        self.assertEqual(result["trace"][0]["status"], "ok")

    def test_dept_confidential_best_match_denies_even_with_hits(self) -> None:
        result = guard_node(
            state(hits=[hit()], best_overall=[hit("confidential", "dept:IT")])
        )
        self.assertEqual(result["terminal"], "denied")

    def test_dept_confidential_denies_when_restricted_is_far_closer(self) -> None:
        result = guard_node(
            state(
                hits=[hit()],
                best_overall=[hit("confidential", "dept:IT")],
                access_margin={"gap": 0.1},
            )
        )
        self.assertEqual(result["terminal"], "denied")

    def test_dept_confidential_passes_when_accessible_is_close(self) -> None:
        result = guard_node(
            state(
                hits=[hit()],
                best_overall=[hit("confidential", "dept:IT")],
                access_margin={"gap": 0.0004},
            )
        )
        self.assertNotIn("terminal", result)
        self.assertEqual(result["trace"][0]["status"], "ok")


class RouteTest(unittest.TestCase):
    def test_aggregation_question_routes_to_structured(self) -> None:
        result = route_node(
            state(question="2026 年 9 月云资源成本的本月合计是多少元？")
        )
        self.assertEqual(result["route_decision"], "structured")

    def test_plain_question_routes_to_rag(self) -> None:
        result = route_node(state(question="员工年假几天？"))
        self.assertEqual(result["route_decision"], "rag")


class SelfCheckTest(unittest.TestCase):
    def test_unresolved_citation_is_flagged(self) -> None:
        result = self_check_node(
            state(answer="答案 [XX-999]", hits=[hit()], route="rag")
        )
        self.assertFalse(result["self_check"]["ok"])
        self.assertEqual(
            result["self_check"]["unresolved_citations"], ["XX-999"]
        )

    def test_resolved_citation_passes(self) -> None:
        result = self_check_node(
            state(answer="答案 [IT-001]", hits=[hit()], route="rag")
        )
        self.assertTrue(result["self_check"]["ok"])

    def test_denied_with_sources_is_inconsistent(self) -> None:
        result = self_check_node(
            state(
                terminal="denied",
                answer="无权访问",
                sources=[{"doc_id": "IT-001"}],
                hits=[],
                route="rag",
            )
        )
        self.assertFalse(result["self_check"]["denied_consistent"])


class GraphTest(unittest.TestCase):
    def test_graph_compiles(self) -> None:
        graph = get_graph()
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
