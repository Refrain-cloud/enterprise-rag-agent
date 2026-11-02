from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.run_multiturn import (
    QUESTIONS_PATH,
    load_groups,
    run_group,
    summarize_multiturn,
)
from src.agent.session import SessionStore


def fake_judged(score: int = 2) -> dict:
    return {
        "score": score,
        "faithful": True,
        "refusal_ok": True,
        "permission_ok": True,
        "reason": "",
        "facts": [],
        "contradictions": [],
        "unsupported_claims": [],
        "refused": False,
    }


def fake_response(question: str, history, *, citations=None) -> dict:
    rewritten = f"改写：{question}" if history else question
    return {
        "answer": f"回答 {question} [DOC-1]。",
        "sources": [{"doc_id": "DOC-1"}],
        "hits": [
            {
                "metadata": {
                    "doc_id": "DOC-1",
                    "title": "测试文档",
                    "status": "active",
                    "classification": "internal",
                },
                "text": "DOC-1 原文",
            }
        ],
        "route": "rag",
        "citations": citations or ["DOC-1"],
        "rewritten_question": rewritten,
        "trace": [
            {
                "node": "prepare",
                "summary": {
                    "followup_rewrite": {
                        "triggered": bool(history),
                        "succeeded": bool(history),
                        "query": rewritten if history else None,
                    }
                },
            }
        ],
        "retrieval": {"queries": [rewritten]},
    }


GROUP = {
    "id": "g1",
    "role": "普通员工",
    "category": "多轮事实追问",
    "turns": [
        {
            "question": "第一问？",
            "gold_answer": "第一答。",
            "gold_doc_ids": ["DOC-1"],
            "key_facts": [],
        },
        {
            "question": "那第二问呢？",
            "gold_answer": "第二答。",
            "gold_doc_ids": ["DOC-1"],
            "key_facts": [],
        },
    ],
}


class QuestionFileTest(unittest.TestCase):
    def test_v1_file_structure(self) -> None:
        groups = load_groups(QUESTIONS_PATH)
        self.assertEqual(len(groups), 6)
        self.assertEqual(sum(len(group["turns"]) for group in groups), 15)
        ids = [group["id"] for group in groups]
        self.assertEqual(len(ids), len(set(ids)))
        for group in groups:
            for turn in group["turns"]:
                self.assertTrue(turn["question"])
                self.assertTrue(turn["gold_answer"])
                if not turn.get("expect_refusal"):
                    self.assertTrue(turn["gold_doc_ids"])
                    self.assertTrue(turn.get("key_facts"))

    def test_rejects_duplicate_group_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.jsonl"
            line = json.dumps(GROUP, ensure_ascii=False)
            path.write_text(f"{line}\n{line}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_groups(path)


class RunGroupTest(unittest.TestCase):
    def test_history_and_session_are_wired(self) -> None:
        store = SessionStore()
        calls: list[dict] = []

        def fake_answer(question: str, **kwargs):
            calls.append({"question": question, **kwargs})
            return fake_response(question, kwargs.get("history"))

        rows = run_group(
            GROUP,
            store=store,
            answer_fn=fake_answer,
            judge_fn=lambda *args, **kwargs: fake_judged(),
        )

        self.assertEqual(len(rows), 2)
        self.assertIsNone(calls[0]["history"])
        self.assertEqual(calls[1]["history"][0]["question"], "第一问？")
        self.assertEqual(rows[1]["history_turns"], 1)
        self.assertTrue(rows[1]["followup_rewrite"]["triggered"])
        self.assertEqual(
            rows[1]["rewritten_question"], "改写：那第二问呢？"
        )
        session = store.snapshot(calls[0]["session_id"])
        self.assertEqual(session.turn_count, 2)

        summary = summarize_multiturn(rows, [GROUP])
        self.assertEqual(summary["rewrite_triggered"], 1)
        self.assertEqual(summary["rewrite_success_rate"], 1.0)
        self.assertEqual(summary["followup_avg_score"], 2.0)
        self.assertEqual(summary["citations_outside_retrieved_turns"], [])

    def test_detects_history_document_citation(self) -> None:
        group = {
            **GROUP,
            "turns": [
                GROUP["turns"][0],
                {
                    **GROUP["turns"][1],
                    "forbidden_citation_doc_ids": ["HR-002"],
                },
            ],
        }
        store = SessionStore()

        def fake_answer(question: str, **kwargs):
            citations = ["HR-002"] if kwargs.get("history") else ["DOC-1"]
            return fake_response(question, kwargs.get("history"), citations=citations)

        rows = run_group(
            group,
            store=store,
            answer_fn=fake_answer,
            judge_fn=lambda *args, **kwargs: fake_judged(),
        )
        self.assertEqual(rows[1]["forbidden_citation_hits"], ["HR-002"])
        self.assertEqual(rows[1]["citations_outside_retrieved"], ["HR-002"])
        summary = summarize_multiturn(rows, [group])
        self.assertEqual(summary["forbidden_citation_turns"], ["g1_t2"])
        self.assertEqual(summary["citations_outside_retrieved_turns"], ["g1_t2"])


if __name__ == "__main__":
    unittest.main()
