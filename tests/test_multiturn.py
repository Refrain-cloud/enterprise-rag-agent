from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agent import nodes
from src.agent.runtime import (
    build_generation_messages,
    format_context,
    format_history,
)
from src.agent.state import initial_state

HIT = {
    "metadata": {
        "doc_id": "HR-002",
        "title": "考勤与休假管理制度 V2",
        "version": "V2.0",
        "status": "active",
        "effective_date": "2024-07-01",
        "classification": "internal",
        "path": "hr/HR-002.docx",
    },
    "text": "满 3 年不满 5 年，年休假 10 天。",
}

HISTORY = [
    {
        "question": "员工满 3 年不满 5 年，年休假几天？",
        "answer": "年休假 10 天 [HR-002]。",
        "rewritten_question": "员工满 3 年不满 5 年，年休假几天？",
    }
]


def fake_chat(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class HistoryPromptTest(unittest.TestCase):
    def test_empty_history_keeps_prompt_identical(self) -> None:
        without, _ = build_generation_messages("年假几天？", "普通员工", [HIT])
        empty, _ = build_generation_messages(
            "年假几天？", "普通员工", [HIT], history=[]
        )
        self.assertEqual(without, empty)
        expected = (
            f"当前用户角色：普通员工\n\n参考资料：\n{format_context([HIT])}"
            "\n\n问题：年假几天？"
        )
        self.assertEqual(without[1]["content"], expected)

    def test_history_strips_citations_and_is_fenced(self) -> None:
        messages, _ = build_generation_messages(
            "那满 5 年呢？", "普通员工", [HIT], history=HISTORY
        )
        content = messages[1]["content"]
        self.assertIn("对话历史（仅用于理解指代，不得作为引用来源）", content)
        history_section = content.split("对话历史", 1)[1].split("参考资料：", 1)[0]
        self.assertNotIn("[HR-002]", history_section)
        self.assertIn("第1轮 用户：", content)
        self.assertIn("参考资料：", content)

    def test_history_truncation(self) -> None:
        history = [
            {"question": "q" * 100, "answer": "a" * 100} for _ in range(5)
        ]
        text = format_history(history, max_turns=3, max_chars=120)
        self.assertLessEqual(len(text), 120)
        self.assertIn("第3轮", text)
        self.assertNotIn("第2轮", text)


class PrepareRewriteTest(unittest.TestCase):
    def test_no_history_skips_rewrite(self) -> None:
        with patch.object(nodes, "chat_completion") as mocked:
            result = nodes.prepare_node(initial_state("年假几天？", "普通员工", 12))
        mocked.assert_not_called()
        self.assertEqual(result["rewritten_question"], "年假几天？")
        summary = result["trace"][0]["summary"]
        self.assertFalse(summary["followup_rewrite"]["triggered"])

    def test_history_triggers_rewrite(self) -> None:
        state = initial_state(
            "那满 5 年呢？", "普通员工", 12, history=HISTORY
        )
        with patch.object(
            nodes,
            "chat_completion",
            return_value=fake_chat("员工满 5 年不满 10 年，年休假几天？"),
        ) as mocked:
            result = nodes.prepare_node(state)
        mocked.assert_called_once()
        self.assertEqual(
            result["rewritten_question"], "员工满 5 年不满 10 年，年休假几天？"
        )
        summary = result["trace"][0]["summary"]
        self.assertTrue(summary["followup_rewrite"]["triggered"])
        self.assertTrue(summary["followup_rewrite"]["succeeded"])
        user_message = mocked.call_args.kwargs["messages"][1]["content"]
        self.assertIn("对话历史：", user_message)
        self.assertIn("那满 5 年呢？", user_message)
        self.assertNotIn("[HR-002]", user_message)

    def test_rewrite_failure_falls_back(self) -> None:
        state = initial_state(
            "那满 5 年呢？", "普通员工", 12, history=HISTORY
        )
        with patch.object(
            nodes, "chat_completion", side_effect=RuntimeError("api down")
        ):
            result = nodes.prepare_node(state)
        self.assertEqual(result["rewritten_question"], "那满 5 年呢？")
        self.assertFalse(
            result["trace"][0]["summary"]["followup_rewrite"]["succeeded"]
        )


class EffectiveQuestionTest(unittest.TestCase):
    def test_retrieve_uses_rewritten_question(self) -> None:
        seen: list[str] = []

        def fake_search(question, role, top_k, **kwargs):
            seen.append(question)
            return [HIT]

        state = initial_state("那满 5 年呢？", "普通员工", 12)
        state["rewritten_question"] = "员工满 5 年不满 10 年，年休假几天？"
        with patch.object(nodes, "hybrid_search", fake_search):
            nodes.retrieve_node(state)
        self.assertEqual(
            set(seen), {"员工满 5 年不满 10 年，年休假几天？"}
        )

    def test_route_uses_rewritten_question(self) -> None:
        state = initial_state("那总共多少？", "普通员工", 12)
        state["rewritten_question"] = "云资源成本合计多少？"
        result = nodes.route_node(state)
        self.assertEqual(result["route_decision"], "structured")


if __name__ == "__main__":
    unittest.main()
