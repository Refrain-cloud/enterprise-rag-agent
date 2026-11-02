from __future__ import annotations

import unittest

from src.agent.runtime import (
    build_generation_messages,
    ensure_draft_disclosure,
    sources,
)
from src.config import GENERATION_MAX_TOKENS, GENERATION_MAX_TOKENS_HEADER


def hit(doc_id: str = "HR-002", status: str = "active") -> dict:
    return {
        "id": "chunk-1",
        "text": "年假：满 3 年不满 5 年 10 天",
        "metadata": {
            "doc_id": doc_id,
            "title": "考勤与休假管理制度 V2",
            "path": "hr/HR-002.docx",
            "format": "docx",
            "dept": "HR",
            "scope": "company",
            "classification": "internal",
            "version": "V2.0",
            "effective_date": "2024-07-01",
            "status": status,
            "chunk_index": 0,
        },
    }


class RuntimeTest(unittest.TestCase):
    def test_generation_messages_preserve_contract(self) -> None:
        messages, max_tokens = build_generation_messages(
            "员工年假几天？", "普通员工", [hit()]
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("北辰智造科技", messages[0]["content"])
        self.assertIn("当前用户角色：普通员工", messages[1]["content"])
        self.assertIn("[HR-002]", messages[1]["content"])
        self.assertEqual(max_tokens, GENERATION_MAX_TOKENS)

    def test_header_question_gets_longer_budget(self) -> None:
        _, max_tokens = build_generation_messages(
            "表头金额与明细合计是否一致？", "普通员工", [hit()]
        )
        self.assertEqual(max_tokens, GENERATION_MAX_TOKENS_HEADER)

    def test_draft_disclosure_added_once(self) -> None:
        hits = [hit(status="draft")]
        answer = ensure_draft_disclosure("计划 9 月上线 [HR-002]", hits)
        self.assertIn("草稿", answer)
        self.assertIn("未对外承诺", answer)
        self.assertEqual(ensure_draft_disclosure(answer, hits), answer)

    def test_sources_deduplicate(self) -> None:
        result = sources([hit(), hit()])
        self.assertEqual([item["doc_id"] for item in result], ["HR-002"])


if __name__ == "__main__":
    unittest.main()
