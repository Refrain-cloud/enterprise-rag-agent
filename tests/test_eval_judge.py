from __future__ import annotations

import unittest

from eval.run_eval import _assemble_judgement, _number_in, _quote_supports_claim


class JudgeRefusalTest(unittest.TestCase):
    def test_unresolved_wording_counts_as_refusal(self) -> None:
        question = {
            "id": "ref_08",
            "expect_refusal": True,
            "key_facts": [],
            "gold_answer": "无法确定。记录只说是否迁移没有定。",
            "category": "拒答",
        }
        answer = "没有确定要迁移。记录只说“有没有下次再问”，未形成最终结论。"

        result = _assemble_judgement(
            question,
            {"facts": [], "contradictions": [], "refused": False},
            "",
            "",
            False,
            answer,
        )

        self.assertTrue(result["refused"])
        self.assertEqual(result["score"], 2)
        self.assertTrue(result["refusal_ok"])


class JudgeNumberMatchTest(unittest.TestCase):
    def test_number_does_not_match_inside_date(self) -> None:
        self.assertFalse(_number_in("10", "生效日期 2024-01-01"))
        self.assertTrue(_number_in("10", "出差结束后 10 个工作日内提交"))

    def test_quote_with_wrong_number_is_not_support(self) -> None:
        claim = "2024 年版出差结束后 10 个工作日内提交报销单"
        quote = "2019 年版出差结束后 15 个工作日内提交报销单 [SC-005]"
        answer = "2024 年版的生效日期为 2024-01-01，2019 年版为 15 个工作日。"

        self.assertFalse(_quote_supports_claim(claim, quote, answer))

    def test_identifier_digits_and_comma_numbers_are_supported(self) -> None:
        answer = "依据 HR-004 与 V3.2 说明，应收账款 1,240 万元 [HR-004]。"
        self.assertTrue(_quote_supports_claim("HR-004 规定", answer, answer))
        self.assertTrue(_quote_supports_claim("V3.2 说明", answer, answer))
        self.assertTrue(_quote_supports_claim("应收账款 1,240 万元", answer, answer))
        self.assertTrue(
            _quote_supports_claim(
                "2024 年生效", "生效日期 2024-01-01", "生效日期 2024-01-01"
            )
        )


if __name__ == "__main__":
    unittest.main()
