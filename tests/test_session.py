from __future__ import annotations

import unittest

from src.agent.session import (
    SessionBusy,
    SessionLimit,
    SessionNotFound,
    SessionRoleMismatch,
    SessionStore,
)


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def usage(tokens: int = 100, cost: float = 0.001) -> dict:
    return {"total_tokens": tokens, "cost_cny": cost}


class SessionStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.store = SessionStore(
            ttl_s=60,
            max_turns=3,
            max_tokens=1000,
            max_cost_cny=1.0,
            max_count=3,
            answer_chars=20,
            clock=self.clock,
            sweep_interval_s=0,
        )

    def _finish(self, session_id: str, answer: str = "答案") -> None:
        self.store.finish_turn(
            session_id,
            question="问题",
            answer=answer,
            rewritten_question="问题",
            usage=usage(),
        )

    def test_create_and_finish_turn(self) -> None:
        view = self.store.create("普通员工")
        self.assertEqual(view.turn_count, 0)
        self.assertEqual(view.remaining_turns, 3)
        self.assertEqual(view.session_id, view.session_id)
        self.assertTrue(view.session_id)

        began = self.store.begin_turn(view.session_id, "普通员工")
        self.assertEqual(began.turn_count, 0)
        self._finish(view.session_id, answer="x" * 50)

        snapshot = self.store.snapshot(view.session_id)
        self.assertEqual(snapshot.turn_count, 1)
        self.assertEqual(snapshot.total_tokens, 100)
        self.assertAlmostEqual(snapshot.total_cost_cny, 0.001)
        self.assertEqual(len(snapshot.history[0].answer), 20)

    def test_unknown_and_expired_session(self) -> None:
        with self.assertRaises(SessionNotFound):
            self.store.begin_turn("missing", "普通员工")

        view = self.store.create("普通员工")
        self.clock.advance(61)
        with self.assertRaises(SessionNotFound):
            self.store.snapshot(view.session_id)

    def test_role_mismatch(self) -> None:
        view = self.store.create("普通员工")
        with self.assertRaises(SessionRoleMismatch):
            self.store.begin_turn(view.session_id, "HR")

    def test_busy_rejects_parallel_turn(self) -> None:
        view = self.store.create("普通员工")
        self.store.begin_turn(view.session_id, "普通员工")
        with self.assertRaises(SessionBusy):
            self.store.begin_turn(view.session_id, "普通员工")
        self.store.end_turn(view.session_id)
        self.store.begin_turn(view.session_id, "普通员工")

    def test_turn_limit(self) -> None:
        view = self.store.create("普通员工")
        for _ in range(3):
            self.store.begin_turn(view.session_id, "普通员工")
            self._finish(view.session_id)
        with self.assertRaises(SessionLimit):
            self.store.begin_turn(view.session_id, "普通员工")

    def test_token_and_cost_limits(self) -> None:
        view = self.store.create("普通员工")
        self.store.begin_turn(view.session_id, "普通员工")
        self.store.finish_turn(
            view.session_id,
            question="问题",
            answer="答案",
            rewritten_question="问题",
            usage={"total_tokens": 1000, "cost_cny": 0.0},
        )
        with self.assertRaises(SessionLimit):
            self.store.begin_turn(view.session_id, "普通员工")

        cost_view = self.store.create("普通员工")
        self.store.begin_turn(cost_view.session_id, "普通员工")
        self.store.finish_turn(
            cost_view.session_id,
            question="问题",
            answer="答案",
            rewritten_question="问题",
            usage={"total_tokens": 0, "cost_cny": 1.0},
        )
        with self.assertRaises(SessionLimit):
            self.store.begin_turn(cost_view.session_id, "普通员工")

    def test_lru_eviction(self) -> None:
        first = self.store.create("普通员工")
        self.clock.advance(1)
        self.store.create("普通员工")
        self.clock.advance(1)
        self.store.create("普通员工")
        self.clock.advance(1)
        self.store.create("普通员工")
        with self.assertRaises(SessionNotFound):
            self.store.snapshot(first.session_id)

    def test_sweep_removes_expired(self) -> None:
        self.store.create("普通员工")
        self.clock.advance(120)
        self.assertEqual(self.store.sweep(), 1)

    def test_history_payload_is_copy(self) -> None:
        view = self.store.create("普通员工")
        self.store.begin_turn(view.session_id, "普通员工")
        self._finish(view.session_id)
        payload = self.store.snapshot(view.session_id).history_payload()
        payload[0]["question"] = "篡改"
        self.assertEqual(self.store.snapshot(view.session_id).history[0].question, "问题")

    def test_stale_turn_cannot_write_history_or_clear_busy(self) -> None:
        view = self.store.create("普通员工")
        first = self.store.begin_turn(view.session_id, "普通员工")
        self.assertTrue(first.turn_token)
        self.store.end_turn(view.session_id, first.turn_token)

        second = self.store.begin_turn(view.session_id, "普通员工")
        self.assertNotEqual(second.turn_token, first.turn_token)

        # 客户端中断后晚到的 worker：只累计用量，不写历史、不动当前轮的 busy
        self.store.finish_turn(
            view.session_id,
            question="旧问题",
            answer="旧答案",
            rewritten_question="旧问题",
            usage=usage(50, 0.0005),
            token=first.turn_token,
        )
        snapshot = self.store.snapshot(view.session_id)
        self.assertEqual(snapshot.turn_count, 0)
        self.assertEqual(snapshot.total_tokens, 50)
        with self.assertRaises(SessionBusy):
            self.store.begin_turn(view.session_id, "普通员工")

        self.store.end_turn(view.session_id, first.turn_token)
        with self.assertRaises(SessionBusy):
            self.store.begin_turn(view.session_id, "普通员工")

        self.store.finish_turn(
            view.session_id,
            question="新问题",
            answer="新答案",
            rewritten_question="新问题",
            usage=usage(),
            token=second.turn_token,
        )
        snapshot = self.store.snapshot(view.session_id)
        self.assertEqual(snapshot.turn_count, 1)
        self.assertEqual(snapshot.history[0].question, "新问题")
        self.assertEqual(snapshot.total_tokens, 150)


if __name__ == "__main__":
    unittest.main()
