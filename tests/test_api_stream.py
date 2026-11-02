from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agent.streaming import emit_delta
from src.api import main

BASE_RESULT = {
    "answer": "满 3 年不满 5 年为 10 天 [HR-002]",
    "sources": [
        {
            "doc_id": "HR-002",
            "title": "考勤与休假管理制度 V2",
            "version": "V2.0",
            "status": "active",
            "effective_date": "2024-07-01",
            "path": "hr/HR-002.docx",
            "classification": "internal",
        }
    ],
    "hits": [],
    "denied": False,
    "route": "rag",
    "citations": ["HR-002"],
    "self_check": {"ok": True, "findings": []},
    "trace": [{"node": "prepare", "seq": 1, "elapsed_ms": 1.0, "status": "ok"}],
    "trace_meta": {"request_id": "test", "total_ms": 1.0, "nodes": 1},
    "rewritten_question": "员工年假几天？",
}


def stream_answer(question, role="普通员工", top_k=12, request_id=None, **kwargs):
    emit_delta("满 3 年不满 5 年")
    emit_delta("为 10 天 [HR-002]")
    return dict(BASE_RESULT)


def parse_events(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event is not None:
            events.append((event, data or {}))
    return events


class StreamApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.warmup = patch.object(main, "_warm_up", lambda: None)
        self.warmup.start()
        self.addCleanup(self.warmup.stop)

    def _client(self) -> TestClient:
        return TestClient(main.app)

    def _stream(self, client: TestClient, payload: dict) -> tuple[int, str]:
        with client.stream("POST", "/api/ask/stream", json=payload) as response:
            body = "".join(response.iter_text())
            return response.status_code, body

    def test_stream_emits_meta_deltas_and_final(self) -> None:
        with self._client() as client, patch.object(
            main, "answer", stream_answer
        ):
            status, body = self._stream(
                client, {"question": "员工年假几天？", "role": "普通员工"}
            )
        self.assertEqual(status, 200)
        events = parse_events(body)
        names = [name for name, _ in events]
        self.assertEqual(names[0], "meta")
        self.assertIn("delta", names)
        self.assertEqual(names[-1], "final")

        meta = dict(events)["meta"]
        self.assertTrue(meta["session_id"])
        self.assertEqual(meta["turn"], 1)
        final = dict(events)["final"]
        self.assertEqual(final["answer"], BASE_RESULT["answer"])
        self.assertEqual(final["session"]["turn_count"], 1)
        self.assertEqual(final["session"]["remaining_turns"], 9)

    def test_session_reuse_passes_history(self) -> None:
        captured: list[list[dict] | None] = []

        def capture(question, role="普通员工", top_k=12, request_id=None, **kwargs):
            captured.append(kwargs.get("history"))
            return dict(BASE_RESULT)

        with self._client() as client, patch.object(main, "answer", capture):
            _, body = self._stream(
                client, {"question": "员工年假几天？", "role": "普通员工"}
            )
            session_id = dict(parse_events(body))["meta"]["session_id"]
            _, body2 = self._stream(
                client,
                {
                    "question": "那满 5 年呢？",
                    "role": "普通员工",
                    "session_id": session_id,
                },
            )
        self.assertIsNone(captured[0])
        self.assertEqual(len(captured[1]), 1)
        self.assertEqual(captured[1][0]["question"], "员工年假几天？")
        final2 = dict(parse_events(body2))["final"]
        self.assertEqual(final2["session"]["turn_count"], 2)

    def test_unknown_session_returns_404(self) -> None:
        with self._client() as client:
            response = client.post(
                "/api/ask/stream",
                json={"question": "年假几天？", "session_id": "missing"},
            )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], 404)

    def test_role_mismatch_returns_409(self) -> None:
        with self._client() as client, patch.object(
            main, "answer", stream_answer
        ):
            _, body = self._stream(
                client, {"question": "员工年假几天？", "role": "普通员工"}
            )
            session_id = dict(parse_events(body))["meta"]["session_id"]
            response = client.post(
                "/api/ask/stream",
                json={"question": "年假几天？", "role": "HR", "session_id": session_id},
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], 409)

    def test_worker_error_becomes_sse_error(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("secret stack detail")

        with self._client() as client, patch.object(
            main, "answer", boom
        ), patch.object(main.logger, "exception"):
            status, body = self._stream(
                client, {"question": "员工年假几天？"}
            )
        self.assertEqual(status, 200)
        events = dict(parse_events(body))
        self.assertEqual(events["error"]["code"], 500)
        self.assertNotIn("secret", events["error"]["message"])

    def test_non_stream_returns_session(self) -> None:
        with self._client() as client, patch.object(
            main, "answer", stream_answer
        ):
            response = client.post(
                "/api/ask", json={"question": "员工年假几天？"}
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["session"]["session_id"])
        self.assertEqual(body["session"]["turn_count"], 1)


if __name__ == "__main__":
    unittest.main()
