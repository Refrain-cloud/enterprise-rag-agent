from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import main
from src.config import API_MAX_CONCURRENCY

FAKE_RESULT = {
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
    "trace": [
        {"node": "prepare", "seq": 1, "elapsed_ms": 1.0, "status": "ok"}
    ],
    "trace_meta": {"request_id": "test", "total_ms": 1.0, "nodes": 1},
}


def fake_answer(question, role="普通员工", top_k=12, request_id=None):
    return dict(FAKE_RESULT)


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.warmup = patch.object(main, "_warm_up", lambda: None)
        self.warmup.start()
        self.addCleanup(self.warmup.stop)

    def _client(self) -> TestClient:
        return TestClient(main.app)

    def test_health_and_roles(self) -> None:
        with self._client() as client:
            health = client.get("/api/health").json()
            roles = client.get("/api/roles").json()
        self.assertTrue(health["index_ready"])
        self.assertIn("普通员工", roles["roles"])

    def test_ask_success(self) -> None:
        with self._client() as client, patch.object(main, "answer", fake_answer):
            response = client.post(
                "/api/ask",
                json={"question": "员工年假几天？", "role": "普通员工"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["route"], "rag")
        self.assertIn("HR-002", body["citations"])
        self.assertIsNone(body["trace"])
        self.assertIn("request_id", body)
        self.assertIn("usage", body)

    def test_debug_returns_trace(self) -> None:
        with self._client() as client, patch.object(main, "answer", fake_answer):
            response = client.post(
                "/api/ask",
                json={"question": "员工年假几天？", "debug": True},
            )
        body = response.json()
        self.assertEqual(len(body["trace"]), 1)
        self.assertEqual(body["self_check"]["ok"], True)

    def test_blank_question_is_rejected(self) -> None:
        with self._client() as client:
            response = client.post("/api/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_unknown_role_is_rejected(self) -> None:
        with self._client() as client:
            response = client.post(
                "/api/ask", json={"question": "年假几天？", "role": "老板"}
            )
        self.assertEqual(response.status_code, 422)

    def test_internal_error_hides_details(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("secret stack detail")

        with self._client() as client, patch.object(main, "answer", boom), patch.object(
            main.logger, "exception"
        ):
            response = client.post(
                "/api/ask", json={"question": "员工年假几天？"}
            )
        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["code"], 500)
        self.assertNotIn("secret", body["message"])
        self.assertTrue(body["request_id"])

    def test_request_timeout_returns_504(self) -> None:
        def slow(*args, **kwargs):
            time.sleep(0.2)
            return dict(FAKE_RESULT)

        with self._client() as client, patch.object(main, "answer", slow):
            with patch.object(main, "API_REQUEST_TIMEOUT_S", 0.01):
                response = client.post(
                    "/api/ask", json={"question": "员工年假几天？"}
                )
        self.assertEqual(response.status_code, 504)

    def test_queue_timeout_returns_429(self) -> None:
        with self._client() as client:
            semaphore = main.app.state.semaphore
            for _ in range(API_MAX_CONCURRENCY):
                client.portal.call(semaphore.acquire)
            try:
                with patch.object(main, "API_QUEUE_TIMEOUT_S", 0.01):
                    response = client.post(
                        "/api/ask", json={"question": "员工年假几天？"}
                    )
            finally:
                for _ in range(API_MAX_CONCURRENCY):
                    client.portal.call(semaphore.release)
        self.assertEqual(response.status_code, 429)

    def test_static_ui_and_security_headers(self) -> None:
        with self._client() as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("企业知识库 Agent", response.text)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])


if __name__ == "__main__":
    unittest.main()
