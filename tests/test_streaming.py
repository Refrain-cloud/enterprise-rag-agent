from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import llm
from src.agent import runtime
from src.agent.streaming import use_sink
from src.config import GENERATION_MAX_TOKENS_CEILING


def chunk(content=None, finish=None, usage=None, with_choice=True):
    if not with_choice:
        return SimpleNamespace(choices=[], usage=usage)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish,
            )
        ],
        usage=usage,
    )


def usage(prompt: int = 10, completion: int = 5, total: int = 15):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )


class FakeCompletions:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.factory(kwargs))


class FakeClient:
    def __init__(self, factory) -> None:
        self.completions = FakeCompletions(factory)
        self.chat = SimpleNamespace(completions=self.completions)


class StreamChatCompletionTest(unittest.TestCase):
    def setUp(self) -> None:
        llm.reset_usage()

    def _run(self, factory, **kwargs):
        client = FakeClient(factory)
        deltas: list[str] = []
        with patch.object(llm, "get_client", return_value=client):
            text, finish, meta = llm.stream_chat_completion(
                "generate",
                on_delta=deltas.append,
                model="qwen-plus",
                messages=[{"role": "user", "content": "你好"}],
                **kwargs,
            )
        return text, finish, meta, deltas, client

    def test_streams_deltas_and_records_usage_once(self) -> None:
        def factory(kwargs):
            return [
                chunk("你", None),
                chunk("好", "stop"),
                chunk(usage=usage(), with_choice=False),
            ]

        text, finish, meta, deltas, client = self._run(factory)
        self.assertEqual(text, "你好")
        self.assertEqual(finish, "stop")
        self.assertEqual(deltas, ["你", "好"])
        self.assertTrue(meta["usage_present"])
        self.assertEqual(
            client.completions.calls[0]["stream_options"], {"include_usage": True}
        )
        snapshot = llm.usage_snapshot()
        self.assertEqual(snapshot["calls"], 1)
        self.assertEqual(snapshot["total_tokens"], 15)
        self.assertEqual(snapshot["estimated_calls"], 0)

    def test_missing_usage_is_estimated(self) -> None:
        def factory(kwargs):
            return [chunk("答案", "stop")]

        text, _, meta, _, _ = self._run(factory)
        self.assertEqual(text, "答案")
        self.assertFalse(meta["usage_present"])
        snapshot = llm.usage_snapshot()
        self.assertEqual(snapshot["calls"], 1)
        self.assertEqual(snapshot["estimated_calls"], 1)

    def test_retries_without_stream_options_before_first_chunk(self) -> None:
        def factory(kwargs):
            if "stream_options" in kwargs:
                raise RuntimeError("unsupported stream_options")
            return [chunk("ok", "stop")]

        text, finish, _, _, client = self._run(factory)
        self.assertEqual(text, "ok")
        self.assertEqual(finish, "stop")
        self.assertEqual(len(client.completions.calls), 2)
        self.assertIn("stream_options", client.completions.calls[0])
        self.assertNotIn("stream_options", client.completions.calls[1])

    def test_error_after_first_chunk_is_not_retried(self) -> None:
        def factory(kwargs):
            def generator():
                yield chunk("半", None)
                raise RuntimeError("connection lost")

            return generator()

        with self.assertRaises(RuntimeError):
            self._run(factory)


class FakeSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, data: dict) -> None:
        self.events.append((event, data))


class StreamingRetryTest(unittest.TestCase):
    def test_truncation_retry_emits_reset_and_doubles_budget(self) -> None:
        sink = FakeSink()
        with use_sink(sink), patch.object(
            runtime,
            "stream_chat_completion",
            side_effect=[("a", "length", {}), ("b", "stop", {})],
        ):
            text, attempts = runtime._complete_streaming("generate", [], 100)
        self.assertEqual(text, "b")
        self.assertEqual([item["max_tokens"] for item in attempts], [100, 200])
        self.assertEqual(attempts[1]["finish_reason"], "stop")
        self.assertEqual(sink.events[0][0], "reset")
        self.assertEqual(sink.events[0][1]["reason"], "truncation_retry")
        self.assertEqual(sink.events[0][1]["max_tokens"], 200)

    def test_ceiling_stops_retry(self) -> None:
        sink = FakeSink()
        with use_sink(sink), patch.object(
            runtime,
            "stream_chat_completion",
            return_value=("partial", "length", {}),
        ):
            text, attempts = runtime._complete_streaming(
                "generate", [], GENERATION_MAX_TOKENS_CEILING
            )
        self.assertEqual(text, "partial")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(sink.events, [])


if __name__ == "__main__":
    unittest.main()
