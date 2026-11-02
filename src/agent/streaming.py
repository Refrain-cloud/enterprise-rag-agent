from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Any, Iterator


class StreamSink:
    """把工作线程里的事件安全地投递到事件循环的队列。"""

    def __init__(self, loop, queue) -> None:
        self._loop = loop
        self._queue = queue
        self._closed = threading.Event()

    def emit(self, event: str, data: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        try:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, {"event": event, "data": data}
            )
        except RuntimeError:
            self._closed.set()

    def close(self) -> None:
        self._closed.set()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


_SINK: contextvars.ContextVar[StreamSink | None] = contextvars.ContextVar(
    "rag_stream_sink", default=None
)


def current_sink() -> StreamSink | None:
    return _SINK.get()


@contextmanager
def use_sink(sink: StreamSink) -> Iterator[None]:
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)


def emit_status(node: str, phase: str, **summary: Any) -> None:
    sink = current_sink()
    if sink is not None:
        sink.emit("status", {"node": node, "phase": phase, **summary})


def emit_delta(text: str) -> None:
    if not text:
        return
    sink = current_sink()
    if sink is not None:
        sink.emit("delta", {"text": text})


def emit_reset(reason: str, **meta: Any) -> None:
    sink = current_sink()
    if sink is not None:
        sink.emit("reset", {"reason": reason, **meta})
