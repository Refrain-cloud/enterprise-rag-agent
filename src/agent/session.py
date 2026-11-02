from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.config import (
    SESSION_ANSWER_CHARS,
    SESSION_MAX_COST_CNY,
    SESSION_MAX_COUNT,
    SESSION_MAX_TOKENS,
    SESSION_MAX_TURNS,
    SESSION_TTL_S,
)


class SessionError(Exception):
    status_code = 400
    code = "session_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SessionNotFound(SessionError):
    status_code = 404
    code = "session_not_found"


class SessionRoleMismatch(SessionError):
    status_code = 409
    code = "session_role_mismatch"


class SessionLimit(SessionError):
    status_code = 429
    code = "session_limit"


class SessionBusy(SessionError):
    status_code = 429
    code = "session_busy"


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str
    rewritten_question: str
    usage: dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class SessionView:
    """只读快照，供服务层与图线程使用，避免并发改写存储内部状态。"""

    session_id: str
    role: str
    history: tuple[Turn, ...]
    total_tokens: int
    total_cost_cny: float
    max_turns: int
    turn_token: str = ""

    @property
    def turn_count(self) -> int:
        return len(self.history)

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self.turn_count)

    def history_payload(self) -> list[dict[str, str]]:
        return [
            {
                "question": turn.question,
                "answer": turn.answer,
                "rewritten_question": turn.rewritten_question,
            }
            for turn in self.history
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "role": self.role,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "total_cost_cny": round(self.total_cost_cny, 6),
            "remaining_turns": self.remaining_turns,
        }


@dataclass
class _Session:
    session_id: str
    role: str
    created_at: float
    last_seen: float
    turns: list[Turn] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_cny: float = 0.0
    busy: bool = False
    turn_token: str | None = None


class SessionStore:
    def __init__(
        self,
        *,
        ttl_s: float = SESSION_TTL_S,
        max_turns: int = SESSION_MAX_TURNS,
        max_tokens: int = SESSION_MAX_TOKENS,
        max_cost_cny: float = SESSION_MAX_COST_CNY,
        max_count: int = SESSION_MAX_COUNT,
        answer_chars: int = SESSION_ANSWER_CHARS,
        clock: Callable[[], float] = time.time,
        sweep_interval_s: float = 60.0,
    ) -> None:
        self.ttl_s = ttl_s
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_cost_cny = max_cost_cny
        self.max_count = max_count
        self.answer_chars = answer_chars
        self._clock = clock
        self._sweep_interval_s = sweep_interval_s
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        self._last_sweep = 0.0

    def create(self, role: str) -> SessionView:
        with self._lock:
            now = self._clock()
            self._maybe_sweep(now)
            self._evict_lru_if_needed(extra=1)
            session_id = self._new_id()
            session = _Session(
                session_id=session_id,
                role=role,
                created_at=now,
                last_seen=now,
            )
            self._sessions[session_id] = session
            return self._view(session)

    def begin_turn(self, session_id: str, role: str) -> SessionView:
        with self._lock:
            session = self._live_session(session_id)
            if session.role != role:
                raise SessionRoleMismatch("会话角色与当前角色不一致，请新建会话")
            if session.busy:
                raise SessionBusy("该会话正在处理上一轮问题，请稍候重试")
            if len(session.turns) >= self.max_turns:
                raise SessionLimit("会话轮次已达上限，请新建会话")
            if (
                session.total_tokens >= self.max_tokens
                or session.total_cost_cny >= self.max_cost_cny
            ):
                raise SessionLimit("会话累计用量已达上限，请新建会话")
            session.busy = True
            session.turn_token = secrets.token_hex(8)
            session.last_seen = self._clock()
            return self._view(session)

    def finish_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        rewritten_question: str,
        usage: dict[str, Any],
        token: str | None = None,
    ) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            now = self._clock()
            if self._expired(session, now):
                self._sessions.pop(session_id, None)
                return
            if token is not None and token != session.turn_token:
                # 已被 end_turn 收尾或已被下一轮接管：只累计用量，不写历史
                self._add_usage(session, usage)
                return
            session.turns.append(
                Turn(
                    question=question,
                    answer=(answer or "")[: self.answer_chars],
                    rewritten_question=rewritten_question,
                    usage=dict(usage or {}),
                    created_at=now,
                )
            )
            self._add_usage(session, usage)
            session.last_seen = now
            session.busy = False
            session.turn_token = None

    def end_turn(self, session_id: str, token: str | None = None) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            if token is not None and token != session.turn_token:
                return
            session.busy = False
            session.turn_token = None

    @staticmethod
    def _add_usage(session: _Session, usage: dict[str, Any]) -> None:
        session.total_tokens += int((usage or {}).get("total_tokens", 0) or 0)
        session.total_cost_cny += float((usage or {}).get("cost_cny", 0.0) or 0.0)

    def snapshot(self, session_id: str) -> SessionView:
        with self._lock:
            session = self._live_session(session_id)
            session.last_seen = self._clock()
            return self._view(session)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def sweep(self, now: float | None = None) -> int:
        with self._lock:
            current = self._clock() if now is None else now
            removed = self._sweep_expired(current)
            removed += self._evict_lru_if_needed(extra=0)
            self._last_sweep = current
            return removed

    def _live_session(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        now = self._clock()
        if session is None or self._expired(session, now):
            self._sessions.pop(session_id, None)
            raise SessionNotFound("会话不存在或已过期，请新建会话")
        session.last_seen = now
        return session

    def _expired(self, session: _Session, now: float) -> bool:
        return now - session.last_seen > self.ttl_s

    def _maybe_sweep(self, now: float) -> None:
        if now - self._last_sweep >= self._sweep_interval_s:
            self._sweep_expired(now)
            self._last_sweep = now

    def _sweep_expired(self, now: float) -> int:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if self._expired(session, now)
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
        return len(expired)

    def _evict_lru_if_needed(self, *, extra: int) -> int:
        overflow = len(self._sessions) + extra - self.max_count
        if overflow <= 0:
            return 0
        ordered = sorted(
            self._sessions.values(), key=lambda session: session.last_seen
        )
        for session in ordered[:overflow]:
            self._sessions.pop(session.session_id, None)
        return overflow

    def _new_id(self) -> str:
        while True:
            session_id = secrets.token_urlsafe(24)
            if session_id not in self._sessions:
                return session_id

    def _view(self, session: _Session) -> SessionView:
        return SessionView(
            session_id=session.session_id,
            role=session.role,
            history=tuple(session.turns),
            total_tokens=session.total_tokens,
            total_cost_cny=session.total_cost_cny,
            max_turns=self.max_turns,
            turn_token=session.turn_token or "",
        )
