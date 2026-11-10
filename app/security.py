"""Security helpers for the public demo: rate limits, session tokens, input checks.

Everything is in memory, which is fine for one small instance (Render free tier).
To run several instances, move RateLimiter and SessionStore to Redis.
"""
from __future__ import annotations

import re
import secrets
import threading
import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


# ---------- Rate limiting ----------
class RateLimiter:
    """Sliding-window limit: at most `limit` hits per `window` seconds for each key.

    Used twice: per client IP (stop one user from spamming) and with one global key
    per day (cap total Gemini calls, so nobody can burn the whole API quota/budget).
    """

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic,
                 max_keys: int = 10_000):
        self.limit, self.window, self.clock, self.max_keys = limit, window, clock, max_keys
        self.hits: OrderedDict[str, deque] = OrderedDict()
        self.lock = threading.Lock()

    def check(self, key: str) -> float:
        """Record a hit. Return 0 if allowed, else how many seconds to wait."""
        now = self.clock()
        with self.lock:
            q = self.hits.pop(key, None) or deque()
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                self.hits[key] = q
                return round(self.window - (now - q[0]), 1)
            q.append(now)
            self.hits[key] = q  # most recent keys at the end
            while len(self.hits) > self.max_keys:  # do not grow forever
                self.hits.popitem(last=False)
            return 0.0


def client_ip(headers, peer: str | None, proxy_hops: int) -> str:
    """Client IP for rate limiting.

    Behind Render/Cloud Run the real IP is in X-Forwarded-For. The client can put fake
    addresses at the START of that header, but each trusted proxy APPENDS the address it
    saw. So with `proxy_hops` trusted proxies, the real client is the hops-th address
    from the END. proxy_hops = 0 means "no proxy": use the TCP peer address.
    """
    if proxy_hops > 0:
        parts = [p.strip() for p in headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(parts) >= proxy_hops:
            return parts[-proxy_hops]
    return peer or "unknown"


# ---------- Sessions ----------
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{32}$")


def new_session_id() -> str:
    """Random 192-bit token made by the server (the client cannot choose or guess it)."""
    return secrets.token_urlsafe(24)  # 32 characters


def is_valid_session_id(value: str | None) -> bool:
    return bool(value) and bool(SESSION_ID_RE.match(value))


class SessionStore(Generic[T]):
    """Chat sessions with an idle timeout and a maximum count (oldest removed first).

    Without limits, a script that opens millions of sessions fills the server memory.
    """

    def __init__(self, max_sessions: int = 500, ttl: float = 30 * 60,
                 clock: Callable[[], float] = time.monotonic):
        self.max_sessions, self.ttl, self.clock = max_sessions, ttl, clock
        self.items: OrderedDict[str, tuple[T, float]] = OrderedDict()
        self.lock = threading.Lock()

    def _drop_expired(self, now: float) -> None:
        while self.items:
            key, (_, last_used) = next(iter(self.items.items()))
            if now - last_used < self.ttl:
                break
            self.items.popitem(last=False)

    def get(self, key: str) -> T | None:
        now = self.clock()
        with self.lock:
            self._drop_expired(now)
            if key not in self.items:
                return None
            value, _ = self.items.pop(key)
            self.items[key] = (value, now)  # mark as recently used
            return value

    def set(self, key: str, value: T) -> None:
        now = self.clock()
        with self.lock:
            self._drop_expired(now)
            self.items.pop(key, None)
            self.items[key] = (value, now)
            while len(self.items) > self.max_sessions:
                self.items.popitem(last=False)

    def pop(self, key: str) -> None:
        with self.lock:
            self.items.pop(key, None)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self.lock:
            self._drop_expired(self.clock())
            return len(self.items)


# The session that is talking right now. The Gemini SDK calls the booking tools for us,
# so the tools read the session from here (contextvars are safe across threads/requests).
current_session: ContextVar[str | None] = ContextVar("current_session", default=None)


# ---------- Input checks ----------
PHONE_RE = re.compile(r"^\+?\d{9,15}$")
NAME_RE = re.compile(r"^[^\x00-\x1f<>{}\[\]\\]{2,80}$")  # no control chars or markup


def normalize_phone(phone: str) -> str | None:
    """'0905 123 456', '+84-905.123.456' -> digits. None if it is not a phone number."""
    cleaned = re.sub(r"[\s.\-()]", "", phone or "")
    return cleaned if PHONE_RE.match(cleaned) else None


def is_valid_name(name: str) -> bool:
    return bool(NAME_RE.match((name or "").strip()))


def mask_phone(phone: str) -> str:
    """Keep only the last 3 digits for logs: 0905123456 -> *******456."""
    return "*" * max(len(phone) - 3, 0) + phone[-3:]
