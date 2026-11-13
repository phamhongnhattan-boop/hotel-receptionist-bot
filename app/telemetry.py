"""What happens in each chat turn, saved so the demo can be measured (/api/stats).

One row per guest message: latency, model, fallback, tools called, language, whether the bot
said "not sure", and errors. Privacy: the message text, names and phone numbers are NOT saved;
the session id is stored only as a short one-way hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from contextvars import ContextVar
from pathlib import Path

log = logging.getLogger("chat")

# Tools append their name here while a reply is being made (see booking.py)
_tool_calls: ContextVar[list[str] | None] = ContextVar("tool_calls", default=None)
# Booking codes that a tool really returned in this reply (used to catch invented codes)
_codes: ContextVar[set[str] | None] = ContextVar("booking_codes", default=None)

VI_CHARS = re.compile(r"[ăâđêôơưàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹ]", re.IGNORECASE)
NOT_SURE = ("không chắc", "chưa chắc", "not sure", "0236 000 0000")


def start_tool_log() -> tuple:
    return _tool_calls.set([]), _codes.set(set())


def stop_tool_log(tokens) -> tuple[list[str], set[str]]:
    calls, codes = _tool_calls.get() or [], _codes.get() or set()
    _tool_calls.reset(tokens[0])
    _codes.reset(tokens[1])
    return calls, codes


def record_code(code: str) -> None:
    codes = _codes.get()
    if codes is not None:
        codes.add(code.upper())


def record_tool(name: str) -> None:
    calls = _tool_calls.get()
    if calls is not None:
        calls.append(name)


def guess_language(text: str) -> str:
    return "vi" if VI_CHARS.search(text or "") else "en"


def said_not_sure(reply: str) -> bool:
    low = (reply or "").lower()
    return any(k in low for k in NOT_SURE)


def short_hash(value: str) -> str:
    return hashlib.sha256((value or "").encode()).hexdigest()[:12]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 1)


class EventLog:
    def __init__(self, db_path: Path | str = ":memory:"):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chat_events (
                ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
                session TEXT, language TEXT, latency_ms REAL, model TEXT, fallback INTEGER,
                tools TEXT, top_source TEXT, top_score REAL, not_sure INTEGER, error TEXT
            );
        """)

    def add(self, *, session_id: str, message: str, latency_ms: float, result: dict | None = None,
            error: str | None = None, main_model: str | None = None) -> None:
        result = result or {}
        sources = result.get("sources") or [{}]
        model = result.get("model")
        row = {
            "session": short_hash(session_id),
            "language": guess_language(message),
            "latency_ms": round(latency_ms, 1),
            "model": model,
            "fallback": int(bool(model and main_model and model != main_model)),
            "tools": ",".join(result.get("tools", [])),
            "top_source": sources[0].get("title"),
            "top_score": sources[0].get("score"),
            "not_sure": int(said_not_sure(result.get("reply", ""))) if not error else None,
            "error": error,
        }
        with self.lock:
            self.conn.execute(f"INSERT INTO chat_events ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                              tuple(row.values()))
            self.conn.commit()
        log.info(json.dumps(row, ensure_ascii=False))  # one JSON line per turn, easy to search

    def stats(self, hours: int = 24) -> dict:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM chat_events WHERE ts >= strftime('%Y-%m-%dT%H:%M:%S', 'now', ?)",
                (f"-{int(hours)} hours",)).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM chat_events LIMIT 0").description]
        rows = [dict(zip(cols, r)) for r in rows]
        ok = [r for r in rows if not r["error"]]
        tools: dict[str, int] = {}
        for r in ok:
            for t in filter(None, (r["tools"] or "").split(",")):
                tools[t] = tools.get(t, 0) + 1
        n = len(rows)
        lat = [r["latency_ms"] for r in ok]
        return {
            "window_hours": hours,
            "messages": n,
            "sessions": len({r["session"] for r in rows}),
            "errors": n - len(ok),
            "latency_ms": {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95), "max": max(lat, default=None)},
            "fallback_rate_%": round(100 * sum(r["fallback"] for r in ok) / len(ok), 1) if ok else None,
            "not_sure_rate_%": round(100 * sum(r["not_sure"] for r in ok) / len(ok), 1) if ok else None,
            "languages": {k: sum(r["language"] == k for r in rows) for k in ("vi", "en")},
            "tool_calls": tools,
        }
