"""Tier-2 durability: a SQLite-backed journal for runs, events, steps and
signals.

Recovery is replay-based: on ``app.recover()`` every unfinished durable run
is re-executed from the top with (a) its step journal pre-loaded, so
completed side effects return recorded results instead of re-running,
(b) its persisted signals pre-queued, so past ``ctx.pause()`` calls resume
with the same payloads, and (c) its event history restored, with re-emitted
events deduplicated by forward subsequence matching — a replayed execution
emits a subsequence of the recorded history (journaled steps skip their
inner tool events), so each re-emitted event is matched against the next
occurrence in history and skipped rather than appended twice.

Writes are tiny synchronous sqlite3 statements in WAL mode; group commit
and a Postgres backend are the production follow-ups (see DESIGN.md).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    route TEXT NOT NULL,
    kwargs TEXT NOT NULL,
    status TEXT NOT NULL,
    tenant TEXT,
    metadata TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    result TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS signals (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    name TEXT NOT NULL,
    payload TEXT,
    PRIMARY KEY (run_id, idx)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL,
    PRIMARY KEY (run_id, idx)
);
"""


class SQLiteBackend:
    def __init__(self, path: str = "agentapi.db", *,
                 group_commit_s: float = 0.0) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        if group_commit_s:
            # Trade a bounded window of durability for far fewer fsyncs:
            # WAL + NORMAL means a commit survives process death (only an
            # OS-level crash can lose the tail).
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.group_commit_s = group_commit_s
        self._pending = 0
        self._last_commit = time.monotonic()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            if not self.group_commit_s:
                self._conn.commit()
            else:
                self._pending += 1
                now = time.monotonic()
                if now - self._last_commit >= self.group_commit_s:
                    self._conn.commit()
                    self._pending = 0
                    self._last_commit = now
            return cursor

    def flush(self) -> None:
        """Commit any group-commit backlog. Called on run completion so a
        finished run is always durable regardless of the batching window."""
        with self._lock:
            if self._pending:
                self._conn.commit()
                self._pending = 0
                self._last_commit = time.monotonic()

    # -- runs ----------------------------------------------------------------
    def create_run(self, run_id: str, route: str, kwargs: dict[str, Any], *,
                   tenant: Optional[str], metadata: dict[str, Any]) -> None:
        self._execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,NULL)",
            (run_id, route, json.dumps(kwargs, default=str), "running",
             tenant, json.dumps(metadata, default=str), time.time()))

    def update_status(self, run_id: str, status: str,
                      finished: bool = False) -> None:
        self._execute(
            "UPDATE runs SET status=?, finished_at=? WHERE id=?",
            (status, time.time() if finished else None, run_id))

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self._execute(
            "SELECT id, route, kwargs, status, tenant, metadata, created_at,"
            " finished_at FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "route": row[1], "kwargs": json.loads(row[2]),
            "status": row[3], "tenant": row[4],
            "metadata": json.loads(row[5] or "{}"),
            "created_at": row[6], "finished_at": row[7],
        }

    def unfinished_runs(self) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT id FROM runs WHERE status IN ('running','paused')"
        ).fetchall()
        return [self.get_run(row[0]) for row in rows]

    # -- events / steps / signals -------------------------------------------
    def append_event(self, run_id: str, seq: int, payload: dict[str, Any]) -> None:
        self._execute(
            "INSERT OR REPLACE INTO events VALUES (?,?,?)",
            (run_id, seq, json.dumps(payload, default=str)))

    def events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT payload FROM events WHERE run_id=? ORDER BY seq",
            (run_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_step(self, run_id: str, key: str, result: Any) -> None:
        self._execute("INSERT OR REPLACE INTO steps VALUES (?,?,?)",
                      (run_id, key, json.dumps(result, default=str)))

    def steps(self, run_id: str) -> dict[str, Any]:
        rows = self._execute(
            "SELECT key, result FROM steps WHERE run_id=?", (run_id,)).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def append_signal(self, run_id: str, name: str, payload: Any) -> None:
        row = self._execute(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM signals WHERE run_id=?",
            (run_id,)).fetchone()
        self._execute("INSERT INTO signals VALUES (?,?,?,?)",
                      (run_id, row[0], name, json.dumps(payload, default=str)))

    def signals(self, run_id: str) -> list[tuple[str, Any]]:
        rows = self._execute(
            "SELECT name, payload FROM signals WHERE run_id=? ORDER BY idx",
            (run_id,)).fetchall()
        return [(row[0], json.loads(row[1])) for row in rows]

    # -- recorded LLM calls (for replay) ------------------------------------
    def record_llm_call(self, run_id: str, model: str,
                        request: dict[str, Any], response: dict[str, Any]) -> None:
        row = self._execute(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM llm_calls WHERE run_id=?",
            (run_id,)).fetchone()
        self._execute("INSERT INTO llm_calls VALUES (?,?,?,?,?)",
                      (run_id, row[0], model,
                       json.dumps(request, default=str),
                       json.dumps(response, default=str)))

    def llm_calls(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT model, request, response FROM llm_calls WHERE run_id=?"
            " ORDER BY idx", (run_id,)).fetchall()
        return [{"model": r[0], "request": json.loads(r[1]),
                 "response": json.loads(r[2])} for r in rows]

    def all_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT id FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
        return [self.get_run(r[0]) for r in rows]

    def close(self) -> None:
        self.flush()
        with self._lock:
            self._conn.close()
