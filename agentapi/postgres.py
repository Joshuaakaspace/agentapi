"""PostgreSQL journal backend.

The SQLite backend is right for a single process; Postgres is what you run
when several workers share one journal — which is also what makes recovery
interesting, because any worker can pick up a run whose original process
died.

Same interface as ``SQLiteBackend``, so ``AgentAPI(durable=...)`` accepts
either. Requires ``psycopg`` (``pip install "agentapi[postgres]"``).

Multi-worker recovery uses ``claim_runs``: an atomic ``UPDATE ... RETURNING``
guarded by a claim column, so two workers recovering at the same moment
cannot both resume the same run and double its side effects.
"""
from __future__ import annotations

import json
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    route TEXT NOT NULL,
    kwargs JSONB NOT NULL,
    status TEXT NOT NULL,
    tenant TEXT,
    owner_id TEXT,
    metadata JSONB,
    created_at DOUBLE PRECISION NOT NULL,
    finished_at DOUBLE PRECISION,
    claimed_by TEXT,
    claimed_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS runs_unfinished ON runs (status)
    WHERE status IN ('running', 'paused');
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    result JSONB NOT NULL,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS signals (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    name TEXT NOT NULL,
    payload JSONB,
    PRIMARY KEY (run_id, idx)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    model TEXT NOT NULL,
    request JSONB NOT NULL,
    response JSONB NOT NULL,
    PRIMARY KEY (run_id, idx)
);
"""


class PostgresBackend:
    """Journal backed by PostgreSQL. Interface-compatible with SQLiteBackend."""

    def __init__(self, dsn: str, *, worker_id: Optional[str] = None,
                 group_commit_s: float = 0.0,
                 claim_lease_s: float = 300.0) -> None:
        try:
            import psycopg
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                'PostgresBackend requires psycopg: pip install "agentapi[postgres]"'
            ) from exc
        self.dsn = dsn
        self.worker_id = worker_id or f"worker-{id(self):x}"
        self.group_commit_s = group_commit_s     # accepted for parity
        self.claim_lease_s = claim_lease_s
        try:
            self._pool = ConnectionPool(dsn, min_size=1, max_size=8,
                                        open=True)
            self._use_pool = True
        except Exception:                         # pool unavailable
            self._conn = psycopg.connect(dsn, autocommit=True)
            self._use_pool = False
        self._run("SET synchronous_commit TO on")
        for statement in filter(None, _SCHEMA.split(";")):
            if statement.strip():
                self._run(statement)

    # -- plumbing -----------------------------------------------------------
    def _run(self, sql: str, params: tuple = ()) -> list[tuple]:
        if self._use_pool:
            with self._pool.connection() as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    if cur.description is None:
                        return []
                    return cur.fetchall()
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return []
            return cur.fetchall()

    # -- runs ---------------------------------------------------------------
    def create_run(self, run_id: str, route: str, kwargs: dict[str, Any], *,
                   tenant: Optional[str], metadata: dict[str, Any]) -> None:
        import time
        self._run(
            "INSERT INTO runs (id, route, kwargs, status, tenant, metadata,"
            " created_at) VALUES (%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (id) DO NOTHING",
            (run_id, route, json.dumps(kwargs, default=str), "running",
             tenant, json.dumps(metadata, default=str), time.time()))

    def update_status(self, run_id: str, status: str,
                      finished: bool = False) -> None:
        import time
        self._run("UPDATE runs SET status=%s, finished_at=%s WHERE id=%s",
                  (status, time.time() if finished else None, run_id))

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        rows = self._run(
            "SELECT id, route, kwargs, status, tenant, metadata, created_at,"
            " finished_at FROM runs WHERE id=%s", (run_id,))
        if not rows:
            return None
        row = rows[0]
        return {"id": row[0], "route": row[1], "kwargs": _load(row[2]),
                "status": row[3], "tenant": row[4],
                "metadata": _load(row[5]) or {},
                "created_at": row[6], "finished_at": row[7]}

    def unfinished_runs(self) -> list[dict[str, Any]]:
        rows = self._run(
            "SELECT id FROM runs WHERE status IN ('running','paused')")
        return [self.get_run(r[0]) for r in rows]

    def claim_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Atomically claim unfinished runs for this worker.

        With several workers recovering at once, only one may resume a given
        run — otherwise its side effects run twice. A run is claimable only
        if nobody holds it or the holder's lease has expired (that worker
        died mid-recovery); the UPDATE ... RETURNING makes claiming and
        reading a single statement, and SKIP LOCKED keeps concurrent workers
        from blocking each other.
        """
        import time
        now = time.time()
        rows = self._run(
            "UPDATE runs SET claimed_by=%s, claimed_at=%s WHERE id IN ("
            "  SELECT id FROM runs WHERE status IN ('running','paused')"
            "   AND (claimed_by IS NULL OR claimed_at < %s)"
            "   FOR UPDATE SKIP LOCKED LIMIT %s"
            ") RETURNING id",
            (self.worker_id, now, now - self.claim_lease_s, limit))
        return [self.get_run(r[0]) for r in rows]

    def release_claim(self, run_id: str) -> None:
        self._run("UPDATE runs SET claimed_by=NULL WHERE id=%s", (run_id,))

    # -- events / steps / signals -------------------------------------------
    def append_event(self, run_id: str, seq: int,
                     payload: dict[str, Any]) -> None:
        self._run("INSERT INTO events VALUES (%s,%s,%s)"
                  " ON CONFLICT (run_id, seq) DO UPDATE SET payload=EXCLUDED.payload",
                  (run_id, seq, json.dumps(payload, default=str)))

    def events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._run(
            "SELECT payload FROM events WHERE run_id=%s ORDER BY seq", (run_id,))
        return [_load(r[0]) for r in rows]

    def save_step(self, run_id: str, key: str, result: Any) -> None:
        self._run("INSERT INTO steps VALUES (%s,%s,%s)"
                  " ON CONFLICT (run_id, key) DO UPDATE SET result=EXCLUDED.result",
                  (run_id, key, json.dumps(result, default=str)))

    def steps(self, run_id: str) -> dict[str, Any]:
        rows = self._run("SELECT key, result FROM steps WHERE run_id=%s",
                         (run_id,))
        return {r[0]: _load(r[1]) for r in rows}

    def append_signal(self, run_id: str, name: str, payload: Any) -> None:
        self._run(
            "INSERT INTO signals SELECT %s, COALESCE(MAX(idx), -1) + 1, %s, %s"
            " FROM signals WHERE run_id=%s",
            (run_id, name, json.dumps(payload, default=str), run_id))

    def signals(self, run_id: str) -> list[tuple[str, Any]]:
        rows = self._run(
            "SELECT name, payload FROM signals WHERE run_id=%s ORDER BY idx",
            (run_id,))
        return [(r[0], _load(r[1])) for r in rows]

    def record_llm_call(self, run_id: str, model: str,
                        request: dict[str, Any],
                        response: dict[str, Any]) -> None:
        self._run(
            "INSERT INTO llm_calls SELECT %s, COALESCE(MAX(idx), -1) + 1,"
            " %s, %s, %s FROM llm_calls WHERE run_id=%s",
            (run_id, model, json.dumps(request, default=str),
             json.dumps(response, default=str), run_id))

    def llm_calls(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._run(
            "SELECT model, request, response FROM llm_calls WHERE run_id=%s"
            " ORDER BY idx", (run_id,))
        return [{"model": r[0], "request": _load(r[1]),
                 "response": _load(r[2])} for r in rows]

    def all_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._run(
            "SELECT id FROM runs ORDER BY created_at DESC LIMIT %s", (limit,))
        return [self.get_run(r[0]) for r in rows]

    def flush(self) -> None:
        """No-op: every statement is already committed (autocommit)."""

    def close(self) -> None:
        if self._use_pool:
            self._pool.close()
        else:
            self._conn.close()


def _load(value: Any) -> Any:
    """psycopg decodes JSONB for us, so this is identity.

    It is a named function rather than nothing because the temptation is to
    "helpfully" json.loads() the result — which corrupts any value that is
    legitimately a JSON string: a step returning "gathered:kv" comes back as
    the Python str 'gathered:kv', and parsing that raises.
    """
    return value
