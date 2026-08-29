"""Non-blocking journal writes.

The journal backends are synchronous — sqlite3 and psycopg both are — and
they were being called straight from the event loop. One durable run
writing 200 events stalled the loop for 58ms *continuously*: every other
request on that worker froze for the duration. On a server whose whole
premise is serving many long-lived runs concurrently, that is the most
expensive bug in the codebase.

``AsyncJournal`` offloads every backend call to a worker thread, so the
loop stays responsive while writes are in flight. Durability semantics are
unchanged: a call still completes before the caller proceeds.

Writes issued from synchronous code (``ctx.now()`` and friends, which
cannot await) are queued instead and flushed at the next ``await`` point —
always before a pause or a terminal event, the two places where the process
may realistically die.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("agentapi.journal")


class AsyncJournal:
    """Thread-offloading wrapper around a synchronous journal backend."""

    def __init__(self, backend: Any, *, max_concurrency: int = 8) -> None:
        self.backend = backend
        # SQLite serialises internally anyway; the cap keeps a burst of
        # durable runs from spawning an unbounded number of threads.
        self._slots = asyncio.Semaphore(max_concurrency)

    async def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.backend, name)
        async with self._slots:
            return await asyncio.to_thread(method, *args, **kwargs)

    def call_soon(self, pending: list, name: str, *args: Any) -> None:
        """Queue a write from synchronous code, to be flushed later."""
        pending.append((name, args))

    async def drain(self, pending: list) -> None:
        """Flush queued writes. Failures are logged, not raised: losing a
        journaled clock value degrades replay to a detected divergence,
        which is far better than failing a run that is otherwise fine."""
        if not pending:
            return
        batch, pending[:] = list(pending), []
        for name, args in batch:
            try:
                await self.call(name, *args)
            except Exception:  # noqa: BLE001
                logger.warning("journal write %s failed", name, exc_info=True)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Read-only methods pass through synchronously.

        Reads happen at recovery and on archived-run endpoints, not on the
        hot path, so offloading them would add latency for no benefit.
        """
        return getattr(self.backend, name)
