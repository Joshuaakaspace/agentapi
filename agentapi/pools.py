"""Admission control: model backends are the real bottleneck.

A Pool models one upstream capacity (an LLM backend, a scraping quota).
Handlers reserve from pools before running. Key behaviour FastAPI lacks:
**deadline-aware shedding** — if the caller's deadline cannot survive the
current queue, reject *now* with retry-after instead of burning a slot to
time out later.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional


class PoolSaturated(Exception):
    """Admission refused. Carries a retry-after hint (seconds)."""

    def __init__(self, message: str, retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class Pool:
    def __init__(self, name: str, *, concurrency: int = 64,
                 rpm: Optional[int] = None,
                 max_queue: int = 256,
                 avg_latency_s: float = 2.0) -> None:
        self.name = name
        self.concurrency = concurrency
        self.rpm = rpm
        self.max_queue = max_queue
        self._sem = asyncio.Semaphore(concurrency)
        self._waiting = 0
        # naive rolling-minute request accounting for the rpm cap
        self._minute_start = time.monotonic()
        self._minute_count = 0
        # observed latency EWMA feeds the deadline-aware admission estimate
        self._latency_ewma = avg_latency_s

    def _estimated_wait(self) -> float:
        """Rough time-to-slot given queue depth and observed latency."""
        if self._sem._value > 0 and self._waiting == 0:  # noqa: SLF001
            return 0.0
        waves = (self._waiting + 1) / max(self.concurrency, 1)
        return waves * self._latency_ewma

    def _check_rpm(self) -> None:
        if self.rpm is None:
            return
        now = time.monotonic()
        if now - self._minute_start >= 60:
            self._minute_start = now
            self._minute_count = 0
        if self._minute_count >= self.rpm:
            raise PoolSaturated(
                f"pool {self.name!r}: rpm limit {self.rpm} reached",
                retry_after=60 - (now - self._minute_start))
        self._minute_count += 1

    @asynccontextmanager
    async def acquire(self, *, deadline_remaining: Optional[float] = None):
        estimated = self._estimated_wait()
        if deadline_remaining is not None and estimated > deadline_remaining:
            # The single highest-leverage behaviour under load: this request
            # cannot make its deadline, so shed it before it costs anything.
            raise PoolSaturated(
                f"pool {self.name!r}: estimated wait {estimated:.1f}s exceeds "
                f"deadline budget {deadline_remaining:.1f}s",
                retry_after=estimated)
        if self._waiting >= self.max_queue:
            raise PoolSaturated(
                f"pool {self.name!r}: queue full ({self.max_queue})",
                retry_after=self._latency_ewma)
        self._check_rpm()
        self._waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self._waiting -= 1
        started = time.monotonic()
        try:
            yield self
        finally:
            elapsed = time.monotonic() - started
            self._latency_ewma = 0.8 * self._latency_ewma + 0.2 * elapsed
            self._sem.release()

    def stats(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "concurrency": self.concurrency,
            "in_flight": self.concurrency - self._sem._value,  # noqa: SLF001
            "waiting": self._waiting,
            "latency_ewma_s": round(self._latency_ewma, 3),
            "estimated_wait_s": round(self._estimated_wait(), 3),
        }
