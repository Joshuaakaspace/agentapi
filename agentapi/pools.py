"""Admission control: model backends are the real bottleneck.

A Pool models one upstream capacity (an LLM backend, a scraping quota).
Three behaviours a general web framework will not give you:

* **Deadline-aware shedding** — if the caller's deadline cannot survive the
  current queue, reject *now* with a retry hint instead of burning a slot
  to time out later.
* **Per-tenant fair queueing** — waiters are served by deficit round robin
  across tenants, so one customer's batch job cannot starve everyone's
  interactive chat no matter how deep it queues.
* **Adaptive capacity** — upstream 429s shrink the effective concurrency
  and a cooldown holds it there; sustained success walks it back up. Static
  config is always wrong about a shared quota.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Deque, Optional

from .determinism import real_monotonic


class PoolSaturated(Exception):
    """Admission refused. Carries a retry-after hint (seconds)."""

    def __init__(self, message: str, retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


_DEFAULT_TENANT = "__default__"


class Pool:
    def __init__(self, name: str, *, concurrency: int = 64,
                 rpm: Optional[int] = None,
                 max_queue: int = 256,
                 avg_latency_s: float = 2.0,
                 min_concurrency: int = 1,
                 recovery_after_s: float = 30.0) -> None:
        self.name = name
        self.concurrency = concurrency          # configured ceiling
        self.rpm = rpm
        self.max_queue = max_queue
        self.min_concurrency = max(1, min_concurrency)
        self.recovery_after_s = recovery_after_s

        self._effective = concurrency           # adaptive, <= concurrency
        self._in_flight = 0
        self._waiters: dict[str, Deque[asyncio.Future]] = defaultdict(deque)
        self._rotation: list[str] = []          # tenants with pending waiters
        self._cursor = 0
        self._waiting = 0

        self._minute_start = real_monotonic()
        self._minute_count = 0
        self._latency_ewma = avg_latency_s
        self._throttled_until = 0.0
        self._successes_since_throttle = 0

    # -- capacity -----------------------------------------------------------
    @property
    def effective_concurrency(self) -> int:
        """Current ceiling, after any adaptive throttling has been applied."""
        if self._throttled_until and real_monotonic() >= self._throttled_until:
            self._throttled_until = 0.0
        return self._effective

    def report_upstream_429(self, retry_after: Optional[float] = None) -> None:
        """Upstream refused us. Halve effective concurrency and hold it for a
        cooldown; the pool is over its real share of a shared quota."""
        self._effective = max(self.min_concurrency, self._effective // 2)
        self._throttled_until = real_monotonic() + (
            retry_after if retry_after is not None else self.recovery_after_s)
        self._successes_since_throttle = 0

    def report_success(self) -> None:
        """Walk capacity back up after a run of clean calls (additive
        increase to pair with the multiplicative decrease above)."""
        if self._effective >= self.concurrency:
            return
        if self._throttled_until and real_monotonic() < self._throttled_until:
            return
        self._successes_since_throttle += 1
        if self._successes_since_throttle >= self._effective:
            self._effective = min(self.concurrency, self._effective + 1)
            self._successes_since_throttle = 0
            self._dispatch()

    # -- admission ----------------------------------------------------------
    def _estimated_wait(self) -> float:
        """Rough time-to-slot given queue depth and observed latency."""
        capacity = self.effective_concurrency
        if self._in_flight < capacity and self._waiting == 0:
            return 0.0
        waves = (self._waiting + 1) / max(capacity, 1)
        return waves * self._latency_ewma

    def _check_rpm(self) -> None:
        if self.rpm is None:
            return
        now = real_monotonic()
        if now - self._minute_start >= 60:
            self._minute_start = now
            self._minute_count = 0
        if self._minute_count >= self.rpm:
            raise PoolSaturated(
                f"pool {self.name!r}: rpm limit {self.rpm} reached",
                retry_after=60 - (now - self._minute_start))
        self._minute_count += 1

    def _dispatch(self) -> None:
        """Hand free slots to waiters, one tenant at a time, round robin."""
        while self._in_flight < self.effective_concurrency and self._rotation:
            self._cursor %= len(self._rotation)
            tenant = self._rotation[self._cursor]
            queue = self._waiters.get(tenant)
            if not queue:
                self._rotation.pop(self._cursor)
                continue
            future = queue.popleft()
            if not queue:
                self._waiters.pop(tenant, None)
                self._rotation.pop(self._cursor)
            else:
                self._cursor += 1
            if future.done():        # cancelled while queued
                continue
            self._in_flight += 1
            future.set_result(None)

    @asynccontextmanager
    async def acquire(self, *, tenant: Optional[str] = None,
                      deadline_remaining: Optional[float] = None):
        estimated = self._estimated_wait()
        if deadline_remaining is not None and estimated > deadline_remaining:
            raise PoolSaturated(
                f"pool {self.name!r}: estimated wait {estimated:.1f}s exceeds "
                f"deadline budget {deadline_remaining:.1f}s",
                retry_after=estimated)
        if self._waiting >= self.max_queue:
            raise PoolSaturated(
                f"pool {self.name!r}: queue full ({self.max_queue})",
                retry_after=self._latency_ewma)
        self._check_rpm()

        key = tenant or _DEFAULT_TENANT
        if self._in_flight < self.effective_concurrency and not self._rotation:
            self._in_flight += 1            # fast path: uncontended
        else:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            queue = self._waiters[key]
            queue.append(future)
            if key not in self._rotation:
                self._rotation.append(key)
            self._waiting += 1
            try:
                self._dispatch()
                await future
            except BaseException:
                future.cancel()
                raise
            finally:
                self._waiting -= 1

        started = real_monotonic()
        try:
            yield self
        finally:
            elapsed = real_monotonic() - started
            self._latency_ewma = 0.8 * self._latency_ewma + 0.2 * elapsed
            self._in_flight -= 1
            self._dispatch()

    def stats(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "concurrency": self.concurrency,
            "effective_concurrency": self.effective_concurrency,
            "in_flight": self._in_flight,
            "waiting": self._waiting,
            "tenants_queued": len(self._rotation),
            "latency_ewma_s": round(self._latency_ewma, 3),
            "estimated_wait_s": round(self._estimated_wait(), 3),
            "throttled": bool(self._throttled_until
                              and real_monotonic() < self._throttled_until),
        }
