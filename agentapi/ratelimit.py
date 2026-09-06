"""Per-principal rate limiting on run creation.

Pools cap what goes *upstream* to a model. Nothing capped how fast one
caller could create runs — so a single client could still exhaust the
process, and with auth in place there is finally a stable identity to
limit against.

A token bucket per key, where the key is the tenant when there is one and
the principal otherwise: teams share a budget, anonymous callers are
limited individually. Refused requests get 429 with a ``Retry-After``
computed from the actual deficit, not a fixed guess.

    app = AgentAPI(rate_limit=RateLimit(per_minute=60, burst=10))
    @app.run("/expensive", rate_limit=RateLimit(per_minute=5))
"""
from __future__ import annotations

from dataclasses import dataclass

from .determinism import real_monotonic


class RateLimited(Exception):
    """Caller exceeded their allowance."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimit:
    """Token bucket keyed by tenant (or principal when there is no tenant).

    ``per_minute`` is the sustained rate; ``burst`` is how much can be spent
    at once (defaults to one minute's worth, so a client may spend its
    allowance whenever it likes rather than being forced into a metronome).
    """

    def __init__(self, *, per_minute: float = 60.0,
                 burst: float | None = None,
                 max_keys: int = 10_000) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.per_minute = per_minute
        self.rate = per_minute / 60.0
        self.burst = float(burst) if burst is not None else float(per_minute)
        self.max_keys = max_keys
        self._buckets: dict[str, _Bucket] = {}

    def key_for(self, principal: object) -> str:
        tenant = getattr(principal, "tenant", None)
        return f"t:{tenant}" if tenant else f"p:{getattr(principal, 'id', '?')}"

    def check(self, key: str, cost: float = 1.0) -> None:
        """Spend ``cost`` for ``key``, or raise RateLimited."""
        now = real_monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self.max_keys:
                self._evict(now)
            bucket = _Bucket(tokens=self.burst, updated=now)
            self._buckets[key] = bucket

        bucket.tokens = min(self.burst,
                            bucket.tokens + (now - bucket.updated) * self.rate)
        bucket.updated = now

        if bucket.tokens < cost:
            deficit = cost - bucket.tokens
            raise RateLimited(
                f"rate limit exceeded: {self.per_minute:g}/min",
                retry_after=deficit / self.rate)
        bucket.tokens -= cost

    def _evict(self, now: float) -> None:
        """Drop the fullest buckets: a full bucket is an idle caller, so
        forgetting it costs them nothing."""
        victims = sorted(self._buckets.items(),
                         key=lambda item: -item[1].tokens)[:len(self._buckets) // 4 or 1]
        for key, _ in victims:
            del self._buckets[key]

    def peek(self, key: str) -> float:
        bucket = self._buckets.get(key)
        if bucket is None:
            return self.burst
        return min(self.burst,
                   bucket.tokens + (real_monotonic() - bucket.updated) * self.rate)

    def stats(self) -> dict[str, object]:
        return {"per_minute": self.per_minute, "burst": self.burst,
                "tracked_keys": len(self._buckets)}
