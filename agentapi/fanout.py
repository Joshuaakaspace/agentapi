"""Cross-worker event fanout.

A run's event log lives in the memory of the worker that owns it, so a
client can only *tail* a run by reaching that worker. Any worker can read a
Postgres-journaled run's history, but "attach to this run and follow it
live" needed sticky routing at the load balancer — which is exactly the
constraint a horizontally scaled deployment should not have.

``RedisFanout`` republishes each appended event to a Redis Stream keyed by
run id. A worker that does not own a run can then serve the same resumable
SSE endpoint by replaying the journal and following the stream, so any
worker can answer for any run.

Redis Streams (not pub/sub) because entries persist: a subscriber that
attaches late, or reconnects with a cursor, still gets everything from its
position — the same guarantee the in-process log gives.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from .events import Event, parse_event

logger = logging.getLogger("agentapi.fanout")


class RedisFanout:
    """Publishes run events to Redis Streams and reads them back.

    Requires ``redis`` (``pip install "agentapi[redis]"``).
    """

    def __init__(self, url: str = "redis://localhost:6379/0", *,
                 prefix: str = "agentapi:run:",
                 maxlen: int = 10_000, ttl_s: int = 86_400) -> None:
        try:
            import redis.asyncio as redis_asyncio
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                'RedisFanout requires redis: pip install "agentapi[redis]"'
            ) from exc
        self.url = url
        self.prefix = prefix
        self.maxlen = maxlen
        self.ttl_s = ttl_s
        self._redis = redis_asyncio.from_url(url, decode_responses=True)

    def _key(self, run_id: str) -> str:
        return f"{self.prefix}{run_id}"

    async def publish(self, run_id: str, event: Event) -> None:
        """Mirror one event. Failures are logged, never raised: fanout is a
        convenience for other workers, and losing it must not fail the run
        that produced the event."""
        try:
            key = self._key(run_id)
            await self._redis.xadd(
                key,
                {"seq": str(event.seq),
                 "payload": json.dumps(event.model_dump(by_alias=True),
                                       default=str)},
                maxlen=self.maxlen, approximate=True)
            await self._redis.expire(key, self.ttl_s)
        except Exception:  # noqa: BLE001
            logger.warning("fanout publish failed for run %s", run_id,
                           exc_info=True)

    async def history(self, run_id: str, from_seq: int = 0) -> list[Event]:
        entries = await self._redis.xrange(self._key(run_id))
        events = [parse_event(json.loads(fields["payload"]))
                  for _, fields in entries]
        return [e for e in events if e.seq >= from_seq]

    async def subscribe(self, run_id: str, from_seq: int = 0,
                        *, idle_ms: int = 1000) -> AsyncIterator[Event]:
        """Replay from ``from_seq``, then follow live until a terminal event.

        Mirrors ``EventLog.subscribe`` so a non-owning worker can serve the
        identical resumable SSE contract.
        """
        from .events import TERMINAL_TYPES

        key = self._key(run_id)
        last_id = "0-0"
        seen = from_seq
        while True:
            entries = await self._redis.xread({key: last_id}, count=100,
                                              block=idle_ms)
            if not entries:
                continue
            for _stream, records in entries:
                for entry_id, fields in records:
                    last_id = entry_id
                    event = parse_event(json.loads(fields["payload"]))
                    if event.seq < seen:
                        continue
                    seen = event.seq + 1
                    yield event
                    if event.type in TERMINAL_TYPES:
                        return

    async def close(self) -> None:
        await self._redis.aclose()


async def attach(fanout: Any, backend: Any, run_id: str,
                 from_seq: int = 0) -> AsyncIterator[Event]:
    """Serve a run from a worker that does not own it.

    Journal first (authoritative, survives Redis eviction), then the stream
    for anything newer. De-duplicates on ``seq`` so the seam between the two
    sources is invisible to the client.
    """
    cursor = from_seq
    if backend is not None:
        for payload in backend.events(run_id):
            event = parse_event(payload)
            if event.seq >= cursor:
                cursor = event.seq + 1
                yield event
                if event.type in ("done", "error"):
                    return
    async for event in fanout.subscribe(run_id, from_seq=cursor):
        yield event
