"""Sessions and prefix-cache-aware routing.

Two related ideas a stateless framework will never do for you:

**Session affinity.** A conversation has state — history, scratchpad, a
warmed cache. Routing its turns to whichever worker is idle throws that away
every time. ``SessionRouter`` maps a session id to a stable backend and
keeps it there while the session is alive.

**Prefix-cache-aware placement.** Model backends cache the KV of a prompt
prefix. Sending a conversation to a backend that already holds its prefix is
the difference between reprocessing 20k tokens and reprocessing 200 — the
largest cheap cost lever in a chat system. So placement scores backends by
how much of *this* prompt each one already has, and falls back to load only
when nothing matches.

The router is transport-agnostic: "backend" is whatever you are balancing
across — vLLM replicas, provider regions, worker processes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


def token_prefix(messages: Iterable[dict[str, Any]], *,
                 granularity: int = 64) -> tuple[str, ...]:
    """Chunk a conversation into coarse prefix blocks.

    Blocks, not characters: KV caches are reused in page-sized units, and
    comparing whole blocks makes the longest-common-prefix cheap and stable
    against tiny edits deep in the text.
    """
    joined = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        joined.append(f"{message.get('role', 'user')}:{content}")
    text = "\n".join(joined)
    return tuple(text[i:i + granularity]
                 for i in range(0, len(text), granularity))


def common_prefix_len(a: Iterable[str], b: Iterable[str]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


@dataclass
class Backend:
    """One placement target and what it is known to hold."""

    name: str
    in_flight: int = 0
    max_concurrency: int = 64
    cached_prefixes: list[tuple[str, ...]] = field(default_factory=list)
    max_cached: int = 32

    @property
    def load(self) -> float:
        return self.in_flight / max(self.max_concurrency, 1)

    @property
    def saturated(self) -> bool:
        return self.in_flight >= self.max_concurrency

    def note_prefix(self, prefix: tuple[str, ...]) -> None:
        """Record that this backend just processed (and so cached) a prompt."""
        if prefix in self.cached_prefixes:
            self.cached_prefixes.remove(prefix)
        self.cached_prefixes.append(prefix)
        if len(self.cached_prefixes) > self.max_cached:
            self.cached_prefixes.pop(0)      # LRU

    def prefix_score(self, prefix: tuple[str, ...]) -> int:
        """Longest prefix this backend already holds, in blocks."""
        return max((common_prefix_len(prefix, cached)
                    for cached in self.cached_prefixes), default=0)


@dataclass
class Session:
    id: str
    backend: Optional[str] = None
    created_at: float = 0.0
    last_seen: float = 0.0
    turns: int = 0
    state: dict[str, Any] = field(default_factory=dict)


class SessionRouter:
    """Sticky, prefix-aware placement across a set of backends."""

    def __init__(self, backends: Iterable[str | Backend] = (), *,
                 ttl_s: float = 3600.0, granularity: int = 64) -> None:
        self.backends: dict[str, Backend] = {}
        for backend in backends:
            self.add_backend(backend)
        self.sessions: dict[str, Session] = {}
        self.ttl_s = ttl_s
        self.granularity = granularity
        self.hits = 0
        self.misses = 0

    def add_backend(self, backend: str | Backend, **kwargs: Any) -> Backend:
        obj = Backend(backend, **kwargs) if isinstance(backend, str) else backend
        self.backends[obj.name] = obj
        return obj

    # -- sessions -----------------------------------------------------------
    def session(self, session_id: str) -> Session:
        now = time.time()
        existing = self.sessions.get(session_id)
        if existing is None or (now - existing.last_seen) > self.ttl_s:
            existing = Session(id=session_id, created_at=now)
            self.sessions[session_id] = existing
        existing.last_seen = now
        return existing

    def gc(self) -> int:
        now = time.time()
        stale = [sid for sid, s in self.sessions.items()
                 if now - s.last_seen > self.ttl_s]
        for sid in stale:
            del self.sessions[sid]
        return len(stale)

    # -- placement ----------------------------------------------------------
    def route(self, *, session_id: Optional[str] = None,
              messages: Optional[list[dict[str, Any]]] = None) -> Backend:
        """Pick a backend for this turn.

        Order of preference:
        1. the session's sticky backend, unless it is saturated;
        2. the backend holding the longest cached prefix of this prompt;
        3. the least loaded backend.
        """
        if not self.backends:
            raise RuntimeError("SessionRouter has no backends")

        prefix = (token_prefix(messages, granularity=self.granularity)
                  if messages else ())
        session = self.session(session_id) if session_id else None

        if session is not None and session.backend in self.backends:
            sticky = self.backends[session.backend]
            if not sticky.saturated:
                self._admit(sticky, session, prefix, sticky_hit=True)
                return sticky

        chosen: Optional[Backend] = None
        if prefix:
            scored = [(b.prefix_score(prefix), -b.load, b)
                      for b in self.backends.values() if not b.saturated]
            if scored:
                best = max(scored, key=lambda item: (item[0], item[1]))
                if best[0] > 0:
                    chosen = best[2]
                    self.hits += 1

        if chosen is None:
            candidates = [b for b in self.backends.values() if not b.saturated]
            chosen = min(candidates or list(self.backends.values()),
                         key=lambda b: b.load)
            if prefix:
                self.misses += 1

        self._admit(chosen, session, prefix, sticky_hit=False)
        return chosen

    def _admit(self, backend: Backend, session: Optional[Session],
               prefix: tuple[str, ...], *, sticky_hit: bool) -> None:
        if session is not None:
            session.backend = backend.name
            session.turns += 1
        if sticky_hit and prefix:
            # A sticky route is a cache hit by construction: the backend just
            # served this conversation's previous turn.
            if backend.prefix_score(prefix) > 0:
                self.hits += 1
            else:
                self.misses += 1
        if prefix:
            backend.note_prefix(prefix)

    def release(self, backend_name: str) -> None:
        backend = self.backends.get(backend_name)
        if backend is not None and backend.in_flight > 0:
            backend.in_flight -= 1

    def acquire(self, backend_name: str) -> None:
        backend = self.backends.get(backend_name)
        if backend is not None:
            backend.in_flight += 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "sessions": len(self.sessions),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "hits": self.hits, "misses": self.misses,
            "backends": [
                {"name": b.name, "in_flight": b.in_flight,
                 "load": round(b.load, 3), "cached": len(b.cached_prefixes)}
                for b in self.backends.values()],
        }
