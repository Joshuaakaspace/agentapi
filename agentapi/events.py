"""Typed run events and the append-only event log.

The event log is the heart of agentapi: every run writes its output as a
sequence of events, and every transport (SSE, WebSocket, MCP, webhooks,
polling) is just a subscriber reading the log from a cursor. This is what
makes streams resumable and runs watchable by more than one client.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from .determinism import real_time


class Event(BaseModel):
    """Base class for all run events.

    Subclass it (or use the built-ins below) and ``yield`` instances from a
    run handler. ``seq`` and ``ts`` are stamped by the event log on append.
    """

    type: str = "event"
    seq: int = -1          # stamped by the log; position in the run's stream
    ts: float = 0.0        # stamped by the log; unix seconds

    def model_post_init(self, __context: Any) -> None:
        if type(self) is not Event and self.type == "event":
            self.type = type(self).__name__.lower()


class Token(Event):
    """A streamed text delta."""
    type: str = "token"
    text: str = ""


class Message(Event):
    """A complete message (e.g. an assistant turn)."""
    type: str = "message"
    role: str = "assistant"
    content: Any = None


class ToolCall(Event):
    type: str = "tool_call"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class ToolResult(Event):
    type: str = "tool_result"
    name: str
    result: Any = None
    call_id: str | None = None
    error: str | None = None


class StateDelta(Event):
    """Arbitrary structured progress (plans, scratchpads, partial objects)."""
    type: str = "state_delta"
    data: dict[str, Any] = Field(default_factory=dict)


class Paused(Event):
    """Run is waiting on an external signal (human-in-the-loop)."""
    type: str = "paused"
    signal: str
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")

    model_config = {"populate_by_name": True}


class Resumed(Event):
    type: str = "resumed"
    signal: str
    payload: Any = None


class Done(Event):
    """Terminal success event."""
    type: str = "done"
    result: Any = None
    usage: dict[str, Any] = Field(default_factory=dict)


class RunError(Event):
    """Terminal failure event."""
    type: str = "error"
    error: str = ""
    kind: str = "error"    # "error" | "cancelled" | "budget" | "deadline"


TERMINAL_TYPES = {"done", "error"}

_EVENT_TYPES: dict[str, type[Event]] = {
    "token": Token, "message": Message, "tool_call": ToolCall,
    "tool_result": ToolResult, "state_delta": StateDelta, "paused": Paused,
    "resumed": Resumed, "done": Done, "error": RunError,
}


class GenericEvent(Event):
    """Round-trips unknown (user-defined) event types with fields intact."""
    model_config = {"extra": "allow"}


def parse_event(data: dict[str, Any]) -> Event:
    """Deserialize a persisted event into its concrete class so replay
    fingerprints and field access behave like the original."""
    cls = _EVENT_TYPES.get(str(data.get("type", "")), GenericEvent)
    return cls.model_validate(data)


class EventLog:
    """Append-only, cursor-addressable event log for one run.

    In-memory implementation (durability tier 1: "resumable"). The interface
    — ``append``, ``read``, ``subscribe(from_seq)`` — is what a Redis or
    Postgres backend would implement for tiers 1+/2.
    """

    def __init__(self, max_events: int = 100_000) -> None:
        self._events: list[Event] = []
        self._max = max_events
        self._changed = asyncio.Condition()
        self._closed = False

    @property
    def next_seq(self) -> int:
        return len(self._events)

    @property
    def closed(self) -> bool:
        return self._closed

    def restore(self, events: list[Event]) -> None:
        """Load persisted history (recovery). No subscribers exist yet, so
        no notification is needed; a terminal event re-closes the log."""
        self._events = list(events)
        self._closed = any(e.type in TERMINAL_TYPES for e in self._events)

    async def append(self, event: Event) -> Event:
        if self._closed:
            raise RuntimeError("event log is closed")
        if len(self._events) >= self._max:
            raise RuntimeError(f"event log overflow (> {self._max} events)")
        event.seq = len(self._events)
        event.ts = real_time()
        self._events.append(event)
        if event.type in TERMINAL_TYPES:
            self._closed = True
        async with self._changed:
            self._changed.notify_all()
        return event

    def read(self, from_seq: int = 0, limit: int | None = None) -> list[Event]:
        """Read already-appended events starting at ``from_seq``."""
        end = None if limit is None else from_seq + limit
        return self._events[from_seq:end]

    async def subscribe(self, from_seq: int = 0) -> AsyncIterator[Event]:
        """Yield events from ``from_seq``, replaying history then following
        live appends until the log closes. Safe for any number of concurrent
        subscribers — this is what makes late-join and reconnect free."""
        cursor = max(0, from_seq)
        while True:
            while cursor < len(self._events):
                event = self._events[cursor]
                cursor += 1
                yield event
            if self._closed:
                return
            async with self._changed:
                if cursor >= len(self._events) and not self._closed:
                    await self._changed.wait()
