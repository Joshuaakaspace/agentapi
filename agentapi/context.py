"""RunContext: the ambient context every run executes under.

Carries deadline, budget, usage accounting, cancellation, tenancy and the
pause/signal channel. Propagated implicitly through ``contextvars`` so LLM
calls, tools, steps and sub-agents inherit it without plumbing.

Access it anywhere inside a run via ``agentapi.ctx`` (a proxy).
"""
from __future__ import annotations

import asyncio
import contextvars
import math
import time
import uuid as _uuid
import random as _random
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from .events import Event, Paused, Resumed


class BudgetExceeded(Exception):
    """Raised at the call site the moment a budget would be exceeded."""


class DeadlineExceeded(Exception):
    """Raised when the run's (or an enclosing scope's) deadline has passed."""


class RunCancelled(Exception):
    """Raised inside a handler when the run is cancelled externally."""


@dataclass
class Usage:
    """Accumulated metered usage for a run (and any scope within it)."""
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0

    def add(self, *, input_tokens: int = 0, output_tokens: int = 0,
            usd: float = 0.0, llm_calls: int = 0, tool_calls: int = 0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += usd
        self.llm_calls += llm_calls
        self.tool_calls += tool_calls

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "usd": round(self.usd, 6),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }


@dataclass
class _Scope:
    """One budget/deadline frame. Scopes nest; children only shrink limits."""
    deadline: float = math.inf            # absolute monotonic time
    max_usd: float = math.inf
    max_tokens: float = math.inf
    usage: Usage = field(default_factory=Usage)
    parent: Optional["_Scope"] = None

    def charge(self, *, input_tokens: int = 0, output_tokens: int = 0,
               usd: float = 0.0, **counts: int) -> None:
        scope: Optional[_Scope] = self
        while scope is not None:
            scope.usage.add(input_tokens=input_tokens,
                            output_tokens=output_tokens, usd=usd, **counts)
            if scope.usage.usd > scope.max_usd:
                raise BudgetExceeded(
                    f"budget exceeded: ${scope.usage.usd:.4f} > ${scope.max_usd:.4f}")
            total = scope.usage.input_tokens + scope.usage.output_tokens
            if total > scope.max_tokens:
                raise BudgetExceeded(
                    f"token budget exceeded: {total} > {int(scope.max_tokens)}")
            scope = scope.parent


def parse_duration(value: str | float | int) -> float:
    """'30s' / '5m' / '2h' / '1d' / bare seconds -> seconds (float)."""
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip().lower()
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    for suffix in ("ms", "s", "m", "h", "d"):
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * units[suffix]
    return float(value)


class RunContext:
    """Everything a run knows about itself while executing."""

    def __init__(self, run_id: str, *, tenant: Optional[str] = None,
                 deadline_s: Optional[float] = None,
                 max_usd: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 metadata: Optional[dict[str, Any]] = None) -> None:
        self.run_id = run_id
        self.tenant = tenant
        self.metadata = metadata or {}
        deadline = math.inf if deadline_s is None else time.monotonic() + deadline_s
        self._scope = _Scope(
            deadline=deadline,
            max_usd=math.inf if max_usd is None else max_usd,
            max_tokens=math.inf if max_tokens is None else max_tokens,
        )
        self._cancelled = False
        self._signals: dict[str, asyncio.Queue[Any]] = {}
        self._emit_cb = None          # wired by the runtime
        self._llm = None              # wired by the app (LLM client facade)
        self._step_journal: dict[str, Any] = {}
        self._step_commit = None      # durable backend hook, wired by the app
        self._rng = _random.Random(run_id)  # deterministic per run

    # -- identity / determinism helpers ------------------------------------
    def now(self) -> float:
        return time.time()

    def uuid(self) -> str:
        return str(_uuid.UUID(int=self._rng.getrandbits(128), version=4))

    def random(self) -> float:
        return self._rng.random()

    # -- deadline / budget --------------------------------------------------
    @property
    def deadline_remaining(self) -> float:
        return self._scope.deadline - time.monotonic()

    def check(self) -> None:
        """Raise if the run is cancelled or out of time. Called by the
        runtime around every event append, LLM call and tool call."""
        if self._cancelled:
            raise RunCancelled(f"run {self.run_id} was cancelled")
        if self.deadline_remaining <= 0:
            raise DeadlineExceeded(f"deadline exceeded for run {self.run_id}")

    @property
    def usage(self) -> Usage:
        # Root scope usage == whole-run usage.
        scope = self._scope
        while scope.parent is not None:
            scope = scope.parent
        return scope.usage

    def charge(self, **kwargs: Any) -> None:
        """Record metered usage against every enclosing scope, raising
        ``BudgetExceeded`` at the call site if any scope's limit is hit."""
        self.check()
        self._scope.charge(**kwargs)

    @asynccontextmanager
    async def budget(self, *, usd: Optional[float] = None,
                     tokens: Optional[int] = None,
                     deadline: Optional[str | float] = None):
        """Open a nested budget/deadline scope. Limits only shrink: a child
        deadline can never outlive its parent, and charges roll up so parent
        budgets see child spend."""
        parent = self._scope
        child_deadline = parent.deadline
        if deadline is not None:
            child_deadline = min(child_deadline,
                                 time.monotonic() + parse_duration(deadline))
        self._scope = _Scope(
            deadline=child_deadline,
            max_usd=math.inf if usd is None else usd,
            max_tokens=math.inf if tokens is None else tokens,
            parent=parent,
        )
        try:
            yield self._scope.usage
        finally:
            self._scope = parent

    # -- cancellation -------------------------------------------------------
    def cancel(self) -> None:
        self._cancelled = True
        # Wake any pending pause() so cancellation isn't stuck behind a 24h wait.
        for queue in self._signals.values():
            queue.put_nowait(RunCancelled(f"run {self.run_id} was cancelled"))

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # -- events -------------------------------------------------------------
    async def emit(self, event: Event) -> Event:
        """Emit an event into the run's log from anywhere (tools, hooks)."""
        self.check()
        if self._emit_cb is None:
            raise RuntimeError("context is not attached to a live run")
        return await self._emit_cb(event)

    # -- pause / signal (human-in-the-loop) ---------------------------------
    async def pause(self, signal: str, *, schema: Any = None,
                    timeout: Optional[str | float] = None) -> Any:
        """Suspend until ``POST /runs/{id}/signals/{signal}`` delivers a
        payload. Emits ``Paused``/``Resumed`` events so attached clients see
        the state change."""
        self.check()
        schema_dict = None
        if schema is not None:
            schema_dict = (schema.model_json_schema()
                           if hasattr(schema, "model_json_schema") else schema)
        queue = self._signals.setdefault(signal, asyncio.Queue())
        await self.emit(Paused(signal=signal, schema=schema_dict))
        timeout_s = None if timeout is None else parse_duration(timeout)
        timeout_s = min(
            timeout_s if timeout_s is not None else math.inf,
            max(self.deadline_remaining, 0) if self.deadline_remaining != math.inf else math.inf,
        )
        try:
            if timeout_s == math.inf:
                payload = await queue.get()
            else:
                payload = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise DeadlineExceeded(f"timed out waiting for signal {signal!r}")
        if isinstance(payload, RunCancelled):
            raise payload
        if schema is not None and hasattr(schema, "model_validate"):
            payload = schema.model_validate(payload)
        await self.emit(Resumed(signal=signal, payload=(
            payload.model_dump() if hasattr(payload, "model_dump") else payload)))
        return payload

    def deliver_signal(self, signal: str, payload: Any) -> bool:
        """Deliver an external signal. Returns False if nothing is waiting
        and nothing has ever registered interest — callers may still queue."""
        queue = self._signals.setdefault(signal, asyncio.Queue())
        queue.put_nowait(payload)
        return True

    @property
    def llm(self):
        if self._llm is None:
            raise RuntimeError(
                "no LLM client configured; pass llm=... to AgentAPI() "
                "or set app.llm")
        return self._llm


_current: contextvars.ContextVar[Optional[RunContext]] = contextvars.ContextVar(
    "agentapi_ctx", default=None)


def get_ctx() -> RunContext:
    context = _current.get()
    if context is None:
        raise RuntimeError("agentapi.ctx accessed outside a run")
    return context


class _CtxProxy:
    """Module-level ``ctx`` that always points at the current run's context."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_ctx(), name)

    def __repr__(self) -> str:
        context = _current.get()
        return f"<ctx {context.run_id}>" if context else "<ctx unbound>"


ctx = _CtxProxy()
