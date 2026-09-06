"""Run: the first-class unit of work.

A run owns an event log and a context; transports merely attach to it.
Client disconnects never cancel a run unless the route's disconnect policy
says so — the run belongs to the RunManager, not to a TCP connection.
"""
from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from typing import Any

from .context import (
    BudgetExceeded,
    DeadlineExceeded,
    RunCancelled,
    RunContext,
    RunDraining,
    _current,
)
from .determinism import NondeterminismError, guarding, real_time
from .events import Done, Event, EventLog, Paused, Resumed, RunError


def _event_fingerprint(event: Event) -> str:
    """Identity of an event ignoring log-assigned seq/ts, for replay dedupe."""
    payload = event.model_dump(by_alias=True)
    payload.pop("seq", None)
    payload.pop("ts", None)
    import json as _json
    return _json.dumps(payload, sort_keys=True, default=str)


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Run:
    def __init__(self, run_id: str, route: str, ctx: RunContext,
                 log: EventLog) -> None:
        self.id = run_id
        self.route = route
        self.ctx = ctx
        self.log = log
        self.status = RunStatus.QUEUED
        self.created_at = real_time()
        self.finished_at: float | None = None
        self.attached = 0            # live transport subscribers
        self.owner: tuple[str, str | None] | None = None  # (id, tenant)
        self.task: asyncio.Task[None] | None = None
        self._hooks: list[Any] = []

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "route": self.route,
            "status": self.status.value,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "events": self.log.next_seq,
            "attached": self.attached,
            "usage": self.ctx.usage.as_dict(),
            "tenant": self.ctx.tenant,
            "metadata": self.ctx.metadata,
        }

    async def events(self, from_seq: int = 0) -> AsyncIterator[Event]:
        self.attached += 1
        try:
            async for event in self.log.subscribe(from_seq):
                yield event
        finally:
            self.attached -= 1


class RunManager:
    """Owns every run in the process. Runs outlive connections."""

    def __init__(self, *, retention_s: float = 3600.0,
                 backend: Any = None, redactor: Any = None,
                 fanout: Any = None) -> None:
        self.runs: dict[str, Run] = {}
        self._idempotency: dict[str, str] = {}   # Idempotency-Key -> run_id
        self.retention_s = retention_s
        self.backend = backend                   # durable journal (tier 2)
        from .redaction import NullRedactor
        self.redactor = redactor or NullRedactor()
        self.fanout = fanout                     # cross-worker mirror

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def by_idempotency_key(self, key: str) -> Run | None:
        run_id = self._idempotency.get(key)
        return self.runs.get(run_id) if run_id else None

    def start(self, route: str, handler: Callable[..., Any],
              kwargs: dict[str, Any], *,
              ctx: RunContext | None = None,
              idempotency_key: str | None = None,
              hooks: Any = None,
              durable: bool = False,
              run_id: str | None = None,
              replay_events: list[Event] | None = None,
              determinism: str = "off",
              owner: tuple | None = None) -> Run:
        recovering = run_id is not None
        run_id = run_id or f"run_{uuid.uuid4().hex[:20]}"
        context = ctx or RunContext(run_id)
        context.run_id = run_id
        log = EventLog()
        run = Run(run_id, route, context, log)
        run.durable = durable
        run.determinism = determinism if durable else "off"
        run.owner = owner
        self.runs[run_id] = run
        if idempotency_key:
            self._idempotency[idempotency_key] = run_id

        backend = self.backend if durable else None
        if backend is not None:
            if recovering:
                log.restore(replay_events or [])
            else:
                backend.create_run(run_id, route, kwargs,
                                   tenant=context.tenant,
                                   metadata=context.metadata)
            context._step_commit = (
                lambda key, result:
                backend.save_step(run_id, key, self.redactor.step(result)))
            redactor = self.redactor

            def _record_llm(model, request, response):
                clean_request, clean_response = redactor.llm_call(
                    request, response)
                backend.record_llm_call(run_id, model, clean_request,
                                        clean_response)

            context._llm_record = _record_llm

        # Replay dedupe: a recovering execution re-emits a subsequence of the
        # restored history. Match each emitted event forward against history
        # and return the recorded one instead of appending a duplicate.
        replay = [(_event_fingerprint(e), e) for e in (replay_events or [])]
        replay_cursor = [0]

        async def emit(event: Event) -> Event:
            if replay_cursor[0] < len(replay):
                fingerprint = _event_fingerprint(event)
                for index in range(replay_cursor[0], len(replay)):
                    if replay[index][0] == fingerprint:
                        replay_cursor[0] = index + 1
                        self._track_pause(run, replay[index][1])
                        return replay[index][1]
                # Nothing ahead matches, yet history still has events the
                # replay should have reproduced: the handler took a
                # different path this time. (An event lost to the crash
                # would sit at the tail, where the cursor is already spent
                # and this branch is not reached.)
                self._diverged(run, event, replay[replay_cursor[0]][1])
            appended = await log.append(event)
            self._track_pause(run, appended)
            if backend is not None:
                # Redact on the way in: what never reaches the journal
                # cannot leak from it. Attached clients still see the real
                # event — they are the caller who supplied the content.
                backend.append_event(
                    run_id, appended.seq,
                    self.redactor.event(appended.model_dump(by_alias=True)))
                if isinstance(appended, Paused):
                    backend.update_status(run_id, "paused")
                elif isinstance(appended, Resumed):
                    backend.update_status(run_id, "running")
            if self.fanout is not None:
                await self.fanout.publish(run_id, appended)
            if hooks is not None:
                await hooks.fire("on_event", run, appended)
            return appended

        context._emit_cb = emit
        run.task = asyncio.create_task(
            self._execute(run, handler, kwargs, hooks), name=f"agentapi:{run_id}")
        return run

    async def _finish(self, run: Run, event: Event) -> None:
        """Append a terminal event directly, bypassing ctx.check() — the run
        is already stopping, so cancellation/drain must not block the end."""
        try:
            await asyncio.shield(run.log.append(event))
        except (RuntimeError, asyncio.CancelledError):
            pass

    def _diverged(self, run: Run, emitted: Event, expected: Event) -> None:
        """A replayed run emitted an event its recorded history does not
        contain — proof that something outside the journal changed."""
        message = (
            f"replay of run {run.id} diverged: emitted {emitted.type!r} where "
            f"the journal recorded {expected.type!r} (seq {expected.seq}). "
            f"Handler code is not reproducible — check for unjournaled "
            f"clock, randomness or I/O outside a @step."
        )
        mode = getattr(run, "determinism", "off")
        if mode == "raise":
            raise NondeterminismError(message)
        if mode == "warn":
            import logging
            logging.getLogger("agentapi.determinism").warning(message)

    def _track_pause(self, run: Run, event: Event) -> None:
        if event.type == "paused" and run.status == RunStatus.RUNNING:
            run.status = RunStatus.PAUSED
        elif event.type == "resumed" and run.status == RunStatus.PAUSED:
            run.status = RunStatus.RUNNING

    async def _execute(self, run: Run, handler: Callable[..., Any],
                       kwargs: dict[str, Any], hooks: Any) -> None:
        run.status = RunStatus.RUNNING
        token = _current.set(run.ctx)
        mode = getattr(run, "determinism", "off")
        guard_cm = None
        try:
            if hooks is not None:
                await hooks.fire("on_run_start", run)
            result: Any = None
            guard_cm = guarding(mode, run.id)
            guard_cm.__enter__()
            if inspect.isasyncgenfunction(handler):
                async for event in handler(**kwargs):
                    run.ctx.check()
                    if isinstance(event, Done):
                        result = event.result
                        if not event.usage:
                            event.usage = run.ctx.usage.as_dict()
                        await run.ctx.emit(event)
                        break
                    if not isinstance(event, Event):
                        raise TypeError(
                            f"run handlers must yield Event instances, got "
                            f"{type(event).__name__}")
                    await run.ctx.emit(event)
                else:
                    result = None
            else:
                result = await handler(**kwargs)
            guard_cm.__exit__(None, None, None)
            guard_cm = None
            if not run.log.closed:
                await run.ctx.emit(Done(result=_plain(result),
                                        usage=run.ctx.usage.as_dict()))
            run.status = RunStatus.COMPLETED
        except RunDraining:
            # Graceful stop: the client is gone and the route asked to drain.
            # Whatever completed is a real result, so end on Done, not error.
            guard_cm = _exit_guard(guard_cm)
            if not run.log.closed:
                await self._finish(run, Done(result=_plain(result),
                                             usage=run.ctx.usage.as_dict()))
            run.status = RunStatus.COMPLETED
        except NondeterminismError as exc:
            run.status = RunStatus.FAILED
            await self._fail(run, str(exc), kind="nondeterminism")
        except (RunCancelled, asyncio.CancelledError):
            run.status = RunStatus.CANCELLED
            await self._fail(run, "run cancelled", kind="cancelled")
        except DeadlineExceeded as exc:
            run.status = RunStatus.FAILED
            await self._fail(run, str(exc), kind="deadline")
        except BudgetExceeded as exc:
            run.status = RunStatus.FAILED
            await self._fail(run, str(exc), kind="budget")
        except Exception as exc:  # noqa: BLE001 - terminal event carries it
            run.status = RunStatus.FAILED
            await self._fail(run, f"{type(exc).__name__}: {exc}", kind="error")
        finally:
            try:
                if guard_cm is not None:
                    guard_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - never mask the real outcome
                pass
            run.finished_at = real_time()
            if getattr(run, "durable", False) and self.backend is not None:
                self.backend.update_status(run.id, run.status.value,
                                           finished=True)
                self.backend.flush()   # a finished run is always durable
            _current.reset(token)
            if hooks is not None:
                await hooks.fire("on_run_end", run)

    async def _fail(self, run: Run, message: str, *, kind: str) -> None:
        if not run.log.closed:
            try:
                await asyncio.shield(
                    run.log.append(RunError(error=message, kind=kind)))
            except (RuntimeError, asyncio.CancelledError):
                pass

    def cancel(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status not in (RunStatus.QUEUED, RunStatus.RUNNING,
                                             RunStatus.PAUSED):
            return False
        run.ctx.cancel()
        # Interrupt whatever the handler is awaiting (an LLM call, a sleep);
        # _execute converts the CancelledError into a terminal "cancelled"
        # event so attached clients see a clean end-of-stream.
        if run.task is not None and not run.task.done():
            run.task.cancel()
        return True

    def drain(self, run_id: str) -> bool:
        """Ask a run to stop at its next step boundary."""
        run = self.runs.get(run_id)
        if run is None or run.status not in (RunStatus.QUEUED,
                                             RunStatus.RUNNING):
            return False
        run.ctx.drain()
        return True

    def signal(self, run_id: str, signal: str, payload: Any) -> bool:
        run = self.runs.get(run_id)
        if run is None:
            return False
        if getattr(run, "durable", False) and self.backend is not None:
            self.backend.append_signal(run_id, signal, payload)
        return run.ctx.deliver_signal(signal, payload)

    def gc(self) -> int:
        """Drop finished runs past the retention window."""
        now = time.time()
        stale = [rid for rid, run in self.runs.items()
                 if run.finished_at and now - run.finished_at > self.retention_s]
        for rid in stale:
            del self.runs[rid]
        self._idempotency = {k: v for k, v in self._idempotency.items()
                             if v in self.runs}
        return len(stale)


def _exit_guard(guard_cm: Any) -> None:
    if guard_cm is not None:
        try:
            guard_cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
    return None


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value
