"""Determinism checking for replay-based recovery.

Replay recovery is only trustworthy if a handler, re-run from the top,
takes the same path it took the first time. A stray ``time.time()`` or
``random.choice()`` in handler code silently corrupts the journal — the
failure mode that kills adoption of this pattern (DESIGN.md §3.3). So the
runtime detects it instead of documenting it.

Two complementary halves:

**Proactive** — ``time``/``random``/``uuid`` module functions are wrapped
so that calling them *inside a durable run handler, outside a step* raises
``NondeterminismError`` at the offending line. The wrappers are inert
everywhere else: they act only when a contextvar marks the current task as
running durable handler code, so other tasks, threads and libraries are
unaffected.

**Reactive** — during recovery, an emitted event that matches nothing in
the remaining recorded history proves the replay diverged, whatever caused
it (``datetime.now()``, dict ordering, an unjournaled network read). See
``RunManager.start``.

Framework internals call the ``real_*`` helpers below so the runtime's own
timestamps never trip its own guard.
"""
from __future__ import annotations

import functools
import logging
import random
import time
import uuid
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional

logger = logging.getLogger("agentapi.determinism")


class NondeterminismError(RuntimeError):
    """Handler code did something a replay could not reproduce."""


# Originals captured before any patching, so framework internals (event
# timestamps, run ids) bypass the guard entirely.
real_time: Callable[[], float] = time.time
real_monotonic: Callable[[], float] = time.monotonic
real_uuid4 = uuid.uuid4

MODES = ("raise", "warn", "off")


class _Guard:
    """Marks the current task as executing durable handler code."""

    __slots__ = ("mode", "run_id", "depth", "violations")

    def __init__(self, mode: str, run_id: str) -> None:
        self.mode = mode
        self.run_id = run_id
        self.depth = 0          # >0 while inside a step / ctx accessor
        self.violations: list[str] = []

    def violation(self, what: str) -> None:
        message = (
            f"nondeterministic call {what}() in durable run {self.run_id}. "
            f"A replay would produce a different value and corrupt the "
            f"journal. Move it inside a @step, or use ctx.now()/ctx.uuid()/"
            f"ctx.random(), which are journaled and replay identically."
        )
        self.violations.append(what)
        if self.mode == "raise":
            raise NondeterminismError(message)
        if self.mode == "warn":
            # Reporting is itself nondeterministic — logging stamps records
            # with time.time() — so suppress the guard while we report, or
            # the first violation recurses until the stack gives out.
            self.depth += 1
            try:
                logger.warning(message)
                warnings.warn(message, RuntimeWarning, stacklevel=3)
            finally:
                self.depth -= 1


_guard: ContextVar[Optional[_Guard]] = ContextVar("agentapi_determinism",
                                                  default=None)

# (module, attribute) pairs wrapped by install(). time.monotonic is
# deliberately absent: it is the framework's own deadline clock and is not
# a value handlers embed in output.
_TARGETS = [
    (time, "time"), (time, "time_ns"),
    (random, "random"), (random, "randint"), (random, "uniform"),
    (random, "choice"), (random, "choices"), (random, "shuffle"),
    (random, "sample"), (random, "getrandbits"),
    (uuid, "uuid1"), (uuid, "uuid4"),
]

_installed = False
_originals: dict[tuple[Any, str], Any] = {}


def install() -> None:
    """Wrap the nondeterministic entry points. Idempotent; safe to call
    from every AgentAPI that enables checking."""
    global _installed
    if _installed:
        return
    for module, name in _TARGETS:
        original = getattr(module, name, None)
        if original is None:
            continue
        _originals[(module, name)] = original
        setattr(module, name, _wrap(original, f"{module.__name__}.{name}"))
    _installed = True


def uninstall() -> None:
    """Restore the originals (used by tests)."""
    global _installed
    for (module, name), original in _originals.items():
        setattr(module, name, original)
    _originals.clear()
    _installed = False


def _wrap(original: Callable[..., Any], label: str) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        guard = _guard.get()
        if guard is not None and guard.depth == 0:
            guard.violation(label)
        return original(*args, **kwargs)
    return wrapper


@contextmanager
def guarding(mode: str, run_id: str):
    """Mark this task as durable handler code for the duration."""
    if mode == "off":
        yield None
        return
    install()
    guard = _Guard(mode, run_id)
    token = _guard.set(guard)
    try:
        yield guard
    finally:
        _guard.reset(token)


@contextmanager
def suppressed():
    """Temporarily allow nondeterminism — inside a @step (whose result is
    journaled) or inside ctx.now()/uuid()/random() (which journal their own
    values), nondeterminism is exactly what we want."""
    guard = _guard.get()
    if guard is None:
        yield
        return
    guard.depth += 1
    try:
        yield
    finally:
        guard.depth -= 1


def active_guard() -> Optional[_Guard]:
    return _guard.get()
