"""Steps: retryable, memoizable units of work inside a run.

Tier-1 durability: step results are memoized per run so re-entering a
handler (retry, resume) doesn't re-execute completed side effects. The
journal interface is what a Postgres backend implements for tier 2.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import hashlib
from typing import Any, Callable, Optional, TypeVar

from .context import get_ctx, parse_duration, _step_depth
from .determinism import suppressed

F = TypeVar("F", bound=Callable[..., Any])


def _step_key(name: str, args: tuple, kwargs: dict) -> str:
    try:
        payload = json.dumps([args, kwargs], sort_keys=True, default=str)
    except TypeError:
        payload = repr((args, kwargs))
    return f"{name}:{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def step(fn: Optional[F] = None, *, retries: int = 0,
         backoff: float = 0.5, timeout: Optional[str | float] = None,
         name: Optional[str] = None) -> Any:
    """Wrap a coroutine function as a journaled, retryable step.

        @step(retries=3, timeout="30s")
        async def fetch_docs(q: str) -> list[dict]: ...

    Within one run, a completed step called again with the same arguments
    returns its journaled result instead of re-executing.
    """
    def decorate(func: F) -> F:
        step_name = name or func.__name__
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"@step requires an async function: {step_name}")

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            context = get_ctx()
            journal = context._step_journal
            key = _step_key(step_name, args, kwargs)
            if key in journal:
                return journal[key]
            timeout_s = None if timeout is None else parse_duration(timeout)
            last_exc: Optional[BaseException] = None
            for attempt in range(retries + 1):
                context.check()
                try:
                    # A step's result is journaled, so nondeterminism
                    # inside it is fine — that is the entire point of steps.
                    # The depth token also tells a draining run not to stop
                    # mid-step: "drain" finishes the current step first.
                    depth = _step_depth.set(_step_depth.get() + 1)
                    try:
                        with suppressed():
                            coro = func(*args, **kwargs)
                            result = (await asyncio.wait_for(coro, timeout_s)
                                      if timeout_s else await coro)
                    finally:
                        _step_depth.reset(depth)
                    journal[key] = result
                    if context._step_commit is not None:
                        context._step_commit(key, result)
                    return result
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < retries:
                        await asyncio.sleep(backoff * (2 ** attempt))
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorate if fn is None else decorate(fn)

