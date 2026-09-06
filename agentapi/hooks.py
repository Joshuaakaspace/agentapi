"""Lifecycle hooks.

Hooks observe and instrument the runtime — run start/end, every event,
every tool call. They are how tracing, prompt logging, metrics, and guard
rails plug in without wrapping handlers by hand.

    @app.hook("on_event")
    async def trace(run, event): ...

Hook failures are contained: an observability bug must not kill a $2 run.
Registered hooks may be sync or async.
"""
from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("agentapi.hooks")

HOOK_NAMES = (
    "on_run_start",    # (run)
    "on_event",        # (run, event)
    "on_run_end",      # (run)
    "on_tool_call",    # (name, kwargs)
    "on_llm_call",     # (model, params)
    "on_llm_result",   # (model, usage)
)


class Hooks:
    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def register(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        if name not in HOOK_NAMES:
            raise ValueError(f"unknown hook {name!r}; valid: {HOOK_NAMES}")
        self._hooks[name].append(fn)
        return fn

    async def fire(self, name: str, *args: Any) -> None:
        for fn in self._hooks.get(name, ()):
            try:
                result = fn(*args)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - hooks never kill runs
                logger.exception("hook %s (%s) raised", name,
                                 getattr(fn, "__name__", fn))

    def merge(self, other: Hooks) -> None:
        for name, fns in other._hooks.items():
            self._hooks[name].extend(fns)
