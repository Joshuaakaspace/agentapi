"""Metrics and structured logging.

Two things an operator needs on day one and cannot add later from outside:
numbers about what the server is doing, and logs that can be tied back to a
specific run.

``/metrics`` speaks Prometheus text format with no client library — the
dependency is not worth it for a dozen series, and a hand-rolled exposition
keeps the metric names stable and readable.

Logging is JSON with the run id attached, so a run's log lines, its trace
(``agentapi.tracing``) and its event log all key on the same identifier.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from typing import Any

from .determinism import real_monotonic, real_time


class Metrics:
    """Counters, gauges and histograms for the Prometheus endpoint."""

    def __init__(self) -> None:
        self.started_at = real_time()
        self.counters: Counter[tuple[str, tuple]] = Counter()
        self.histograms: dict[tuple[str, tuple], list[float]] = defaultdict(list)
        self._max_samples = 2048

    def incr(self, name: str, value: float = 1.0, **labels: Any) -> None:
        self.counters[(name, _labels(labels))] += value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        samples = self.histograms[(name, _labels(labels))]
        samples.append(value)
        if len(samples) > self._max_samples:
            # Reservoir-free downsample: keep the recent half. Quantiles stay
            # representative of current behaviour, which is what alerts on.
            del samples[:len(samples) // 2]

    # -- run lifecycle instrumentation --------------------------------------
    def install(self, app: Any) -> None:
        """Attach as hooks so metrics need no changes to handler code."""
        starts: dict[str, float] = {}

        @app.hook("on_run_start")
        def on_start(run: Any) -> None:
            starts[run.id] = real_monotonic()
            self.incr("agentapi_runs_started_total", route=run.route)

        @app.hook("on_run_end")
        def on_end(run: Any) -> None:
            elapsed = real_monotonic() - starts.pop(run.id, real_monotonic())
            usage = run.ctx.usage
            self.incr("agentapi_runs_finished_total",
                      route=run.route, status=run.status.value)
            self.observe("agentapi_run_duration_seconds", elapsed,
                         route=run.route)
            self.observe("agentapi_run_cost_usd", usage.usd, route=run.route)
            self.incr("agentapi_tokens_total", usage.input_tokens,
                      route=run.route, direction="input")
            self.incr("agentapi_tokens_total", usage.output_tokens,
                      route=run.route, direction="output")
            self.incr("agentapi_cost_usd_total", usage.usd, route=run.route)

        @app.hook("on_tool_call")
        def on_tool(name: str, _kwargs: dict) -> None:
            self.incr("agentapi_tool_calls_total", tool=name)

        @app.hook("on_llm_call")
        def on_llm(model: str, _params: dict) -> None:
            self.incr("agentapi_llm_calls_total", model=model)

    # -- exposition ---------------------------------------------------------
    def render(self, app: Any) -> str:
        lines: list[str] = []

        def emit(name: str, kind: str, help_text: str, series: list[str]) -> None:
            if not series:
                return
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            lines.extend(series)

        grouped: dict[str, list[str]] = defaultdict(list)
        for (name, labels), value in sorted(self.counters.items()):
            grouped[name].append(f"{name}{_render_labels(labels)} {value:g}")
        for name, series in grouped.items():
            emit(name, "counter", name.replace("_", " "), series)

        for (name, labels), samples in sorted(self.histograms.items()):
            if not samples:
                continue
            ordered = sorted(samples)
            rendered = _render_labels(labels)
            quantiles = [
                f'{name}{_with_quantile(rendered, "0.5")} '
                f'{ordered[len(ordered) // 2]:g}',
                f'{name}{_with_quantile(rendered, "0.95")} '
                f'{ordered[int(len(ordered) * 0.95)]:g}',
                f'{name}{_with_quantile(rendered, "0.99")} '
                f'{ordered[int(len(ordered) * 0.99)]:g}',
                f"{name}_sum{rendered} {sum(ordered):g}",
                f"{name}_count{rendered} {len(ordered)}",
            ]
            emit(name, "summary", name.replace("_", " "), quantiles)

        # Live gauges, read at scrape time rather than tracked incrementally.
        active = sum(1 for r in app.runs.runs.values()
                     if r.status.value in ("running", "paused", "queued"))
        attached = sum(r.attached for r in app.runs.runs.values())
        emit("agentapi_runs_active", "gauge", "runs currently in flight",
             [f"agentapi_runs_active {active}"])
        emit("agentapi_runs_tracked", "gauge", "runs held in memory",
             [f"agentapi_runs_tracked {len(app.runs.runs)}"])
        emit("agentapi_subscribers", "gauge", "attached event subscribers",
             [f"agentapi_subscribers {attached}"])
        emit("agentapi_uptime_seconds", "gauge", "process uptime",
             [f"agentapi_uptime_seconds {real_time() - self.started_at:g}"])

        pool_series, queue_series = [], []
        for pool in app.pools.values():
            stats = pool.stats()
            label = _render_labels((("pool", str(stats["name"])),))
            pool_series.append(f"agentapi_pool_in_flight{label} "
                               f"{stats['in_flight']}")
            queue_series.append(f"agentapi_pool_queued{label} "
                                f"{stats['waiting']}")
        emit("agentapi_pool_in_flight", "gauge", "slots in use", pool_series)
        emit("agentapi_pool_queued", "gauge", "waiters queued", queue_series)

        return "\n".join(lines) + "\n"


def _labels(labels: dict[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _render_labels(labels: tuple) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
    return "{" + inner + "}"


def _with_quantile(rendered: str, quantile: str) -> str:
    if rendered:
        return rendered[:-1] + f',quantile="{quantile}"' + "}"
    return '{quantile="' + quantile + '"}'


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the current run id when there is one."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        try:
            from .context import _current
            context = _current.get()
            if context is not None:
                payload["run_id"] = context.run_id
                if context.tenant:
                    payload["tenant"] = context.tenant
        except Exception:  # noqa: BLE001 - logging must never raise
            pass
        for key, value in getattr(record, "extra", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure root logging. Call once at process start."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output
                         else logging.Formatter(
                             "%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
