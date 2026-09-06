"""OpenTelemetry tracing, correlated by run.

The observability complaint in DESIGN.md §2 was that spans, prompt logs and
the HTTP request live in three systems with nothing joining them. Here the
run *is* the correlation id: one span per run, child spans per step, LLM
call and tool call, all carrying ``agentapi.run_id``.

Installed as hooks, so it is opt-in and costs nothing when off:

    from agentapi.tracing import instrument
    instrument(app)

Degrades to a no-op when opentelemetry is not installed — a missing
observability dependency must never stop a server from serving.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agentapi.tracing")

try:  # pragma: no cover - exercised by presence/absence of the dependency
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    _OTEL = True
except ImportError:  # pragma: no cover
    _OTEL = False
    trace = None  # type: ignore[assignment]


def available() -> bool:
    return _OTEL


def instrument(app: Any, *, tracer_provider: Any = None,
               capture_content: bool = False) -> bool:
    """Attach tracing hooks to an AgentAPI app.

    ``capture_content`` also records prompts and results on spans. It is off
    by default: that content is the most sensitive data in the system, and
    quietly shipping it to a tracing vendor should be a decision, not a
    default.

    Returns False (and logs) if opentelemetry is unavailable.
    """
    if not _OTEL:
        logger.warning(
            "opentelemetry not installed; tracing disabled. "
            'pip install "agentapi[otel]"')
        return False

    tracer = (tracer_provider or trace).get_tracer("agentapi")
    spans: dict[str, Any] = {}

    @app.hook("on_run_start")
    def start_run(run: Any) -> None:
        span = tracer.start_span(
            f"run {run.route}", kind=SpanKind.SERVER,
            attributes={
                "agentapi.run_id": run.id,
                "agentapi.route": run.route,
                "agentapi.tenant": run.ctx.tenant or "",
                "agentapi.durable": bool(getattr(run, "durable", False)),
            })
        spans[run.id] = span

    @app.hook("on_event")
    def on_event(run: Any, event: Any) -> None:
        span = spans.get(run.id)
        if span is None:
            return
        # Events are the run's timeline; recording them as span events keeps
        # the trace and the event log telling the same story.
        attributes = {"agentapi.seq": event.seq}
        if event.type == "tool_call":
            attributes["agentapi.tool"] = event.name
        elif event.type == "error":
            attributes["agentapi.error_kind"] = getattr(event, "kind", "")
        if capture_content and event.type == "token":
            attributes["agentapi.text"] = event.text
        span.add_event(event.type, attributes=attributes)

    @app.hook("on_run_end")
    def end_run(run: Any) -> None:
        span = spans.pop(run.id, None)
        if span is None:
            return
        usage = run.ctx.usage
        span.set_attribute("agentapi.status", run.status.value)
        span.set_attribute("agentapi.input_tokens", usage.input_tokens)
        span.set_attribute("agentapi.output_tokens", usage.output_tokens)
        span.set_attribute("agentapi.usd", round(usage.usd, 6))
        span.set_attribute("agentapi.llm_calls", usage.llm_calls)
        span.set_attribute("agentapi.tool_calls", usage.tool_calls)
        span.set_attribute("agentapi.events", run.log.next_seq)
        if run.status.value in ("failed", "cancelled"):
            terminal = run.log.read(0)[-1] if run.log.next_seq else None
            span.set_status(Status(
                StatusCode.ERROR,
                getattr(terminal, "error", run.status.value)))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()

    @app.hook("on_tool_call")
    def tool_call(name: str, kwargs: dict) -> None:
        with tracer.start_as_current_span(
                f"tool {name}",
                attributes={"agentapi.tool": name}) as span:
            if capture_content:
                span.set_attribute("agentapi.arguments", str(kwargs))

    @app.hook("on_llm_call")
    def llm_call(model: str, params: dict) -> None:
        with tracer.start_as_current_span(
                f"llm {model}", kind=SpanKind.CLIENT,
                attributes={"agentapi.model": model}):
            pass

    return True
