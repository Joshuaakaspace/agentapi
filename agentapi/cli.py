"""``agentapi`` command line: inspect, replay and evaluate recorded runs.

Because the journal records every event, step result and LLM exchange, a
production run is already a reproducible test case:

    agentapi runs        --app examples.research_agent:app
    agentapi show  <id>  --app examples.research_agent:app
    agentapi replay <id> --app examples.research_agent:app
    agentapi eval  cases.json --app examples.research_agent:app

``replay`` re-executes the handler against the *recorded* LLM responses —
no network, no spend — and reports whether the run still reaches the same
events. A diff means your prompt or code change altered behaviour: exactly
the regression signal that is otherwise expensive to get.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from typing import Any

from .context import RunContext
from .events import parse_event
from .run import RunStatus


def _load_app(spec: str) -> Any:
    """Load ``module:attribute`` (the uvicorn convention)."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--app must look like 'module:attribute'")
    sys.path.insert(0, "")
    module = importlib.import_module(module_name)
    app = getattr(module, attr, None)
    if app is None:
        raise SystemExit(f"{spec}: no attribute {attr!r}")
    if getattr(app, "backend", None) is None:
        raise SystemExit(
            f"{spec} has no journal; construct it with "
            f'AgentAPI(durable="runs.db") to record runs')
    return app


def _fmt_event(event: dict[str, Any]) -> str:
    kind = event.get("type", "?")
    detail = {k: v for k, v in event.items()
              if k not in ("type", "seq", "ts")}
    text = json.dumps(detail, default=str)
    if len(text) > 120:
        text = text[:117] + "..."
    return f"  {event.get('seq'):>4}  {kind:<12} {text}"


def cmd_runs(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    rows = app.backend.all_runs(limit=args.limit)
    if not rows:
        print("no recorded runs")
        return 0
    print(f"{'RUN':<26} {'ROUTE':<16} {'STATUS':<12} EVENTS")
    for row in rows:
        count = len(app.backend.events(row["id"]))
        print(f"{row['id']:<26} {row['route']:<16} {row['status']:<12} {count}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    row = app.backend.get_run(args.run_id)
    if row is None:
        print(f"run {args.run_id} not found", file=sys.stderr)
        return 1
    print(f"run     {row['id']}")
    print(f"route   {row['route']}")
    print(f"status  {row['status']}")
    print(f"input   {json.dumps(row['kwargs'], default=str)}")
    calls = app.backend.llm_calls(args.run_id)
    print(f"llm     {len(calls)} recorded call(s)")
    print("events:")
    for event in app.backend.events(args.run_id):
        print(_fmt_event(event))
    return 0


async def _replay(app: Any, run_id: str) -> tuple[bool, list[str]]:
    """Re-run a recorded run against its recorded LLM responses.

    Returns (matched, diff_lines). Nothing is written back to the journal —
    replay is a read-only experiment on a scratch context.
    """
    row = app.backend.get_run(run_id)
    if row is None:
        raise SystemExit(f"run {run_id} not found")
    route = app._run_routes.get(row["route"])
    if route is None:
        raise SystemExit(f"route {row['route']} is not registered on this app")

    recorded = [parse_event(e) for e in app.backend.events(run_id)]
    context = RunContext(f"replay_{run_id}", tenant=row["tenant"],
                         metadata=row["metadata"])
    context._llm = app.llm
    context._llm_replay = app.backend.llm_calls(run_id)
    # Journaled clock/ids/dice and step results come from the original run,
    # so the replay sees exactly the world the first execution saw.
    context._step_journal = dict(app.backend.steps(run_id))
    for name, payload in app.backend.signals(run_id):
        context.deliver_signal(name, payload)

    # Run on a throwaway manager so nothing touches the real journal.
    from .run import RunManager
    manager = RunManager()
    run = manager.start(route.path, route.handler, row["kwargs"], ctx=context)
    try:
        await asyncio.wait_for(run.task, timeout=60)
    except TimeoutError:
        return False, ["replay timed out after 60s"]

    produced = run.log.read(0)
    diff: list[str] = []
    for index in range(max(len(recorded), len(produced))):
        before = recorded[index].type if index < len(recorded) else None
        after = produced[index].type if index < len(produced) else None
        if before != after:
            diff.append(f"  seq {index}: recorded {before!r} -> replay {after!r}")
    return not diff, diff


def cmd_replay(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    matched, diff = asyncio.run(_replay(app, args.run_id))
    if matched:
        print(f"replay of {args.run_id}: identical event sequence ✓")
        return 0
    print(f"replay of {args.run_id}: DIVERGED", file=sys.stderr)
    for line in diff:
        print(line, file=sys.stderr)
    return 1


def cmd_eval(args: argparse.Namespace) -> int:
    """Run a dataset of cases through a route and report pass/fail.

    cases.json: [{"route": "/research", "input": {...},
                  "expect_events": ["state_delta", "done"],
                  "expect_result": {...}}]
    """
    app = _load_app(args.app)
    with open(args.dataset) as handle:
        cases = json.load(handle)
    passed = failed = 0
    for index, case in enumerate(cases):
        route = app._run_routes.get(case["route"])
        if route is None:
            print(f"case {index}: unknown route {case['route']}")
            failed += 1
            continue
        outcome = asyncio.run(_run_case(app, route, case))
        if outcome is None:
            print(f"case {index}: PASS")
            passed += 1
        else:
            print(f"case {index}: FAIL — {outcome}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


async def _run_case(app: Any, route: Any, case: dict[str, Any]) -> str | None:
    from .run import RunManager
    context = RunContext("eval")
    context._llm = app.llm
    manager = RunManager()
    run = manager.start(route.path, route.handler, case.get("input", {}),
                        ctx=context)
    try:
        await asyncio.wait_for(run.task, timeout=case.get("timeout", 60))
    except TimeoutError:
        return "timed out"
    events = run.log.read(0)
    types = [e.type for e in events]
    expected = case.get("expect_events")
    if expected is not None and types != expected:
        return f"events {types} != expected {expected}"
    if run.status is not RunStatus.COMPLETED and not case.get("expect_failure"):
        return f"status {run.status.value}: {getattr(events[-1], 'error', '')}"
    if "expect_result" in case:
        actual = getattr(events[-1], "result", None)
        if actual != case["expect_result"]:
            return f"result {actual!r} != expected {case['expect_result']!r}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentapi", description=__doc__)
    parser.add_argument("--app", default="app:app",
                        help="module:attribute of the AgentAPI instance")
    sub = parser.add_subparsers(dest="command", required=True)

    p_runs = sub.add_parser("runs", help="list recorded runs")
    p_runs.add_argument("--limit", type=int, default=50)
    p_runs.set_defaults(func=cmd_runs)

    p_show = sub.add_parser("show", help="print a run's full event history")
    p_show.add_argument("run_id")
    p_show.set_defaults(func=cmd_show)

    p_replay = sub.add_parser(
        "replay", help="re-execute a run against its recorded LLM responses")
    p_replay.add_argument("run_id")
    p_replay.set_defaults(func=cmd_replay)

    p_eval = sub.add_parser("eval", help="run a dataset of cases")
    p_eval.add_argument("dataset")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
