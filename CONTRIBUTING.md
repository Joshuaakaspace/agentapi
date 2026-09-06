# Contributing to agentapi

Thanks for looking. This project is young and moving fast, which means your
contribution has an outsized effect on where it ends up.

## The fastest ways to help

- **Run it against a real model and report what breaks.** `AnthropicLLM` has
  been exercised less than any other component. A bug report from a live run
  is worth more than a feature right now.
- **Pick a `good first issue`.** They are scoped to be finishable in an
  evening and each links to the file you'll be touching.
- **Write a skill.** Drop a directory with a `SKILL.md` into
  `examples/skills/` — see `examples/skills/code-review` for the shape.
- **Add an MCP server integration** and document it.

## Setting up

```bash
git clone https://github.com/Joshuaakaspace/agentapi
cd agentapi
pip install -e ".[dev]"
pytest -q                     # ~10s; Postgres/Redis tests skip if absent
ruff check agentapi tests benchmarks
```

To run the durability tests for real, point them at services:

```bash
AGENTAPI_TEST_PG=postgresql://postgres@localhost:5432/postgres \
AGENTAPI_TEST_REDIS=redis://localhost:6379/0 pytest -q
```

## What a good PR looks like

- **One concern per PR.** A feature and an unrelated refactor are two PRs.
- **A test that fails without your change.** Interactive behaviour
  (mid-stream disconnects, WebSockets, pauses) needs the live-server fixture,
  not `ASGITransport` — see `tests/test_agentapi.py::live_app` and the note
  on why.
- **A commit message that explains *why*.** The codebase's commit history is
  written as a design log; keep it that way.
- **Honesty about limits.** If your change has a caveat, put it in the
  docstring and, if it's operational, in `PRODUCTION.md`.

## Design principles (so reviews are predictable)

1. **The run is the unit of work.** Anything that assumes a request owns
   the work will be pushed back on.
2. **Never trust a tool more than a local one** — MCP, skills and built-ins
   all pass through the same policy gate and event log.
3. **Nothing blocks the event loop.** Synchronous I/O goes through a thread.
   There's a regression test that will catch you if you forget.
4. **Fail loudly over silently.** A run that errors with a clear message
   beats one that completes with corrupt state.
5. **Measure before optimising.** Benchmarks live in `benchmarks/`.

## Reporting a security issue

Please don't open a public issue. See [SECURITY.md](SECURITY.md).

## Licence

By contributing you agree your work is released under the MIT licence in
[LICENSE](LICENSE).
