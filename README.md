# agentapi

**A server runtime where the Run — not the Request — is the unit of work.**

FastAPI is superb at requests: short, stateless, owned by a TCP connection,
cheap to retry. LLM and agent workloads are none of those things. A run is
long, stateful, multi-step, expensive to retry, must survive the connection
that started it, and is metered in dollars rather than requests. `agentapi`
keeps FastAPI-shaped ergonomics but makes the run first-class; HTTP, SSE and
MCP are just ways to *attach* to one.

See [DESIGN.md](DESIGN.md) for the full rationale.

```python
from agentapi import AgentAPI, AnthropicLLM, Done, Token, ctx

app = AgentAPI(llm=AnthropicLLM())

@app.run("/chat", on_disconnect="detach", deadline="120s", budget_usd=0.50)
async def chat(prompt: str):
    async for tok in ctx.llm.stream(model="claude-opus-5",
                                    messages=[{"role": "user", "content": prompt}]):
        yield Token(text=tok)
    yield Done()
```

```bash
uvicorn examples.research_agent:app --port 8000
curl -N localhost:8000/research -X POST \
     -H 'content-type: application/json' -d '{"topic": "event logs"}'
```

## What you get

| Problem with request-shaped servers | What agentapi does |
|---|---|
| Client disconnect kills the generation you already paid for | `on_disconnect="detach" \| "cancel" \| "drain"` — a per-route policy, not a framework decree |
| Streams can't resume; no late joiners; one viewer per response | Every run writes an append-only **event log**; any number of clients attach by cursor: `GET /runs/{id}/events?from=42` (or `Last-Event-ID`) |
| No budget/cost model | `budget_usd=` / `budget_tokens=` per route, nested `ctx.budget(...)` scopes; `BudgetExceeded` raises **at the call site**, mid-stream, not on the invoice |
| No deadline propagation | `deadline="120s"` shrinks monotonically through every LLM call, tool, step and nested scope |
| Backends melt under uncontrolled concurrency | `app.pool(...)` admission control with **deadline-aware shedding**: reject with `429 + Retry-After` *now* instead of timing out later |
| Human-in-the-loop is impossible mid-request | `await ctx.pause("approval", schema=Approval, timeout="24h")` → resumed by `POST /runs/{id}/signals/approval` |
| Retried steps re-execute side effects | `@step(retries=3, timeout="30s")` — journaled per run, memoized on re-entry |
| The same function is hand-declared 3× (HTTP, LLM tool, MCP) | `@app.op` — one signature+docstring → HTTP route, Anthropic/OpenAI tool defs (`GET /llm/tools`), and an MCP server (`POST /mcp`) |
| Expensive double-fired requests | `Idempotency-Key` honoured on run creation |
| Observability bolted on per project | Lifecycle **hooks** (`on_run_start/on_event/on_run_end/on_tool_call/on_llm_call/...`); hook failures never kill a run |
| Half-streamed JSON can't be validated | `ctx.llm.stream_as(Invoice, ...)` yields a `PartialModel` per delta — fields fill in as tokens arrive — with a bounded repair loop if it never validates |
| `x-tenant-id` trusted as sent | Pluggable `@app.authenticator` → `Principal`; the principal's tenant is authoritative, every run records its owner, and run endpoints enforce it (404, not 403, so existence can't be probed) |
| Conversations bounce between backends, losing the KV cache | `app.sessions([...])` — sticky session affinity plus prefix-cache-aware placement, with a measurable hit rate on `GET /sessions` |
| The journal stores prompts and secrets verbatim, forever | `redactor=Redactor()` masks secrets, PII and named fields **on the way into** the journal — the live stream is untouched, and redacted journals still recover |
| One caller can exhaust the process | `rate_limit=RateLimit(per_minute=60)` — token bucket keyed by tenant (or principal), overridable per route, 429 with a computed `Retry-After` |
| A run can only be tailed on the worker that owns it | `fanout="redis://..."` mirrors events to Redis Streams, so **any** worker serves the same resumable stream for **any** run |
| Traces, prompt logs and the request live in three systems | `instrument(app)` — one OTEL span per run, children per step/tool/LLM call, all keyed by `agentapi.run_id`; prompts stay out unless you opt in |
| Capabilities copy-pasted between services | **Skills**: instructions + ops + hooks in one mountable bundle, loadable from a `SKILL.md` directory; skill instructions become the agent loop's system prompt |
| Every project rewrites the model↔tools loop | `await app.agent(model=..., messages=...)` — Anthropic-format tool loop over your ops; tool calls/results land in the event log, budgets and deadlines apply per turn |
| Process crash loses hours of agent work | `AgentAPI(durable="runs.db")` + `durability="durable"` routes: SQLite journal of events, steps and signals; `app.recover()` replays unfinished runs — completed steps don't re-execute, past signals re-deliver, history isn't duplicated |

## Agent harness

A sandboxed coding agent, wired to everything above:

```python
harness = attach_harness(
    app, "/agent",
    workspace="./work",
    policy=Policy(default="ask").allow("read_file", "grep", "glob"),
    skills_dir="./skills",
)
```

That gives a durable run route with a real model-tools loop over a
**sandboxed toolset** (`bash`, `read_file`, `write_file`, `edit_file`,
`grep`, `glob`, `list_dir`), each registered as an op — so every tool is
also an HTTP endpoint and an MCP tool.

**Sandbox.** Every path resolves (symlinks included) inside the workspace
or is refused; commands run under CPU/memory/process/file-size rlimits with
a wall-clock timeout that kills the process *group*; the environment is
scrubbed, so credentials in the server's environment are invisible to
model-authored commands; output is capped so a runaway command cannot flood
the context window. This narrows blast radius for a cooperative agent — for
genuinely hostile code, still run the server in a container.

**Policy with approval that survives a crash.** Rules are `allow` / `deny` /
`ask`, matched on tool name and argument patterns, deny-first so a broad
`allow("*")` cannot outrank a specific `deny`. An `ask` escalates through
`ctx.pause()` — which means on a durable run **the process may exit while
an approval is pending** and resume when the answer arrives. Approval
prompts dying because a worker restarted is the usual reason teams abandon
human-in-the-loop; here that failure mode does not exist.

**Progressive disclosure for skills.** `Skill.discover(dir)` loads a tree of
`SKILL.md` directories (the same convention Claude Code uses). The system
prompt carries only names and descriptions; the agent calls `load_skill`
for a body when a task needs it, so a dozen skills don't crowd out the
conversation. Bundled files are readable via `read_skill_resource`, jailed
to the skill's own directory.

See [`examples/agent.py`](examples/agent.py) for a complete one.

## Authentication

Without an authenticator the app is open — the same as FastAPI with no
dependencies. Register one and tenancy is enforced everywhere:

```python
@app.authenticator
async def authenticate(request):
    record = await lookup(request.headers.get("authorization"))
    return None if record is None else Principal(
        id=record.user, tenant=record.org)
```

The principal's tenant **overrides** any client-sent header, runs record
their owner at creation, idempotency keys are scoped per tenant, and another
tenant asking about your run gets a 404 rather than a 403. Pass
`require_auth=True` to turn "no authenticator registered" into a startup
error instead of a silent hole.

## Performance vs FastAPI

Measured, not asserted (`python benchmarks/bench.py`, full numbers in
[BENCHMARKS.md](BENCHMARKS.md)):

| Scenario | FastAPI | agentapi |
|---|---|---|
| JSON request/response | 396 req/s | 397 req/s — **tie** |
| SSE streaming (40 events) | 237 req/s | 196 req/s — **FastAPI wins by ~17%** |
| Client hangs up mid-stream | work is gone | run completes, 41 events replayable |

The streaming gap is real: appending to a log, sequencing and serialising
each event costs ~0.3 ms per event. That is the price of resumability, and
it is 0.1%–0.01% of a real LLM turn. Routes that need none of it can opt out
with `durability="ephemeral"`.

## Surfaces

```
POST /{route}                  create a run (SSE stream; ?stream=false to block)
GET  /runs/{id}                status + usage
GET  /runs/{id}/events         resumable SSE (?from= cursor / Last-Event-ID)
WS   /ws/runs/{id}             bidirectional: events out, signals/cancel in
POST /runs/{id}/signals/{s}    deliver a human-in-the-loop signal
POST /runs/{id}/cancel         cancel (interrupts in-flight awaits)
POST /v1/chat/completions      OpenAI-compatible (app.openai_compat("/chat"))
GET  /ops                      op catalog          GET /llm/tools   tool defs
POST /mcp                      MCP server          GET /skills      skills
GET  /pools                    admission stats     GET /sessions    routing
GET  /healthz
```

Already have a FastAPI service? Adopt per route instead of rewriting:

```python
app.mount("/legacy", existing_fastapi_app)
```

## Replay and eval

The journal records every event, step result and LLM exchange, so a
production run is already a reproducible test case:

```bash
agentapi runs   --app myapp:app          # list recorded runs
agentapi show   run_abc --app myapp:app  # full event history
agentapi replay run_abc --app myapp:app  # re-run offline against recorded
                                         # LLM responses; reports divergence
agentapi eval   cases.json --app myapp:app
```

`replay` never touches the provider — it feeds journaled responses back in,
so regression-testing a prompt change against real traffic costs nothing.

## Durability tiers

| `durability=` | Survives | Backing |
|---|---|---|
| `ephemeral` | nothing (classic request) | — |
| `resumable` (default) | client disconnects, reattach, replay | in-memory event log |
| `durable` | **process crashes**, deploys, long HITL pauses | SQLite journal (`AgentAPI(durable="runs.db")`) or Postgres (`AgentAPI(durable="postgresql://...")`) |

On Postgres several workers share one journal, so any worker can resume a
run whose original process died. `claim_runs` makes that safe with an
atomic `UPDATE ... RETURNING` behind a lease: two workers recovering at the
same instant cannot both resume a run and double its side effects.

Recovery is replay-based: `app.recover()` re-executes unfinished durable
runs from the top — `@step` results return from the journal instead of
re-running side effects, past signals re-deliver the same payloads, and
re-emitted events are deduplicated against persisted history.

### Determinism is checked, not just documented

Replay only works if a handler re-run takes the same path. A stray
`time.time()` would silently corrupt the journal, so the runtime catches it
two ways:

- **Proactively** — `time`/`random`/`uuid` calls made inside a durable
  handler *outside a step* raise `NondeterminismError` at the offending
  line, naming the fix. The wrappers are inert everywhere else: they act
  only on tasks running durable handler code, so other threads, tasks and
  libraries are untouched.
- **Reactively** — during recovery, an emitted event matching nothing in
  the remaining history proves divergence, whatever the cause (`datetime.now()`,
  dict ordering, an unjournaled read). The run fails loudly instead of
  writing a corrupt journal.

`ctx.now()`, `ctx.uuid()` and `ctx.random()` are **journaled**: a recovered
run sees the same clock, ids and dice as the execution it resumes. Anything
else nondeterministic belongs inside a `@step`, whose result is journaled.

Set the policy with `AgentAPI(determinism="raise" | "warn" | "off")`
(default `"raise"`; applies to durable runs only).

## Status

Working core with a 110-test suite: run lifecycle, resume-by-cursor,
detach/cancel/drain policies, budgets, deadlines, pause/signal, steps, all
three op surfaces, the agent loop, hooks, skills, fair-queueing pools, crash
recovery (incl. crash-mid-stream with no duplicated events), determinism
checking, WebSocket and OpenAI-compatible transports, partial validation,
the replay CLI, authentication and tenant isolation, a Postgres journal
(exercised against a real server, including multi-worker claim), session
affinity with prefix-cache routing, OpenTelemetry tracing, journal
redaction, per-principal rate limiting, cross-worker event fanout, and the
sandboxed agent harness (containment, tool policy, approvals, skills).
CI runs the suite on Python 3.11-3.13 against real Postgres and Redis
services, plus ruff.

**`AnthropicLLM` has never been run against the live API.** Every test uses
`MockLLM`, so the real provider path — SSE parsing, error handling, retries
— is unverified. That is the largest remaining unknown in the project.

Also not built: multi-region/replicated journals, a UI for browsing runs,
and streaming tool-call deltas (tool calls are dispatched only once the
model's turn completes).

```bash
pip install -e ".[dev]" && pytest
```
