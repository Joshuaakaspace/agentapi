# AgentAPI — a server runtime for LLM, agent and orchestration workloads

Status: design sketch / brainstorm. Nothing implemented yet.

---

## 1. The core thesis

> FastAPI's unit of work is a **Request**. An agent's unit of work is a **Run**.

Almost every practical complaint about FastAPI-for-agents follows from that one
mismatch. A request is short, stateless, owned by a TCP connection, and cheap to
retry. A run is long (seconds to hours), stateful, multi-step, expensive to
retry, must survive the connection that started it, and is metered in dollars
rather than in requests.

So the design goal is not "a faster FastAPI". It is: **make the Run the
first-class object, and make HTTP/SSE/WS/MCP merely ways to *attach* to a run.**

---

## 2. What actually hurts today

Ranked by how often it burns real production systems.

| # | Problem | Why FastAPI can't fix it |
|---|---------|--------------------------|
| 1 | **Client disconnect kills the generation.** You already paid for the tokens; the ASGI task is cancelled and the output is gone. Mobile clients, tab closes, LB blips. | Cancel-on-disconnect is ASGI lifecycle semantics. The work is owned by the connection. |
| 2 | **No resumable streams.** No event IDs, no `Last-Event-ID` replay, no late-joiners, no second viewer on the same run. | SSE is a raw `StreamingResponse`; there is no event log behind it. |
| 3 | **No durable execution.** Multi-step agents need checkpoints, per-step retries, crash recovery. `BackgroundTasks` is in-process and dies with the worker. | Out of scope by design; you bolt on Celery/Temporal and now run two programming models. |
| 4 | **No admission control.** LLM backends have hard concurrency/RPM/TPM limits. FastAPI accepts 10k concurrent requests and lets them all time out at once. | No queueing, priority, fairness, or load-shedding primitives. |
| 5 | **No cost/budget awareness.** The scarce resource is tokens and dollars, not requests. Per-tenant quota, budget ceilings, and cost accounting get rebuilt in every company. | Framework has no concept of a metered resource. |
| 6 | **No deadline propagation.** Client has 30s; the LLM call, the tool calls and the sub-agents don't know that. Nothing cancels the tree. | No context object beyond `Request`. |
| 7 | **No human-in-the-loop pause.** "Wait 24h for an approval" means the process must be able to die and come back. | Impossible without a journal. |
| 8 | **One function, three surfaces.** The same Python function is an HTTP endpoint, an LLM tool, and an MCP tool. You hand-write the schema 2–3 times. | Only generates OpenAPI. |
| 9 | **Output validation is the wrong direction.** Pydantic validates input beautifully. Agents need *output* validation with repair loops, and **partial** validation of half-streamed JSON. | Pydantic v2 has no first-class partial parsing. |
| 10 | **Sessions are stateless-by-decree.** No session affinity, no actor-per-conversation, no prefix-cache-aware routing (which is a large real cost lever). | Stateless is a core assumption. |
| 11 | **Observability doesn't line up.** OTEL spans, prompt logs and the HTTP request live in three systems with no shared correlation. | No run identity to hang them on. |
| 12 | **The sync/async footgun.** One `def` endpoint doing blocking work silently eats one of ~40 threadpool slots and tanks throughput under long LLM latency. | Inherent to the Starlette threadpool model. |

Honest caveat: items 1, 12 are really **ASGI/Starlette/uvicorn**, and some
timeout pain is **load balancer** config, not FastAPI. But that distinction
doesn't help the user — the framework is where the fix has to land.

---

## 3. Design

### 3.1 Runs, events, attachment

Every handler produces a **run**. A run has an identity, a journal, and an
append-only **event log**. Transports subscribe to the log by cursor.

```python
from agentapi import AgentAPI, ctx, step

app = AgentAPI()

@app.run("/chat", durability="durable", on_disconnect="detach")
async def chat(req: ChatRequest):
    history = await step.load(f"session:{req.session_id}")

    async for tok in ctx.llm.stream(model="claude-opus-5", messages=history + [req.message]):
        yield Token(tok)                      # -> event log -> every attached transport

    yield Done(usage=ctx.usage)
```

Consequences that fall out for free:

- `GET /runs/{id}/events?from=142` → resume exactly where the stream broke.
- Two clients can watch the same run. So can your dashboard.
- The event log **is** the trace. `GET /runs/{id}` returns the full replayable history.
- Webhooks, WebSocket and MCP are just other subscribers.

### 3.2 Durability tiers (opt-in, per route)

Durability is expensive; not every route needs it. Three tiers, incrementally adoptable:

| Tier | Name | Survives | Journal | Cost |
|------|------|----------|---------|------|
| 0 | `ephemeral` | nothing (FastAPI behaviour) | none | zero |
| 1 | `resumable` | client disconnect, reattach, replay | in-memory / Redis ring buffer | ~free |
| 2 | `durable` | process crash, deploy, 24h HITL pause | Postgres/SQLite step journal | one write per step |

Tier 1 alone — "your streams stop dying" — is already a product.

### 3.3 The step contract (how durability actually works)

Replay-based recovery, Temporal-style, but embedded and stream-native:

```python
@step(retries=3, backoff="exp", timeout="30s")
async def fetch_docs(q: str) -> list[Doc]: ...
```

- On recovery, the handler is re-run from the top; completed steps return
  journaled results instead of re-executing.
- **Determinism rule:** anything nondeterministic must go through `step()` or
  `ctx.*` (`ctx.now()`, `ctx.uuid()`, `ctx.random()`).
- Crucially: ship a **runtime determinism checker**, not just docs. Patch
  `time`/`random`/`uuid`/socket in handler scope and raise loudly outside steps.
  Silent journal corruption is the failure mode that kills adoption of this
  pattern, and it is preventable.
  *(Built — see `agentapi/determinism.py`. Two halves: proactive wrappers on
  `time`/`random`/`uuid` that fire only on tasks running durable handler code,
  and reactive replay-divergence detection that catches everything the
  wrappers cannot see. `ctx.now()/uuid()/random()` are journaled so a replay
  sees the same clock, ids and dice. Known blind spot: `datetime.datetime.now`
  cannot be wrapped — it is a C type — so it is caught reactively, not at the
  call site.)*
- Journal writes are batched/group-committed so a 20-step agent isn't 20 fsyncs.

### 3.4 `ctx` — deadline, budget, cancellation, tenancy

One context object, propagated implicitly through every LLM and tool call
(contextvars), including into sub-agents.

```python
async with ctx.budget(usd=0.50, tokens=200_000, deadline="120s"):
    result = await sub_agent(...)        # inherits remaining budget & deadline
```

- Exceeding the budget raises `BudgetExceeded` at the call site, not after the bill.
- Deadline shrinks monotonically down the tree; a call that cannot finish in the
  remaining time fails fast instead of starting.
- `ctx.usage` accumulates tokens/cost automatically → emitted in the terminal event and in metrics.

### 3.5 Admission control and scheduling

Model backends are the real bottleneck. Declare them as pools; handlers reserve from them.

```python
gpt = app.pool("openai:gpt-4o", concurrency=64, rpm=5_000, tpm=2_000_000)

@app.run("/chat", pools=[gpt], priority="interactive")
```

- **Adaptive token bucket:** meter *observed* usage back into the pool and shape
  it with upstream 429/`Retry-After` feedback rather than trusting static config.
- **Fair queueing per tenant** (deficit round robin) so one customer's batch job
  can't starve everyone's chat.
- **Deadline-aware admission:** if queue depth means the request's deadline is
  already unmeetable, reject *now* with `429 + Retry-After` instead of burning a
  slot to time out later. This is the single highest-leverage behaviour under load.
- Priority classes: `interactive` > `batch`, with preemption for batch.

### 3.6 Disconnect policy — a per-route decision, not a framework decree

```python
on_disconnect = "detach" | "cancel" | "drain"
```

`detach` (default for durable runs) keeps the run going and lets the client come
back to `/runs/{id}/events`. `cancel` reproduces FastAPI behaviour. `drain`
finishes the current step, then stops. This alone fixes complaint #1.

### 3.7 Idempotency

`Idempotency-Key` is first-class on run creation. Retrying a `$2` agent run
because a mobile client hiccuped is a real and expensive bug.

### 3.8 Human-in-the-loop

```python
approval = await ctx.pause("approval", schema=Approval, timeout="24h")
```

The run suspends, the journal persists, **the process may exit**. Resumed by
`POST /runs/{id}/signals/approval`. FastAPI structurally cannot do this.

### 3.9 One definition, many surfaces

```python
@app.op(http="POST /tools/search", llm_tool=True, mcp=True)
async def search(q: str, limit: int = 10) -> list[Doc]:
    """Search the corpus."""
```

Generates: an HTTP route + OpenAPI, a JSON-schema tool definition for the model,
and an MCP tool — from one signature and docstring. Schema drift between "the
tool the model sees" and "the function that runs" is a common and silent bug.

### 3.10 Typed streaming output with partial validation

```python
async for partial in ctx.llm.stream_as(Invoice, prompt=...):
    ...   # Partial[Invoice]; fields populate as tokens arrive, typed the whole way
```

Backed by a streaming JSON parser that tolerates truncation, plus an automatic
**repair loop** (re-prompt with the validation error) bounded by budget.

### 3.11 Sessions as addressable state

Sticky routing of a session to the node holding its state, and
**prefix-cache-aware placement** — routing a conversation back to the worker/
backend that already has its KV prefix cached is a straightforward large cost win
that no general web framework will ever do for you.

### 3.12 Replay, debugging and eval

Because the journal records every LLM call and result:

- `agentapi replay <run_id>` — re-execute against recorded responses. Time-travel debugging.
- Any production run becomes a regression test, for free.
- `agentapi eval <dataset>` — run a dataset through the real handler with the real journal.
- Prompt/model changes get a diff against recorded traffic before deploy.

---

## 4. What this is NOT

Scope discipline matters more than features here:

- Not an inference engine (vLLM/TGI do that).
- Not a vector DB, not a memory product.
- Not a prompt DSL or a chain/graph authoring framework — it's plain async Python.
- Not a general-purpose workflow engine competing with Temporal on non-agent workloads.
- Not a model gateway (it *uses* one; LiteLLM/Bedrock/etc. plug in).

---

## 5. Prior art and the gap

| Tool | Has | Missing for this use case |
|------|-----|---------------------------|
| FastAPI | ergonomics, ecosystem | everything in §2 |
| Temporal / Restate | durable execution | not stream-native, heavy op burden, no LLM resource/cost model |
| LangGraph | checkpointing, graph state | a library, not a server; no admission control, no transport story |
| Ray Serve | scaling, batching | no durability, no agent semantics |
| Celery / ARQ | background work | no streaming, no resume, no typed contract |
| litellm proxy | rate limits, cost tracking | proxy only — no application runtime |

**The gap:** nobody offers *FastAPI's ergonomics + Temporal's durability +
stream-native transport + LLM-aware admission control and cost* in one
programming model. That's the slot.

---

## 6. Build order (each stage independently useful)

1. **Run kernel + event log + resumable SSE + `detach` on disconnect.** ← ship this first; it's the whole value prop in miniature.
2. `ctx`: deadline, budget, cancellation propagation, usage accounting.
3. Admission control pools: fair queueing, deadline-aware shedding, adaptive rate limits.
4. Durable journal (SQLite → Postgres), step memoization, crash recovery, determinism checker.
5. Pause/signal (human-in-the-loop).
6. Multi-surface `@app.op` (HTTP + LLM tool + MCP).
7. Replay / eval CLI.
8. Session affinity + prefix-cache-aware routing.

---

## 7. Open decisions

1. **Python-only, or Rust core with a Python API?** Recommendation: Python-first
   (PyO3 later for the scheduler/event log/SSE fanout). LLM latency dominates,
   so raw hot-path throughput is not the bottleneck on day one. Starting in Rust
   trades the ecosystem — the one thing FastAPI's users will not give up.
2. **Build on ASGI, or own the transport?** Recommendation: keep ASGI as *a*
   transport (Starlette/uvicorn, so middleware and deploy targets keep working)
   but do **not** inherit its lifecycle for runs. Runs outlive connections.
3. **How deep does durability go on day one?** Recommendation: tiers 0 and 1
   first. Tier 2 is the deepest engineering and can land later without an API break.
4. **OpenAI-compatible `/v1/chat/completions` as a first-class surface?**
   Cheap to add, enormous adoption lever — every existing client works instantly.
5. **Migration story from FastAPI.** Mirror the decorator ergonomics exactly, and
   support `app.mount(fastapi_app)` so adoption is per-route, not a rewrite.

---

## 8. Biggest risks

- **Scope.** This is five hard distributed-systems problems at once. Mitigation:
  the build order above — every stage must stand alone.
- **Determinism constraints confuse users.** Mitigation: the runtime checker in §3.3.
- **Journal write amplification.** Mitigation: batched group commit, tiered durability.
- **"Two runtimes" complexity** (web server + scheduler). Mitigation: single
  process by default, scale out only when configured.
