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
| Capabilities copy-pasted between services | **Skills**: instructions + ops + hooks in one mountable bundle, loadable from a `SKILL.md` directory |

## Surfaces

```
POST /{route}                  create a run (SSE stream; ?stream=false to block)
GET  /runs/{id}                status + usage
GET  /runs/{id}/events         resumable SSE (?from= cursor / Last-Event-ID)
POST /runs/{id}/signals/{s}    deliver a human-in-the-loop signal
POST /runs/{id}/cancel         cancel (interrupts in-flight awaits)
GET  /ops                      op catalog          GET /llm/tools   tool defs
POST /mcp                      MCP server          GET /skills      skills
GET  /pools                    admission stats     GET /healthz
```

## Status

Working core (durability tier 1, "resumable") with a full test suite:
run lifecycle, resume-by-cursor, detach/cancel policies, budgets, deadlines,
pause/signal, steps, all three op surfaces, hooks, skills, pools. Tier-2
durable journals (Postgres), prefix-cache-aware routing, and replay/eval CLI
are designed in DESIGN.md but not built yet.

```bash
pip install -e ".[dev]" && pytest
```
