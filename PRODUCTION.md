# Running agentapi in production

An honest account of what is ready, what you must configure, and what is
still unverified. Read the last section before you put real traffic on it.

---

## What the runtime handles for you

| Concern | Status |
|---|---|
| Runs survive client disconnects | ✅ `on_disconnect="detach"` (default) |
| Runs survive process crashes | ✅ `durability="durable"` + a journal |
| Runs survive rolling deploys | ✅ graceful shutdown drains in flight; paused runs resume elsewhere |
| Streams resume after a reconnect | ✅ cursor / `Last-Event-ID` |
| Any worker can serve any run | ✅ Postgres journal + Redis fanout |
| Cost and deadline enforcement | ✅ per route and per nested scope |
| Backend overload | ✅ pools with fair queueing and deadline-aware shedding |
| One tenant starving others | ✅ deficit round robin |
| One caller exhausting the process | ✅ per-principal rate limiting |
| Tenant isolation | ✅ auth-derived, enforced on every run endpoint |
| Secrets in the journal | ✅ redaction on write |
| Untrusted model-authored commands | ✅ sandbox (see its limits below) |
| Observability | ✅ Prometheus `/metrics`, OTEL traces, JSON logs keyed by run id |
| Health checking | ✅ `/healthz` liveness, `/readyz` readiness |

## Minimum production configuration

```bash
AGENTAPI_DURABLE=postgresql://…      # SQLite is single-process only
AGENTAPI_FANOUT=redis://…            # any worker serves any run
AGENTAPI_REQUIRE_AUTH=true           # fail startup rather than run open
AGENTAPI_REDACT=true                 # keep secrets out of the journal
AGENTAPI_RATE_LIMIT_PER_MINUTE=120
AGENTAPI_SHUTDOWN_GRACE_S=30
AGENTAPI_LOG_JSON=true
```

Then register an authenticator — `require_auth=true` without one is a
startup error by design, not a silent open door:

```python
@app.authenticator
async def authenticate(request):
    record = await verify(request.headers.get("authorization"))
    return None if record is None else Principal(id=record.user,
                                                 tenant=record.org)
```

## Deployment notes that bite people

**Load balancer idle timeout must exceed your longest run.** An agent
streaming for 20 minutes dies at an ALB's default 60s idle timeout. Either
raise it, or rely on `detach` + reconnect (the runtime supports it; your LB
still has to allow the initial connection to live long enough to hand back
a run id).

**Shutdown ordering.** `--timeout-graceful-shutdown` must be *longer* than
`AGENTAPI_SHUTDOWN_GRACE_S`, or uvicorn kills the worker while the app is
still draining. The supplied Dockerfile uses 45s against a 30s grace.

**Kubernetes.** Point the readiness probe at `/readyz` and liveness at
`/healthz` — never the reverse. `/readyz` reports `draining` during
shutdown so the pod leaves the service before it stops accepting; `/healthz`
deliberately ignores dependencies so a database blip does not turn into a
crash loop. Set `terminationGracePeriodSeconds` above the shutdown grace.

**Journal growth.** Events, steps and LLM exchanges accumulate forever.
There is no retention job — add one, and decide how long you are willing to
hold prompts.

**Postgres connection pool.** `PostgresBackend` opens up to 8 connections
per worker. Multiply by worker count and compare against `max_connections`.

## Sandbox: what it does and does not promise

It reduces blast radius for a **cooperative** agent — one that may be
confused or misled, but is not actively attacking you:

- paths cannot escape the workspace (symlinks resolved first)
- rlimits on CPU, memory, processes and file size
- timeouts kill the process group
- the environment carries no credentials
- output is capped

It is **not** a boundary against hostile code sharing your kernel. There is
no seccomp filter and no user namespace beyond an optional `unshare -n`. If
you will run genuinely untrusted code, put the whole server in a container
or microVM and treat the sandbox as a second layer, not the only one.

## Still your job

- **A retention/deletion policy** for the journal.
- **Secret management** — `ANTHROPIC_API_KEY` from a secret store, not an
  env var baked into an image.
- **Backups** of the Postgres journal; it is the record of every run.
- **Alerting** on the metrics: `agentapi_runs_finished_total{status="failed"}`,
  `agentapi_pool_queued`, `agentapi_cost_usd_total`.
- **A cost ceiling you actually believe in.** `budget_usd` per route is
  enforced, but nothing stops a thousand runs each spending their maximum.
- **Load testing at your traffic shape.** The benchmark here is loopback
  and single-process.

## Not yet verified — read this before trusting it

**`AnthropicLLM` has never made a live API call.** Every one of the 126
tests runs against `MockLLM`. The provider path — SSE parsing, error
handling, retries, rate-limit responses — is written but unexercised. Run a
real request in staging before you trust it in production. This is the
largest known unknown in the project.

**No load testing at scale.** The benchmark is loopback, one process, no
network. Nothing here has been run under sustained concurrent production
traffic.

**No third-party security audit.** The auth and sandbox code is tested and
reviewed only by the people who wrote it.

**Journal write throughput is untested at volume.** Writes no longer block
the event loop, but sustained multi-worker write load against one Postgres
has not been measured.
