"""Auth, Postgres durability, session routing and tracing."""
import asyncio
import json
import os

import httpx
import pytest
from pydantic import BaseModel

from agentapi import (AgentAPI, Done, MockLLM, Principal, StateDelta, Token,
                      bearer_tokens, ctx, step)

pytestmark = pytest.mark.asyncio

PG_DSN = os.environ.get(
    "AGENTAPI_TEST_PG",
    "postgresql://postgres@/postgres?host=/var/tmp&port=55432")


def client_for(app, **kwargs):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", **kwargs)


import contextlib
import socket

import uvicorn


@contextlib.asynccontextmanager
async def live_app(app):
    """Serve on a real socket: httpx's ASGITransport buffers the whole
    response, so any test that reads mid-stream needs a real server."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}",
                                     timeout=10) as client:
            yield client
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


# --- authentication and tenant isolation ----------------------------------

def auth_app():
    app = AgentAPI(llm=MockLLM())
    app.authenticator(bearer_tokens({
        "tok-alice": Principal(id="alice", tenant="acme"),
        "tok-bob": Principal(id="bob", tenant="globex"),
        "tok-alice2": Principal(id="alice2", tenant="acme"),   # same tenant
    }))

    @app.run("/chat")
    async def chat(prompt: str = "hi"):
        yield Token(text="secret-" + prompt)
        yield Done(result="secret-result")

    @app.op(http="POST /tools/ping")
    async def ping() -> str:
        """Ping."""
        return "pong"

    return app


async def test_unauthenticated_requests_are_rejected():
    app = auth_app()
    async with client_for(app) as client:
        for method, path in (("POST", "/chat"), ("POST", "/tools/ping"),
                             ("POST", "/mcp"), ("GET", "/runs/run_x")):
            response = await client.request(method, path, json={})
            assert response.status_code == 401, path
            if method == "POST" and path == "/chat":
                assert response.headers["www-authenticate"] == "Bearer"


async def test_bad_token_rejected_good_token_accepted():
    app = auth_app()
    async with client_for(app) as client:
        bad = await client.post("/tools/ping", json={},
                                headers={"authorization": "Bearer nope"})
        assert bad.status_code == 401
        good = await client.post("/tools/ping", json={},
                                 headers={"authorization": "Bearer tok-alice"})
        assert good.json()["result"] == "pong"


async def test_tenant_cannot_read_another_tenants_run():
    app = auth_app()
    alice = {"authorization": "Bearer tok-alice"}
    bob = {"authorization": "Bearer tok-bob"}

    async with client_for(app) as client:
        async with client.stream("POST", "/chat", json={"prompt": "x"},
                                 headers=alice) as response:
            run_id = response.headers["x-run-id"]
            await response.aread()

        # the owner sees it
        assert (await client.get(f"/runs/{run_id}", headers=alice)
                ).status_code == 200
        # a different tenant gets 404 — not 403, which would confirm it exists
        for path in (f"/runs/{run_id}", f"/runs/{run_id}/events"):
            assert (await client.get(path, headers=bob)).status_code == 404
        for path in (f"/runs/{run_id}/cancel",
                     f"/runs/{run_id}/signals/approval"):
            assert (await client.post(path, json={}, headers=bob)
                    ).status_code == 404
        # and cannot cancel it
        assert app.runs.get(run_id).status.value == "completed"


async def test_same_tenant_different_user_shares_access():
    """Ownership is by tenant: a colleague can attach to the same run."""
    app = auth_app()
    async with client_for(app) as client:
        async with client.stream("POST", "/chat", json={"prompt": "x"},
                                 headers={"authorization": "Bearer tok-alice"}
                                 ) as response:
            run_id = response.headers["x-run-id"]
            await response.aread()
        peer = await client.get(f"/runs/{run_id}",
                                headers={"authorization": "Bearer tok-alice2"})
    assert peer.status_code == 200


async def test_tenant_comes_from_auth_not_from_header():
    """The old hole: x-tenant-id was trusted as sent."""
    app = auth_app()
    async with client_for(app) as client:
        async with client.stream(
                "POST", "/chat", json={"prompt": "x"},
                headers={"authorization": "Bearer tok-alice",
                         "x-tenant-id": "globex"}) as response:
            run_id = response.headers["x-run-id"]
            await response.aread()
    assert app.runs.get(run_id).ctx.tenant == "acme"     # not "globex"
    assert app.runs.get(run_id).owner == ("alice", "acme")


async def test_idempotency_keys_are_scoped_per_tenant():
    app = auth_app()
    async with client_for(app) as client:
        first = await client.post(
            "/chat?stream=false", json={"prompt": "x"},
            headers={"authorization": "Bearer tok-alice",
                     "idempotency-key": "shared"})
        second = await client.post(
            "/chat?stream=false", json={"prompt": "x"},
            headers={"authorization": "Bearer tok-bob",
                     "idempotency-key": "shared"})
    assert first.json()["id"] != second.json()["id"]     # no cross-tenant reuse


async def test_require_auth_rejects_a_misconfigured_app():
    app = AgentAPI(llm=MockLLM(), require_auth=True)     # no authenticator

    @app.run("/chat")
    async def chat():
        yield Done()

    async with client_for(app) as client:
        response = await client.post("/chat", json={})
    assert response.status_code == 500
    assert "no authenticator" in response.json()["error"]


async def test_open_app_still_works_without_auth():
    """No authenticator registered = open, same as FastAPI with no deps."""
    app = AgentAPI(llm=MockLLM())

    @app.run("/open")
    async def open_route():
        yield Done(result="ok")

    async with client_for(app) as client:
        response = await client.post("/open?stream=false", json={})
    assert response.status_code == 200


# --- Postgres journal ------------------------------------------------------

def pg_available() -> bool:
    try:
        import psycopg
        with psycopg.connect(PG_DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pg_only = pytest.mark.skipif(not pg_available(),
                             reason="no postgres at AGENTAPI_TEST_PG")


@pg_only
async def test_postgres_backend_round_trips_everything():
    from agentapi.postgres import PostgresBackend
    backend = PostgresBackend(PG_DSN)
    run_id = f"run_pg_{os.urandom(6).hex()}"

    backend.create_run(run_id, "/x", {"topic": "kv"}, tenant="acme",
                       metadata={"principal": "alice"})
    backend.append_event(run_id, 0, {"type": "token", "text": "hi", "seq": 0})
    backend.save_step(run_id, "step:1", {"gathered": True})
    backend.append_signal(run_id, "approval", {"approved": True})
    backend.record_llm_call(run_id, "claude", {"messages": []}, {"usage": {}})

    row = backend.get_run(run_id)
    assert row["kwargs"] == {"topic": "kv"} and row["tenant"] == "acme"
    assert backend.events(run_id)[0]["text"] == "hi"
    assert backend.steps(run_id) == {"step:1": {"gathered": True}}
    assert backend.signals(run_id) == [("approval", {"approved": True})]
    assert backend.llm_calls(run_id)[0]["model"] == "claude"

    backend.update_status(run_id, "completed", finished=True)
    assert backend.get_run(run_id)["status"] == "completed"
    assert run_id not in [r["id"] for r in backend.unfinished_runs()]
    backend.close()


@pg_only
async def test_postgres_crash_recovery_end_to_end():
    """The full durability story, on Postgres instead of SQLite."""
    from agentapi.postgres import PostgresBackend

    side_effects = []
    suffix = os.urandom(6).hex()

    def build():
        app = AgentAPI(llm=MockLLM(script=["one two"]),
                       durable=PostgresBackend(PG_DSN, worker_id=f"w-{suffix}"))

        class Approval(BaseModel):
            approved: bool

        @step
        async def expensive(topic: str) -> str:
            side_effects.append(topic)
            return f"gathered:{topic}"

        @app.run(f"/pipeline-{suffix}", durability="durable")
        async def pipeline(topic: str):
            gathered = await expensive(topic)
            yield StateDelta(data={"gathered": gathered})
            approval = await ctx.pause("approval", schema=Approval,
                                       timeout="60s")
            yield Done(result={"approved": approval.approved})
        return app

    app1 = build()
    async with live_app(app1) as client:
        async with client.stream("POST", f"/pipeline-{suffix}",
                                 json={"topic": "kv"}) as response:
            run_id = response.headers["x-run-id"]
            async for line in response.aiter_lines():
                if line.startswith("event: paused"):
                    break
    assert side_effects == ["kv"]
    run1 = app1.runs.get(run_id)
    run1.durable = False               # abrupt death: nothing written on the way down
    run1.task.cancel()
    try:
        await run1.task
    except (asyncio.CancelledError, Exception):
        pass

    app2 = build()                     # a different "process", same journal
    assert run_id in app2.recover()
    await asyncio.sleep(0.05)
    run2 = app2.runs.get(run_id)
    assert run2.status.value == "paused"
    assert side_effects == ["kv"]      # the expensive step did not re-run

    app2.runs.signal(run_id, "approval", {"approved": True})
    await asyncio.wait_for(run2.task, timeout=5)
    assert run2.status.value == "completed"
    types = [e.type for e in run2.log.read(0)]
    assert types.count("state_delta") == 1 and types[-1] == "done"


@pg_only
async def test_postgres_claim_prevents_double_recovery():
    """Two workers recovering at once must not both resume a run."""
    from agentapi.postgres import PostgresBackend
    run_id = f"run_claim_{os.urandom(6).hex()}"

    writer = PostgresBackend(PG_DSN, worker_id="writer")
    writer.create_run(run_id, "/x", {}, tenant=None, metadata={})

    worker_a = PostgresBackend(PG_DSN, worker_id=f"a-{run_id}")
    worker_b = PostgresBackend(PG_DSN, worker_id=f"b-{run_id}")
    claimed_a = [r["id"] for r in worker_a.claim_runs()]
    claimed_b = [r["id"] for r in worker_b.claim_runs()]

    assert run_id in claimed_a
    assert run_id not in claimed_b       # b saw it already claimed
    for backend in (writer, worker_a, worker_b):
        backend.close()


# --- session affinity and prefix-cache routing -----------------------------

def test_session_affinity_is_sticky():
    from agentapi import SessionRouter
    router = SessionRouter(["gpu-0", "gpu-1", "gpu-2"])
    convo = [{"role": "user", "content": "hello " * 100}]

    first = router.route(session_id="s1", messages=convo)
    for turn in range(5):
        convo = convo + [{"role": "user", "content": f"turn {turn}"}]
        assert router.route(session_id="s1", messages=convo).name == first.name
    assert router.sessions["s1"].turns == 6


def test_prefix_cache_routing_beats_round_robin():
    """A cold session whose prompt shares a prefix lands on the holder."""
    from agentapi import SessionRouter
    router = SessionRouter(["gpu-0", "gpu-1", "gpu-2"])
    shared = [{"role": "system", "content": "You are a helpful agent. " * 50}]

    warm = router.route(session_id="warmup", messages=shared)
    # a brand-new session with the same system prompt
    followup = shared + [{"role": "user", "content": "a new question"}]
    landed = router.route(session_id="cold", messages=followup)

    assert landed.name == warm.name          # routed to the cached prefix
    assert router.cache_hit_rate > 0


def test_router_falls_back_to_least_loaded_when_saturated():
    from agentapi import Backend, SessionRouter
    router = SessionRouter([Backend("busy", max_concurrency=1),
                            Backend("free", max_concurrency=4)])
    convo = [{"role": "user", "content": "x" * 200}]

    first = router.route(session_id="s", messages=convo)
    router.acquire(first.name)               # saturate the sticky backend
    second = router.route(session_id="s", messages=convo)
    assert second.name != first.name         # affinity yields to saturation


def test_router_gc_expires_idle_sessions():
    from agentapi import SessionRouter
    router = SessionRouter(["a"], ttl_s=0.0)
    router.route(session_id="old", messages=[{"role": "user", "content": "x"}])
    assert router.gc() == 1
    assert router.sessions == {}


async def test_sessions_endpoint_reports_hit_rate():
    app = AgentAPI(llm=MockLLM())
    app.sessions(["gpu-0", "gpu-1"])
    convo = [{"role": "user", "content": "shared prefix " * 40}]
    app.router.route(session_id="a", messages=convo)
    app.router.route(session_id="b", messages=convo)

    async with client_for(app) as client:
        stats = (await client.get("/sessions")).json()
    assert stats["sessions"] == 2
    assert stats["cache_hit_rate"] > 0
    assert len(stats["backends"]) == 2


# --- OpenTelemetry tracing -------------------------------------------------

async def test_tracing_emits_spans_correlated_by_run():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter)
    from agentapi import instrument

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    app = AgentAPI(llm=MockLLM(script=["hello world"]))

    @app.op
    async def lookup(q: str) -> str:
        """Look something up."""
        return f"found {q}"

    @app.run("/traced")
    async def traced():
        await app.call_op("lookup", {"q": "x"})
        async for tok in ctx.llm.stream(model="mock", messages=[
                {"role": "user", "content": "hi"}]):
            yield Token(text=tok)
        yield Done(result="ok")

    assert instrument(app, tracer_provider=provider) is True

    async with client_for(app) as client:
        async with client.stream("POST", "/traced", json={}) as response:
            run_id = response.headers["x-run-id"]
            await response.aread()
    await asyncio.wait_for(app.runs.get(run_id).task, timeout=2)

    spans = exporter.get_finished_spans()
    run_spans = [s for s in spans if s.name.startswith("run ")]
    assert len(run_spans) == 1
    run_span = run_spans[0]
    assert run_span.attributes["agentapi.run_id"] == run_id
    assert run_span.attributes["agentapi.status"] == "completed"
    assert run_span.attributes["agentapi.output_tokens"] == 2
    assert run_span.attributes["agentapi.tool_calls"] == 1
    # the event log and the trace tell the same story
    event_names = [e.name for e in run_span.events]
    assert "tool_call" in event_names and "done" in event_names
    assert any(s.name == "tool lookup" for s in spans)
    assert any(s.name == "llm mock" for s in spans)


async def test_tracing_marks_failed_runs_as_error():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter)
    from opentelemetry.trace import StatusCode
    from agentapi import instrument

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    app = AgentAPI(llm=MockLLM())
    instrument(app, tracer_provider=provider)

    @app.run("/boom")
    async def boom():
        yield Token(text="before")
        raise RuntimeError("kaboom")

    async with client_for(app) as client:
        async with client.stream("POST", "/boom", json={}) as response:
            await response.aread()

    run_span = [s for s in exporter.get_finished_spans()
                if s.name.startswith("run ")][0]
    assert run_span.status.status_code is StatusCode.ERROR
    assert "kaboom" in run_span.status.description


async def test_tracing_does_not_capture_content_by_default():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter)
    from agentapi import instrument

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    app = AgentAPI(llm=MockLLM(script=["sensitive customer data"]))
    instrument(app, tracer_provider=provider)

    @app.run("/private")
    async def private():
        async for tok in ctx.llm.stream(model="mock", messages=[
                {"role": "user", "content": "secret"}]):
            yield Token(text=tok)
        yield Done()

    async with client_for(app) as client:
        async with client.stream("POST", "/private", json={}) as response:
            await response.aread()

    dumped = json.dumps([
        {"name": s.name,
         "attrs": {k: str(v) for k, v in (s.attributes or {}).items()},
         "events": [{"n": e.name,
                     "a": {k: str(v) for k, v in (e.attributes or {}).items()}}
                    for e in s.events]}
        for s in exporter.get_finished_spans()])
    assert "sensitive" not in dumped         # prompts stay out of traces
