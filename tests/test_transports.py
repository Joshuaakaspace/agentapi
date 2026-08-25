"""Transport and typed-output tests: WebSocket, ASGI mounting, partial
validation, and the replay CLI."""
import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from agentapi import AgentAPI, Done, MockLLM, StateDelta, Token, complete_json, ctx

pytestmark = pytest.mark.asyncio


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test")


import contextlib
import socket

import uvicorn


@contextlib.asynccontextmanager
async def live_app(app):
    """Serve on a real socket. Starlette's TestClient runs the app in a
    single portal, so a still-streaming SSE response blocks it and websocket
    frames never arrive — an artifact of the harness, not the server."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


async def start_run(host, path, payload=None):
    """POST a run and drop the SSE stream immediately (detach), returning
    the run id from the response headers."""
    async with httpx.AsyncClient(base_url=f"http://{host}", timeout=10) as c:
        async with c.stream("POST", path, json=payload or {}) as response:
            return response.headers["x-run-id"]



# --- WebSocket transport ---------------------------------------------------

async def test_websocket_streams_events_and_accepts_signals():
    """Bidirectional attach: events out, signals back in over one socket —
    the transport SSE cannot provide without a second connection."""
    import websockets

    app = AgentAPI(llm=MockLLM())

    class Approval(BaseModel):
        approved: bool

    @app.run("/ws-hitl")
    async def ws_hitl():
        yield Token(text="working")
        approval = await ctx.pause("approval", schema=Approval, timeout="10s")
        yield Done(result=approval.approved)

    async with live_app(app) as host:
        run_id = await start_run(host, "/ws-hitl")
        async with websockets.connect(f"ws://{host}/ws/runs/{run_id}") as ws:
            assert json.loads(await ws.recv())["type"] == "token"
            assert json.loads(await ws.recv())["type"] == "paused"
            await ws.send(json.dumps({"action": "signal",
                                      "signal": "approval",
                                      "payload": {"approved": True}}))
            assert json.loads(await ws.recv())["type"] == "resumed"
            done = json.loads(await ws.recv())
            assert done["type"] == "done" and done["result"] is True


async def test_websocket_cancel_action():
    import websockets

    app = AgentAPI(llm=MockLLM())

    @app.run("/ws-long")
    async def ws_long():
        yield Token(text="tick")
        await asyncio.sleep(30)
        yield Done()

    async with live_app(app) as host:
        run_id = await start_run(host, "/ws-long")
        async with websockets.connect(f"ws://{host}/ws/runs/{run_id}") as ws:
            assert json.loads(await ws.recv())["type"] == "token"
            await ws.send(json.dumps({"action": "cancel"}))
            terminal = json.loads(await ws.recv())
            assert terminal["type"] == "error"
            assert terminal["kind"] == "cancelled"


# --- mounting another ASGI app (the migration path) ------------------------

async def test_mount_foreign_asgi_app():
    """Adopt agentapi per route instead of rewriting a service."""
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route as SRoute

    legacy = Starlette(routes=[
        SRoute("/health", lambda r: PlainTextResponse("legacy-ok"))])

    app = AgentAPI(llm=MockLLM())

    @app.run("/new")
    async def new():
        yield Done(result="new-ok")

    app.mount("/legacy", legacy)

    async with client_for(app) as client:
        assert (await client.get("/legacy/health")).text == "legacy-ok"
        assert (await client.get("/healthz")).json() == {"ok": True}


# --- typed streaming output (partial validation) ---------------------------

def test_complete_json_repairs_truncation():
    assert complete_json('{"a": 1, "b": "hel') == {"a": 1, "b": "hel"}
    assert complete_json('{"a": [1, 2') == {"a": [1, 2]}
    assert complete_json('{"a": 1, "b"') == {"a": 1}       # dangling key dropped
    assert complete_json('{"a": {"n": 1}, ') == {"a": {"n": 1}}
    backslash = chr(92)                                    # keep escapes legible
    assert complete_json('{"a": "x' + backslash) == {"a": "x"}       # dangling
    assert (complete_json('{"a": "x' + backslash * 2)
            == {"a": "x" + backslash})                               # complete
    assert complete_json('') is None


async def test_stream_as_yields_partials_then_completes():
    class Invoice(BaseModel):
        vendor: str
        total: float

    payload = '{"vendor": "Acme", "total": 42.5}'
    app = AgentAPI(llm=MockLLM(script=[payload]))
    seen = []

    @app.run("/extract")
    async def extract():
        async for partial in ctx.llm.stream_as(Invoice, prompt="extract"):
            seen.append((partial.vendor, partial.total,
                         partial.complete is not None))
        yield Done(result=list(seen[-1][:2]))

    async with client_for(app) as client:
        async with client.stream("POST", "/extract", json={}) as response:
            body = b"".join([c async for c in response.aiter_bytes()])

    assert b'"type": "done"' in body
    assert seen[-1][2] is True                    # last partial validated
    assert seen[-1][0] == "Acme" and seen[-1][1] == 42.5
    assert seen[0][2] is False                    # not complete at the start
    assert any(v is None for v, _, _ in seen)     # vendor arrives mid-stream


async def test_stream_as_repairs_invalid_output():
    class Point(BaseModel):
        x: int
        y: int

    # first attempt is not JSON at all; the repair prompt gets it right
    app = AgentAPI(llm=MockLLM(script=["sorry I cannot", '{"x": 1, "y": 2}']))
    result = {}

    @app.run("/pt")
    async def pt():
        async for partial in ctx.llm.stream_as(Point, prompt="give a point"):
            result["last"] = partial
        yield Done(result="ok")

    async with client_for(app) as client:
        async with client.stream("POST", "/pt", json={}) as response:
            body = b"".join([c async for c in response.aiter_bytes()])

    assert b'"type": "done"' in body
    assert result["last"].complete.x == 1 and result["last"].complete.y == 2


# --- replay CLI ------------------------------------------------------------

async def test_replay_reproduces_a_recorded_run(tmp_path, monkeypatch):
    """A recorded run replays offline against its journaled LLM responses."""
    from agentapi.cli import _replay

    db = tmp_path / "cli.db"
    app = AgentAPI(llm=MockLLM(script=["one two three"]), durable=str(db))

    @app.run("/summarize", durability="durable")
    async def summarize(topic: str):
        yield StateDelta(data={"topic": topic})
        response = await ctx.llm.complete(model="mock", messages=[
            {"role": "user", "content": topic}])
        yield Done(result=response["content"][0]["text"])

    async with client_for(app) as client:
        async with client.stream("POST", "/summarize",
                                 json={"topic": "event logs"}) as response:
            run_id = response.headers["x-run-id"]
            await response.aread()
    await asyncio.wait_for(app.runs.get(run_id).task, timeout=2)

    calls = app.backend.llm_calls(run_id)
    assert len(calls) == 1                       # the exchange was recorded

    # Replay against a provider that would fail if actually called.
    class Exploding(MockLLM):
        async def _complete(self, **kwargs):
            raise AssertionError("replay must not hit the provider")

    app.llm = Exploding()
    matched, diff = await _replay(app, run_id)
    assert matched, diff


async def test_replay_detects_behaviour_change(tmp_path):
    """Change the handler, and replay reports the divergence — a free
    regression test from a production run."""
    from agentapi.cli import _replay

    db = tmp_path / "cli2.db"
    app = AgentAPI(llm=MockLLM(script=["hi"]), durable=str(db))

    @app.run("/flow", durability="durable")
    async def flow():
        yield StateDelta(data={"stage": "one"})
        yield Done(result="done")

    async with client_for(app) as client:
        async with client.stream("POST", "/flow", json={}) as response:
            run_id = response.headers["x-run-id"]
            await response.aread()
    await asyncio.wait_for(app.runs.get(run_id).task, timeout=2)

    # ship a "new version" that emits an extra event
    async def flow_v2():
        yield StateDelta(data={"stage": "one"})
        yield Token(text="new behaviour")
        yield Done(result="done")

    app._run_routes["/flow"].handler = flow_v2
    matched, diff = await _replay(app, run_id)
    assert not matched
    assert any("token" in line for line in diff)
