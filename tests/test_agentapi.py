"""End-to-end tests for the agentapi runtime, exercised through the real
ASGI surface with httpx."""
import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from agentapi import (AgentAPI, BudgetExceeded, Done, MockLLM, Skill,
                      StateDelta, Token, ctx, step)

pytestmark = pytest.mark.asyncio


def make_app(**llm_kwargs):
    app = AgentAPI(llm=MockLLM(**llm_kwargs))

    @app.run("/chat", on_disconnect="detach")
    async def chat(prompt: str):
        async for tok in ctx.llm.stream(model="mock", messages=[
                {"role": "user", "content": prompt}]):
            yield Token(text=tok)
        yield Done(result="finished")

    return app



import contextlib
import socket

import uvicorn


@contextlib.asynccontextmanager
async def live_app(app):
    """Serve the app on a real socket: httpx's ASGITransport buffers whole
    responses, so mid-stream reads, disconnects and concurrent signals need
    an actual server."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            yield client
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


def client_for(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def sse_events(response):
    events = []
    current = {}
    async for line in response.aiter_lines():
        if line.startswith("id:"):
            current["id"] = int(line[3:].strip())
        elif line.startswith("event:"):
            current["event"] = line[6:].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[5:].strip())
        elif line == "" and current:
            events.append(current)
            current = {}
    return events


# --- run lifecycle & streaming ------------------------------------------

async def test_run_streams_tokens_and_terminal_done():
    app = make_app(script=["hello world from agentapi"])
    async with client_for(app) as client:
        async with client.stream("POST", "/chat",
                                 json={"prompt": "hi"}) as response:
            assert response.status_code == 201
            run_id = response.headers["x-run-id"]
            events = await sse_events(response)
    types = [e["event"] for e in events]
    assert types[:-1] == ["token"] * (len(types) - 1)
    assert types[-1] == "done"
    text = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert text.strip() == "hello world from agentapi"
    # usage was metered automatically
    done = events[-1]["data"]
    assert done["usage"]["output_tokens"] == 4
    assert done["usage"]["llm_calls"] == 1
    # run is queryable after the stream is gone
    async with client_for(app) as client:
        info = (await client.get(f"/runs/{run_id}")).json()
    assert info["status"] == "completed"


async def test_resume_from_cursor_replays_missed_events():
    app = make_app(script=["a b c d e"])
    async with client_for(app) as client:
        async with client.stream("POST", "/chat",
                                 json={"prompt": "x"}) as response:
            run_id = response.headers["x-run-id"]
            await sse_events(response)
        # late joiner replays from an arbitrary cursor
        resumed = (await client.get(
            f"/runs/{run_id}/events?from=2&stream=false")).json()
        assert resumed["closed"] is True
        assert [e["seq"] for e in resumed["events"]] == list(
            range(2, resumed["next"]))
        # Last-Event-ID reconnect semantics
        async with client.stream("GET", f"/runs/{run_id}/events",
                                 headers={"Last-Event-ID": "1"}) as response:
            events = await sse_events(response)
        assert events[0]["id"] == 2


async def test_detach_run_survives_client_disconnect():
    app = AgentAPI(llm=MockLLM())
    release = asyncio.Event()

    @app.run("/slow", on_disconnect="detach")
    async def slow():
        yield Token(text="one")
        await release.wait()
        yield Token(text="two")
        yield Done(result="survived")

    async with live_app(app) as client:
        async with client.stream("POST", "/slow", json={}) as response:
            run_id = response.headers["x-run-id"]
            # read one event, then disconnect mid-run
            async for _ in response.aiter_lines():
                break
        await asyncio.sleep(0.05)
        run = app.runs.get(run_id)
        assert run.status.value == "running"  # disconnect did NOT cancel
        release.set()
        await asyncio.wait_for(run.task, timeout=2)
        assert run.status.value == "completed"
        # ...and the full stream is replayable after the fact
        replay = (await client.get(
            f"/runs/{run_id}/events?stream=false")).json()
        assert [e["type"] for e in replay["events"]] == [
            "token", "token", "done"]


async def test_cancel_policy_cancels_on_disconnect():
    app = AgentAPI(llm=MockLLM())
    started = asyncio.Event()

    @app.run("/burn", on_disconnect="cancel")
    async def burn():
        yield Token(text="tick")
        started.set()
        await asyncio.sleep(30)
        yield Done()

    async with live_app(app) as client:
        async with client.stream("POST", "/burn", json={}) as response:
            run_id = response.headers["x-run-id"]
            async for _ in response.aiter_lines():
                break
        await asyncio.wait_for(started.wait(), timeout=2)
        run = app.runs.get(run_id)
        await asyncio.wait_for(run.task, timeout=2)
        assert run.status.value == "cancelled"


async def test_idempotency_key_reuses_run():
    app = make_app(script=["only once", "should not happen"])
    async with client_for(app) as client:
        first = await client.post("/chat?stream=false", json={"prompt": "x"},
                                  headers={"Idempotency-Key": "k1"})
        second = await client.post("/chat?stream=false", json={"prompt": "x"},
                                   headers={"Idempotency-Key": "k1"})
    assert first.json()["id"] == second.json()["id"]
    assert len(app.runs.runs) == 1


async def test_validation_error_is_422():
    app = make_app()
    async with client_for(app) as client:
        response = await client.post("/chat", json={})
    assert response.status_code == 422


# --- budgets, deadlines, cancellation ------------------------------------

async def test_budget_exceeded_fails_run_with_typed_error():
    app = AgentAPI(llm=MockLLM(script=["a " * 500],
                               usd_per_output_mtok=1_000_000.0))

    @app.run("/pricey", budget_usd=0.0001)
    async def pricey():
        async for tok in ctx.llm.stream(model="mock", messages=[
                {"role": "user", "content": "go"}]):
            yield Token(text=tok)
        yield Done()

    async with client_for(app) as client:
        async with client.stream("POST", "/pricey", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["kind"] == "budget"


async def test_nested_budget_scope_raises_at_call_site():
    app = AgentAPI(llm=MockLLM())
    caught = {}

    @app.run("/scoped")
    async def scoped():
        try:
            async with ctx.budget(usd=0.000001):
                ctx.charge(usd=0.5)
        except BudgetExceeded as exc:
            caught["error"] = str(exc)
        yield Done(result="handled")

    async with client_for(app) as client:
        async with client.stream("POST", "/scoped", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["event"] == "done"
    assert "budget exceeded" in caught["error"]


async def test_deadline_kills_run():
    app = AgentAPI(llm=MockLLM())

    @app.run("/forever", deadline="0.05s")
    async def forever():
        yield Token(text="start")
        await asyncio.sleep(0.2)
        ctx.check()
        yield Done()

    async with client_for(app) as client:
        async with client.stream("POST", "/forever", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["kind"] == "deadline"


async def test_cancel_endpoint():
    app = AgentAPI(llm=MockLLM())
    started = asyncio.Event()

    @app.run("/long")
    async def long_run():
        yield Token(text="going")
        started.set()
        await asyncio.sleep(30)
        yield Done()

    async with live_app(app) as client:
        async with client.stream("POST", "/long", json={}) as response:
            run_id = response.headers["x-run-id"]
            await asyncio.wait_for(started.wait(), timeout=2)
            cancel = await client.post(f"/runs/{run_id}/cancel")
            assert cancel.json()["ok"] is True
            events = await sse_events(response)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["kind"] == "cancelled"
    run = app.runs.get(run_id)
    await asyncio.wait_for(run.task, timeout=2)
    assert run.status.value == "cancelled"


# --- human-in-the-loop ----------------------------------------------------

async def test_pause_and_signal_resume():
    app = AgentAPI(llm=MockLLM())

    class Approval(BaseModel):
        approved: bool
        note: str = ""

    @app.run("/hitl")
    async def hitl():
        yield StateDelta(data={"stage": "awaiting-approval"})
        approval = await ctx.pause("approval", schema=Approval, timeout="5s")
        yield Done(result={"approved": approval.approved,
                           "note": approval.note})

    async with live_app(app) as client:
        async with client.stream("POST", "/hitl", json={}) as response:
            run_id = response.headers["x-run-id"]
            lines = response.aiter_lines()   # single iterator: httpx forbids
            seen_pause = False               # re-iterating a live stream
            async for line in lines:
                if line.startswith("event: paused"):
                    seen_pause = True
                    break
            assert seen_pause
            await client.post(f"/runs/{run_id}/signals/approval",
                              json={"approved": True, "note": "lgtm"})
            rest = []
            current = {}
            async for line in lines:
                if line.startswith("event:"):
                    current["event"] = line[6:].strip()
                elif line.startswith("data:"):
                    current["data"] = json.loads(line[5:].strip())
                elif line == "" and current:
                    rest.append(current)
                    current = {}
    done = rest[-1]
    assert done["event"] == "done"
    assert done["data"]["result"] == {"approved": True, "note": "lgtm"}
    types = [e.type for e in app.runs.get(run_id).log.read(0)]
    assert "paused" in types and "resumed" in types


# --- steps ---------------------------------------------------------------

async def test_step_memoizes_and_retries():
    app = AgentAPI(llm=MockLLM())
    calls = {"flaky": 0, "fetch": 0}

    @step(retries=2, backoff=0.01)
    async def flaky():
        calls["flaky"] += 1
        if calls["flaky"] < 3:
            raise RuntimeError("transient")
        return "ok"

    @step
    async def fetch(q: str):
        calls["fetch"] += 1
        return f"docs:{q}"

    @app.run("/steppy")
    async def steppy():
        assert await flaky() == "ok"
        first = await fetch("x")
        second = await fetch("x")     # journaled: must not re-execute
        yield Done(result=[first, second])

    async with client_for(app) as client:
        async with client.stream("POST", "/steppy", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["data"]["result"] == ["docs:x", "docs:x"]
    assert calls == {"flaky": 3, "fetch": 1}


# --- ops: one definition, three surfaces ----------------------------------

def build_ops_app():
    app = AgentAPI(llm=MockLLM())

    @app.op(http="POST /tools/search")
    async def search(q: str, limit: int = 3) -> list[str]:
        """Search the corpus."""
        return [f"{q}-{i}" for i in range(limit)]

    return app


async def test_op_http_surface():
    app = build_ops_app()
    async with client_for(app) as client:
        response = await client.post("/tools/search", json={"q": "cats"})
    assert response.json()["result"] == ["cats-0", "cats-1", "cats-2"]


async def test_op_llm_tool_surface_both_styles():
    app = build_ops_app()
    async with client_for(app) as client:
        anthropic = (await client.get("/llm/tools")).json()["tools"]
        openai = (await client.get("/llm/tools?style=openai")).json()["tools"]
    assert anthropic[0]["name"] == "search"
    assert anthropic[0]["description"] == "Search the corpus."
    assert anthropic[0]["input_schema"]["required"] == ["q"]
    assert openai[0]["function"]["parameters"]["properties"]["limit"][
        "default"] == 3


async def test_op_mcp_surface_full_handshake():
    app = build_ops_app()
    async with client_for(app) as client:
        init = (await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}})).json()
        assert init["result"]["serverInfo"]["name"] == "agentapi"
        listed = (await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/list"})).json()
        assert listed["result"]["tools"][0]["name"] == "search"
        called = (await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "search",
                       "arguments": {"q": "dogs", "limit": 2}}})).json()
    payload = json.loads(called["result"]["content"][0]["text"])
    assert payload == ["dogs-0", "dogs-1"]
    assert called["result"]["isError"] is False


async def test_mcp_tool_error_is_content_not_crash():
    app = AgentAPI(llm=MockLLM())

    @app.op
    async def boom() -> str:
        raise ValueError("nope")

    async with client_for(app) as client:
        called = (await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "boom", "arguments": {}}})).json()
    assert called["result"]["isError"] is True
    assert "nope" in called["result"]["content"][0]["text"]


async def test_op_called_inside_run_emits_tool_events():
    app = build_ops_app()

    @app.run("/agentic")
    async def agentic():
        result = await app.call_op("search", {"q": "birds", "limit": 1})
        yield Done(result=result)

    async with client_for(app) as client:
        async with client.stream("POST", "/agentic", json={}) as response:
            events = await sse_events(response)
    types = [e["event"] for e in events]
    assert types == ["tool_call", "tool_result", "done"]
    assert events[1]["data"]["result"] == ["birds-0"]
    assert events[-1]["data"]["usage"]["tool_calls"] == 1


# --- hooks ----------------------------------------------------------------

async def test_hooks_fire_and_failures_are_contained():
    app = make_app(script=["hi there"])
    seen = {"start": 0, "events": [], "end": 0, "llm": 0}

    @app.hook("on_run_start")
    async def on_start(run):
        seen["start"] += 1

    @app.hook("on_event")
    def on_event(run, event):
        seen["events"].append(event.type)

    @app.hook("on_event")
    async def bad_hook(run, event):
        raise RuntimeError("observability bug")   # must not kill the run

    @app.hook("on_run_end")
    async def on_end(run):
        seen["end"] += 1

    @app.hook("on_llm_call")
    async def on_llm(model, params):
        seen["llm"] += 1

    async with client_for(app) as client:
        async with client.stream("POST", "/chat",
                                 json={"prompt": "x"}) as response:
            events = await sse_events(response)
    assert events[-1]["event"] == "done"          # bad hook didn't kill it
    assert seen["start"] == 1 and seen["end"] == 1 and seen["llm"] == 1
    assert seen["events"][-1] == "done"


# --- skills ---------------------------------------------------------------

async def test_skill_mounts_ops_instructions_and_mcp(tmp_path):
    app = AgentAPI(llm=MockLLM())
    research = Skill("web-research", description="Research things.",
                     instructions="Always cite sources.")

    @research.op
    async def summarize(url: str) -> str:
        """Summarize a page."""
        return f"summary of {url}"

    app.include_skill(research)

    # loadable from disk too
    skill_dir = tmp_path / "notes"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "name: notes\ndescription: Note keeping.\n---\nKeep notes short.")
    app.include_skill(Skill.from_dir(skill_dir))

    async with client_for(app) as client:
        skills = (await client.get("/skills")).json()["skills"]
        tools = (await client.get("/llm/tools")).json()["tools"]
        mcp = (await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list"})).json()
        ops = (await client.get("/ops")).json()["ops"]
    assert [s["name"] for s in skills] == ["web-research", "notes"]
    assert skills[1]["instructions"] == "Keep notes short."
    assert "Always cite sources." in app.skills.combined_instructions()
    assert any(t["name"] == "summarize" for t in tools)
    assert any(t["name"] == "summarize" for t in mcp["result"]["tools"])
    assert any(o.get("skill") == "web-research" for o in ops)


# --- pools ----------------------------------------------------------------

async def test_pool_limits_concurrency():
    app = AgentAPI(llm=MockLLM())
    pool = app.pool("mock-backend", concurrency=2)
    peak = {"now": 0, "max": 0}

    async def work():
        async with pool.acquire():
            peak["now"] += 1
            peak["max"] = max(peak["max"], peak["now"])
            await asyncio.sleep(0.02)
            peak["now"] -= 1

    await asyncio.gather(*[work() for _ in range(8)])
    assert peak["max"] == 2


async def test_pool_deadline_aware_shedding():
    from agentapi import PoolSaturated
    pool = Pool = None
    from agentapi.pools import Pool
    pool = Pool("tiny", concurrency=1, avg_latency_s=10.0)

    async def hold():
        async with pool.acquire():
            await asyncio.sleep(0.1)

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    with pytest.raises(PoolSaturated) as excinfo:
        async with pool.acquire(deadline_remaining=0.5):
            pass
    assert excinfo.value.retry_after > 0
    await holder


async def test_pools_endpoint_reports_stats():
    app = AgentAPI(llm=MockLLM())
    app.pool("backend", concurrency=4)
    async with client_for(app) as client:
        stats = (await client.get("/pools")).json()["pools"]
    assert stats[0]["name"] == "backend"
    assert stats[0]["in_flight"] == 0
