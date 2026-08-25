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


# --- agent loop -----------------------------------------------------------

async def test_agent_loop_dispatches_tools_and_finishes():
    app = AgentAPI(llm=MockLLM(script=[
        {"text": "Let me search.",
         "tool_use": [{"name": "search", "input": {"q": "cats", "limit": 1}}]},
        "Cats are documented in [cats-0].",
    ]))

    @app.op
    async def search(q: str, limit: int = 3) -> list[str]:
        """Search the corpus."""
        return [f"{q}-{i}" for i in range(limit)]

    @app.run("/agent")
    async def agent(task: str):
        result = await app.agent(model="mock", messages=[
            {"role": "user", "content": task}], max_turns=5)
        yield Done(result=result)

    async with client_for(app) as client:
        async with client.stream("POST", "/agent",
                                 json={"task": "find cats"}) as response:
            events = await sse_events(response)
    types = [e["event"] for e in events]
    assert types == ["message", "tool_call", "tool_result", "message", "done"]
    assert events[1]["data"]["name"] == "search"
    assert events[2]["data"]["result"] == ["cats-0"]
    result = events[-1]["data"]["result"]
    assert result["turns"] == 2
    assert "cats-0" in result["text"]
    assert events[-1]["data"]["usage"]["llm_calls"] == 2
    assert events[-1]["data"]["usage"]["tool_calls"] == 1


async def test_agent_loop_tool_error_is_fed_back_to_model():
    app = AgentAPI(llm=MockLLM(script=[
        {"tool_use": [{"name": "explode", "input": {}}]},
        "The tool failed, moving on.",
    ]))

    @app.op
    async def explode() -> str:
        raise RuntimeError("kaboom")

    @app.run("/agent")
    async def agent():
        yield Done(result=await app.agent(model="mock", messages=[
            {"role": "user", "content": "go"}]))

    async with client_for(app) as client:
        async with client.stream("POST", "/agent", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["event"] == "done"                 # loop survived
    tool_results = [e for e in events if e["event"] == "tool_result"]
    assert "kaboom" in tool_results[0]["data"]["error"]
    assert events[-1]["data"]["result"]["turns"] == 2


async def test_agent_loop_skill_instructions_become_system_prompt():
    captured = {}

    class SpyLLM(MockLLM):
        async def _complete(self, **kwargs):
            captured.update(kwargs)
            return await super()._complete(
                model=kwargs["model"], messages=kwargs["messages"])

    app = AgentAPI(llm=SpyLLM(script=["done"]))
    skill = Skill("style", instructions="Answer tersely.")
    app.include_skill(skill)

    @app.run("/agent")
    async def agent():
        yield Done(result=await app.agent(model="mock", messages=[
            {"role": "user", "content": "hi"}]))

    async with client_for(app) as client:
        async with client.stream("POST", "/agent", json={}) as response:
            await sse_events(response)
    assert "Answer tersely." in captured["system"]


# --- tier-2 durability ----------------------------------------------------

def durable_app(db_path, side_effects, llm_script=None, token_delay=0.0):
    """Two 'processes' over the same journal are two AgentAPI instances."""
    class DelayLLM(MockLLM):
        async def _stream(self, **kwargs):
            async for tok in super()._stream(**kwargs):
                if token_delay:
                    await asyncio.sleep(token_delay)
                yield tok

    app = AgentAPI(llm=DelayLLM(script=llm_script or ["summary text here"]),
                   durable=str(db_path))

    class Approval(BaseModel):
        approved: bool

    @step
    async def expensive(topic: str) -> str:
        side_effects.append(topic)
        return f"gathered:{topic}"

    @app.run("/pipeline", durability="durable")
    async def pipeline(topic: str):
        gathered = await expensive(topic)
        yield StateDelta(data={"gathered": gathered})
        approval = await ctx.pause("approval", schema=Approval, timeout="60s")
        async for tok in ctx.llm.stream(model="mock", messages=[
                {"role": "user", "content": gathered}]):
            yield Token(text=tok)
        yield Done(result={"approved": approval.approved})

    return app


async def test_durable_run_survives_process_death(tmp_path):
    db = tmp_path / "runs.db"
    side_effects = []

    # -- process 1: run reaches the pause, then the process "dies" ----------
    app1 = durable_app(db, side_effects)
    async with live_app(app1) as client:
        async with client.stream("POST", "/pipeline",
                                 json={"topic": "kv"},
                                 timeout=2) as response:
            run_id = response.headers["x-run-id"]
            async for line in response.aiter_lines():
                if line.startswith("event: paused"):
                    break
    run1 = app1.runs.get(run_id)
    run1.durable = False   # a real crash persists nothing on the way down
    run1.task.cancel()
    try:
        await run1.task
    except (asyncio.CancelledError, Exception):
        pass
    assert side_effects == ["kv"]
    assert app1.backend.get_run(run_id)["status"] in ("paused", "running")

    # -- process 2: fresh instance over the same journal --------------------
    app2 = durable_app(db, side_effects)
    recovered = app2.recover()
    assert recovered == [run_id]
    await asyncio.sleep(0.05)
    run2 = app2.runs.get(run_id)
    assert run2.status.value == "paused"     # replayed back to the pause
    assert side_effects == ["kv"]            # step did NOT re-execute

    app2.runs.signal(run_id, "approval", {"approved": True})
    await asyncio.wait_for(run2.task, timeout=2)
    assert run2.status.value == "completed"

    types = [e.type for e in run2.log.read(0)]
    assert types[0] == "state_delta"         # history continued, not duplicated
    assert types.count("state_delta") == 1
    assert types.count("paused") == 1
    assert types[-1] == "done"
    assert app2.backend.get_run(run_id)["status"] == "completed"


async def test_durable_signal_before_crash_replays_identically(tmp_path):
    db = tmp_path / "runs.db"
    side_effects = []

    app1 = durable_app(db, side_effects, llm_script=["one two three four"],
                       token_delay=0.2)
    async with live_app(app1) as client:
        async with client.stream("POST", "/pipeline",
                                 json={"topic": "x"}, timeout=2) as response:
            run_id = response.headers["x-run-id"]
            async for line in response.aiter_lines():
                if line.startswith("event: paused"):
                    break
    # approval arrives, a token or two streams, THEN the process dies
    app1.runs.signal(run_id, "approval", {"approved": True})
    await asyncio.sleep(0.3)
    run1 = app1.runs.get(run_id)
    events_before_crash = len(run1.log.read(0))
    assert events_before_crash >= 3          # state_delta, paused, resumed...
    run1.durable = False   # abrupt death: no terminal status reaches the DB
    run1.task.cancel()
    try:
        await run1.task
    except (asyncio.CancelledError, Exception):
        pass

    app2 = durable_app(db, side_effects, llm_script=["one two three four"])
    app2.recover()
    run2 = app2.runs.get(run_id)
    await asyncio.wait_for(run2.task, timeout=2)
    # every pre-crash event survived and the stream completed exactly once
    texts = [e.text for e in run2.log.read(0) if e.type == "token"]
    assert "".join(texts).strip() == "one two three four"
    assert run2.status.value == "completed"
    types = [e.type for e in run2.log.read(0)]
    # the pre-crash signal was journaled: the replayed pause resumed with the
    # same payload without any new POST
    assert types.count("paused") == 1 and types.count("resumed") == 1
    assert side_effects == ["x"]
    assert types[-1] == "done"


async def test_archived_run_readable_from_journal_after_restart(tmp_path):
    db = tmp_path / "runs.db"
    app1 = durable_app(db, [])
    async with live_app(app1) as client:
        async with client.stream("POST", "/pipeline",
                                 json={"topic": "y"}, timeout=2) as response:
            run_id = response.headers["x-run-id"]
            async for line in response.aiter_lines():
                if line.startswith("event: paused"):
                    break
        app1.runs.signal(run_id, "approval", {"approved": True})
        run1 = app1.runs.get(run_id)
        await asyncio.wait_for(run1.task, timeout=2)

    # fresh instance, no recovery needed (run finished) — history still there
    app2 = durable_app(db, [])
    assert app2.recover() == []
    async with client_for(app2) as client:
        info = (await client.get(f"/runs/{run_id}")).json()
        events = (await client.get(
            f"/runs/{run_id}/events?stream=false")).json()
    assert info["archived"] is True and info["status"] == "completed"
    assert events["closed"] is True
    assert [e["type"] for e in events["events"]][-1] == "done"


# --- determinism checking -------------------------------------------------

async def test_raw_clock_in_durable_handler_is_rejected(tmp_path):
    import time as time_module

    app = AgentAPI(llm=MockLLM(), durable=str(tmp_path / "d.db"))

    @app.run("/sloppy", durability="durable")
    async def sloppy():
        yield StateDelta(data={"stamped": time_module.time()})  # not replayable
        yield Done()

    async with client_for(app) as client:
        async with client.stream("POST", "/sloppy", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["kind"] == "nondeterminism"
    assert "time.time" in events[-1]["data"]["error"]
    assert "ctx.now()" in events[-1]["data"]["error"]   # names the remedy


async def test_raw_randomness_and_uuid_are_rejected(tmp_path):
    import random as random_module
    import uuid as uuid_module

    for label, offender in (("random", lambda: random_module.random()),
                            ("uuid", lambda: str(uuid_module.uuid4()))):
        app = AgentAPI(llm=MockLLM(), durable=str(tmp_path / f"{label}.db"))

        @app.run("/x", durability="durable")
        async def handler(_offender=offender):
            yield StateDelta(data={"v": _offender()})
            yield Done()

        async with client_for(app) as client:
            async with client.stream("POST", "/x", json={}) as response:
                events = await sse_events(response)
        assert events[-1]["data"]["kind"] == "nondeterminism", label


async def test_ctx_accessors_and_steps_are_allowed(tmp_path):
    import time as time_module

    app = AgentAPI(llm=MockLLM(), durable=str(tmp_path / "ok.db"))

    @step
    async def stamped() -> float:
        return time_module.time()      # fine: a step's result is journaled

    @app.run("/clean", durability="durable")
    async def clean():
        yield StateDelta(data={
            "now": ctx.now(), "uuid": ctx.uuid(), "rand": ctx.random(),
            "step": await stamped(),
        })
        yield Done(result="clean")

    async with client_for(app) as client:
        async with client.stream("POST", "/clean", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["result"] == "clean"


async def test_checker_is_inert_outside_durable_runs():
    import time as time_module

    app = AgentAPI(llm=MockLLM())          # no journal -> nothing to corrupt

    @app.run("/loose")                      # default durability="resumable"
    async def loose():
        yield StateDelta(data={"t": time_module.time()})
        yield Done(result="fine")

    async with client_for(app) as client:
        async with client.stream("POST", "/loose", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["data"]["result"] == "fine"
    # and plain module use outside any run is untouched
    assert isinstance(time_module.time(), float)


async def test_warn_mode_records_violation_without_failing(tmp_path):
    import time as time_module

    app = AgentAPI(llm=MockLLM(), durable=str(tmp_path / "w.db"),
                   determinism="warn")

    @app.run("/noisy", durability="durable")
    async def noisy():
        yield StateDelta(data={"t": time_module.time()})
        yield Done(result="survived")

    with pytest.warns(RuntimeWarning, match="nondeterministic"):
        async with client_for(app) as client:
            async with client.stream("POST", "/noisy", json={}) as response:
                events = await sse_events(response)
    assert events[-1]["data"]["result"] == "survived"


async def test_ctx_now_replays_the_recorded_timestamp(tmp_path):
    """The bug this fixes: ctx.now() used to return live wall clock, so a
    recovered run saw a different 'now' than the execution it replaced."""
    db = tmp_path / "now.db"

    def build():
        app = AgentAPI(llm=MockLLM(), durable=str(db))

        class Approval(BaseModel):
            approved: bool

        @app.run("/stamp", durability="durable")
        async def stamp():
            yield StateDelta(data={"started": ctx.now(), "id": ctx.uuid()})
            approval = await ctx.pause("approval", schema=Approval,
                                       timeout="60s")
            yield Done(result={"started": ctx.now(), "ok": approval.approved})
        return app

    app1 = build()
    async with live_app(app1) as client:
        async with client.stream("POST", "/stamp", json={}) as response:
            run_id = response.headers["x-run-id"]
            async for line in response.aiter_lines():
                if line.startswith("event: paused"):
                    break
    first = app1.runs.get(run_id).log.read(0)[0].data
    app1.runs.get(run_id).durable = False
    app1.runs.get(run_id).task.cancel()
    try:
        await app1.runs.get(run_id).task
    except (asyncio.CancelledError, Exception):
        pass

    await asyncio.sleep(0.01)              # ensure wall clock has moved on
    app2 = build()
    app2.recover()
    run2 = app2.runs.get(run_id)
    app2.runs.signal(run_id, "approval", {"approved": True})
    await asyncio.wait_for(run2.task, timeout=2)
    assert run2.status.value == "completed"

    replayed = run2.log.read(0)[0].data
    assert replayed["started"] == first["started"]   # same clock, not "now"
    assert replayed["id"] == first["id"]             # same id


async def test_replay_divergence_is_detected(tmp_path):
    """Reactive half: even nondeterminism the patches cannot see (here, an
    external mutable) is caught when the replay stops matching history."""
    db = tmp_path / "div.db"
    path_choice = {"value": "left"}

    def build():
        app = AgentAPI(llm=MockLLM(), durable=str(db))

        class Approval(BaseModel):
            approved: bool

        @app.run("/forky", durability="durable")
        async def forky():
            yield StateDelta(data={"branch": path_choice["value"]})
            approval = await ctx.pause("approval", schema=Approval,
                                       timeout="60s")
            yield Done(result=approval.approved)
        return app

    app1 = build()
    async with live_app(app1) as client:
        async with client.stream("POST", "/forky", json={}) as response:
            run_id = response.headers["x-run-id"]
            async for line in response.aiter_lines():
                if line.startswith("event: paused"):
                    break
    app1.runs.get(run_id).durable = False
    app1.runs.get(run_id).task.cancel()
    try:
        await app1.runs.get(run_id).task
    except (asyncio.CancelledError, Exception):
        pass

    path_choice["value"] = "right"        # the world changed under the replay
    app2 = build()
    app2.recover()
    run2 = app2.runs.get(run_id)
    await asyncio.wait_for(run2.task, timeout=2)
    assert run2.status.value == "failed"
    terminal = run2.log.read(0)[-1]
    assert terminal.kind == "nondeterminism"
    assert "diverged" in terminal.error


# --- drain policy ---------------------------------------------------------

async def test_drain_finishes_current_step_then_stops():
    app = AgentAPI(llm=MockLLM())
    progress = []
    in_step = asyncio.Event()
    let_step_finish = asyncio.Event()

    @step
    async def slow_step():
        in_step.set()
        await let_step_finish.wait()
        progress.append("step-finished")
        return "done-work"

    @app.run("/drainable", on_disconnect="drain")
    async def drainable():
        yield Token(text="start")
        await slow_step()
        progress.append("after-step")
        yield Token(text="more")          # drain stops the run here
        progress.append("emitted-more")   # unreachable
        yield Done(result="full")

    async with live_app(app) as client:
        async with client.stream("POST", "/drainable", json={}) as response:
            run_id = response.headers["x-run-id"]
            async for _ in response.aiter_lines():
                break
            await asyncio.wait_for(in_step.wait(), timeout=2)
        await asyncio.sleep(0.05)          # disconnect registered -> drain
        let_step_finish.set()
        run = app.runs.get(run_id)
        await asyncio.wait_for(run.task, timeout=2)

    # The in-flight step ran to completion rather than being interrupted,
    # and the run then stopped at the next checkpoint (the emit) — so no
    # further output reached the log.
    assert progress == ["step-finished", "after-step"]
    assert "emitted-more" not in progress
    assert run.status.value == "completed"  # graceful stop, not an error
    assert [e.type for e in run.log.read(0)] == ["token", "done"]


async def test_ctx_draining_flag_visible_to_handlers():
    app = AgentAPI(llm=MockLLM())
    seen = {}

    @app.run("/loop", on_disconnect="drain")
    async def looper():
        for i in range(100):
            if ctx.draining:               # handler-controlled early exit
                seen["stopped_at"] = i
                break
            yield Token(text=str(i))
            await asyncio.sleep(0.01)
        yield Done(result=seen.get("stopped_at"))

    async with live_app(app) as client:
        async with client.stream("POST", "/loop", json={}) as response:
            run_id = response.headers["x-run-id"]
            async for _ in response.aiter_lines():
                break
        run = app.runs.get(run_id)
        await asyncio.wait_for(run.task, timeout=3)
    assert 0 < seen["stopped_at"] < 100


# --- pool fair queueing and adaptive capacity -----------------------------

async def test_pool_fair_queueing_across_tenants():
    """One tenant flooding the queue must not starve another."""
    from agentapi.pools import Pool
    pool = Pool("shared", concurrency=1)
    order = []

    async def work(tenant, tag):
        async with pool.acquire(tenant=tenant):
            order.append(tag)
            await asyncio.sleep(0.01)

    async with pool.acquire(tenant="hold"):          # occupy the only slot
        tasks = [asyncio.create_task(work("noisy", f"noisy{i}"))
                 for i in range(5)]
        await asyncio.sleep(0.01)
        tasks.append(asyncio.create_task(work("quiet", "quiet0")))
        await asyncio.sleep(0.01)
    await asyncio.gather(*tasks)

    # round robin: the quiet tenant is served second, not after all 5
    assert order.index("quiet0") == 1, order


async def test_pool_adapts_capacity_to_upstream_429():
    from agentapi.pools import Pool
    pool = Pool("adaptive", concurrency=8, recovery_after_s=0.0)
    assert pool.effective_concurrency == 8

    pool.report_upstream_429(retry_after=0.0)
    assert pool.effective_concurrency == 4          # multiplicative decrease
    pool.report_upstream_429(retry_after=0.0)
    assert pool.effective_concurrency == 2
    assert pool.stats()["effective_concurrency"] == 2

    for _ in range(10):                              # additive increase back
        pool.report_success()
    assert pool.effective_concurrency > 2


async def test_llm_429_shrinks_the_pool():
    """A provider 429 during a real call feeds back into admission control."""
    from agentapi.pools import Pool

    class Boom(Exception):
        def __init__(self):
            self.response = type("R", (), {"status_code": 429,
                                           "headers": {"retry-after": "0"}})()

    class FailingLLM(MockLLM):
        async def _complete(self, **kwargs):
            raise Boom()

    pool = Pool("upstream", concurrency=8, recovery_after_s=0.0)
    app = AgentAPI(llm=FailingLLM(pool=pool))

    @app.run("/hits429")
    async def hits429():
        await ctx.llm.complete(model="mock", messages=[
            {"role": "user", "content": "x"}])
        yield Done()

    async with client_for(app) as client:
        async with client.stream("POST", "/hits429", json={}) as response:
            events = await sse_events(response)
    assert events[-1]["event"] == "error"
    assert pool.effective_concurrency == 4      # capacity shrank on the 429


# --- OpenAI-compatible surface --------------------------------------------

def openai_app():
    app = AgentAPI(llm=MockLLM(script=["hello there friend"]))

    @app.run("/chat")
    async def chat(messages: list = None, model: str = "mock"):
        async for tok in ctx.llm.stream(model=model, messages=messages or []):
            yield Token(text=tok)
        yield Done()

    app.openai_compat("/chat")
    return app


async def test_openai_compat_non_streaming():
    app = openai_app()
    async with client_for(app) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "gpt-4o", "messages": [
                {"role": "user", "content": "hi"}]})
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].strip() == "hello there friend"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 3


async def test_openai_compat_streaming_chunks():
    app = openai_app()
    async with live_app(app) as client:
        async with client.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4o", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}) as response:
            payloads = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    payloads.append(line[6:])
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.strip() == "hello there friend"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


