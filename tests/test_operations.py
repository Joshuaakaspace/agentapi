"""Operational surface: probes, metrics, shutdown, config, limits."""
import asyncio
import json

import httpx
import pytest

from agentapi import AgentAPI, Done, MockLLM, StateDelta, Token, ctx

pytestmark = pytest.mark.asyncio


def client_for(app, **kwargs):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", **kwargs)


# --- probes ----------------------------------------------------------------

async def test_liveness_ignores_dependencies():
    """A database blip must not make the orchestrator kill a healthy
    process — that turns a brief outage into a crash loop."""
    class BrokenBackend:
        def get_run(self, _run_id):
            raise ConnectionError("database is down")
        def flush(self):
            pass

    app = AgentAPI(llm=MockLLM(), durable=BrokenBackend())
    async with client_for(app) as client:
        live = await client.get("/healthz")
        ready = await client.get("/readyz")

    assert live.status_code == 200 and live.json()["ok"] is True
    assert ready.status_code == 503
    assert "error" in ready.json()["journal"]


async def test_readiness_is_ok_with_a_working_journal(tmp_path):
    app = AgentAPI(llm=MockLLM(), durable=str(tmp_path / "r.db"))
    async with client_for(app) as client:
        ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"ready": True, "draining": False,
                            "journal": "ok", "runs_in_flight": 0,
                            "runs_paused": 0}


async def test_draining_flips_readiness_and_sheds_new_runs():
    app = AgentAPI(llm=MockLLM())

    @app.run("/work")
    async def work():
        yield Done()

    async with client_for(app) as client:
        assert (await client.get("/readyz")).status_code == 200
        app._draining = True                     # as SIGTERM would
        ready = await client.get("/readyz")
        rejected = await client.post("/work?stream=false", json={})

    assert ready.status_code == 503 and ready.json()["draining"] is True
    assert rejected.status_code == 503
    assert rejected.headers["retry-after"] == "5"


# --- graceful shutdown -----------------------------------------------------

async def test_shutdown_waits_for_in_flight_runs():
    app = AgentAPI(llm=MockLLM(), shutdown_grace_s=5)
    finished = []

    @app.run("/slow")
    async def slow():
        yield Token(text="working")
        await asyncio.sleep(0.3)
        finished.append(True)
        yield Done(result="completed cleanly")

    run = app.runs.start("/slow", app._run_routes["/slow"].handler, {},
                         hooks=app.hooks)
    await asyncio.sleep(0.05)
    await app.shutdown()

    assert finished == [True]                    # not killed mid-run
    assert run.status.value == "completed"
    assert app._draining is True


async def test_shutdown_does_not_wait_for_paused_runs():
    """A run parked on a human may sit for hours; blocking a deploy on it
    would make every rolling restart hang."""
    app = AgentAPI(llm=MockLLM(), shutdown_grace_s=30)

    @app.run("/hitl")
    async def hitl():
        yield StateDelta(data={"stage": "waiting"})
        await ctx.pause("approval", timeout="1h")
        yield Done()

    run = app.runs.start("/hitl", app._run_routes["/hitl"].handler, {},
                         hooks=app.hooks)
    for _ in range(200):
        await asyncio.sleep(0.01)
        if run.status.value == "paused":
            break
    assert run.status.value == "paused"

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(app.shutdown(), timeout=5)
    assert loop.time() - started < 2             # returned promptly
    run.task.cancel()


async def test_shutdown_grace_expiry_asks_runs_to_drain():
    app = AgentAPI(llm=MockLLM(), shutdown_grace_s=0.3)

    @app.run("/endless", on_disconnect="drain")
    async def endless():
        for _ in range(1000):
            yield Token(text="tick")
            await asyncio.sleep(0.02)
        yield Done()

    run = app.runs.start("/endless", app._run_routes["/endless"].handler, {},
                         hooks=app.hooks)
    await asyncio.sleep(0.1)
    await asyncio.wait_for(app.shutdown(), timeout=10)
    assert run.ctx.draining is True              # asked to stop cleanly


# --- metrics ---------------------------------------------------------------

async def test_metrics_expose_runs_tokens_and_cost():
    app = AgentAPI(llm=MockLLM(script=["one two three"]))

    @app.op
    async def helper() -> str:
        """A helper."""
        return "ok"

    @app.run("/measured")
    async def measured():
        await app.call_op("helper", {})
        async for tok in ctx.llm.stream(model="mock", messages=[
                {"role": "user", "content": "hi"}]):
            yield Token(text=tok)
        yield Done()

    async with client_for(app) as client:
        async with client.stream("POST", "/measured", json={}) as response:
            run_id = response.headers["x-run-id"]
            await response.aread()
        await asyncio.wait_for(app.runs.get(run_id).task, timeout=2)
        body = (await client.get("/metrics")).text

    assert 'agentapi_runs_started_total{route="/measured"} 1' in body
    assert 'agentapi_runs_finished_total{route="/measured",status="completed"} 1' in body
    assert 'agentapi_tool_calls_total{tool="helper"} 1' in body
    assert 'agentapi_llm_calls_total{model="mock"} 1' in body
    assert 'direction="output"' in body
    assert "agentapi_run_duration_seconds" in body
    assert 'quantile="0.5"' in body
    assert "agentapi_uptime_seconds" in body
    assert "# TYPE agentapi_runs_started_total counter" in body


async def test_metrics_report_live_gauges_and_pools():
    app = AgentAPI(llm=MockLLM())
    app.pool("backend", concurrency=4)
    async with client_for(app) as client:
        body = (await client.get("/metrics")).text
    assert 'agentapi_pool_in_flight{pool="backend"} 0' in body
    assert "agentapi_runs_active 0" in body


async def test_metrics_can_be_disabled():
    app = AgentAPI(llm=MockLLM(), metrics=False)
    async with client_for(app) as client:
        assert (await client.get("/metrics")).status_code == 404


def test_metric_labels_are_escaped():
    from agentapi.observability import Metrics
    metrics = Metrics()
    metrics.incr("agentapi_test_total", route='we"ird\nvalue')
    app = type("App", (), {"runs": type("R", (), {"runs": {}})(),
                           "pools": {}})()
    body = metrics.render(app)
    assert 'we\\"ird value' in body        # quotes escaped, newline flattened


# --- body limits -----------------------------------------------------------

async def test_oversized_body_is_refused_with_413():
    app = AgentAPI(llm=MockLLM(), max_body_bytes=1024)

    @app.run("/small")
    async def small(text: str = ""):
        yield Done(result=len(text))

    async with client_for(app) as client:
        ok = await client.post("/small?stream=false",
                               json={"text": "x" * 100})
        too_big = await client.post("/small?stream=false",
                                    json={"text": "x" * 5000})
    assert ok.status_code == 200
    assert too_big.status_code == 413
    assert "exceeds" in too_big.json()["error"]


# --- configuration ---------------------------------------------------------

def test_config_reads_the_environment(monkeypatch):
    from agentapi.config import Config
    monkeypatch.setenv("AGENTAPI_DURABLE", "runs.db")
    monkeypatch.setenv("AGENTAPI_REQUIRE_AUTH", "true")
    monkeypatch.setenv("AGENTAPI_RATE_LIMIT_PER_MINUTE", "120")
    monkeypatch.setenv("AGENTAPI_REDACT", "yes")
    monkeypatch.setenv("AGENTAPI_LOG_LEVEL", "DEBUG")

    config = Config.from_env()
    assert config.durable == "runs.db"
    assert config.require_auth is True
    assert config.rate_limit_per_minute == 120.0
    assert config.redact is True
    assert config.log_level == "DEBUG"


def test_config_never_logs_dsn_credentials():
    from agentapi.config import Config
    config = Config(durable="postgresql://user:hunter2@db.internal:5432/app",
                    fanout="redis://:secretpass@cache:6379/0")
    described = json.dumps(config.describe())
    assert "hunter2" not in described and "secretpass" not in described
    assert "db.internal:5432/app" in described


def test_from_env_applies_config_and_kwargs_win(monkeypatch):
    monkeypatch.setenv("AGENTAPI_REQUIRE_AUTH", "true")
    monkeypatch.setenv("AGENTAPI_RATE_LIMIT_PER_MINUTE", "30")
    monkeypatch.setenv("AGENTAPI_REDACT", "true")

    app = AgentAPI.from_env(llm=MockLLM(), title="from-env")
    assert app.require_auth is True
    assert app.rate_limit.per_minute == 30.0
    assert app.redactor.__class__.__name__ == "Redactor"
    assert app.title == "from-env"        # explicit kwarg respected

    override = AgentAPI.from_env(llm=MockLLM(), require_auth=False)
    assert override.require_auth is False  # kwargs beat the environment


# --- structured logging ----------------------------------------------------

def test_json_logs_carry_the_run_id():
    import logging

    from agentapi.context import RunContext, _current
    from agentapi.observability import JsonFormatter

    record = logging.LogRecord("test", logging.INFO, __file__, 1,
                               "something happened", (), None)
    formatter = JsonFormatter()

    plain = json.loads(formatter.format(record))
    assert plain["message"] == "something happened" and "run_id" not in plain

    token = _current.set(RunContext("run_abc", tenant="acme"))
    try:
        inside = json.loads(formatter.format(record))
    finally:
        _current.reset(token)
    assert inside["run_id"] == "run_abc" and inside["tenant"] == "acme"


# --- the fix this branch exists for ---------------------------------------

async def test_journal_writes_do_not_block_the_event_loop(tmp_path):
    """Synchronous journal writes used to stall the loop for the whole
    duration of a durable run — freezing every other request on the worker.
    """
    app = AgentAPI(llm=MockLLM(), durable=str(tmp_path / "b.db"))

    @app.run("/burst", durability="durable")
    async def burst():
        for i in range(60):
            yield Token(text=f"t{i}")
        yield Done()

    ticks = 0
    stop = False

    async def heartbeat():
        nonlocal ticks
        while not stop:
            await asyncio.sleep(0)
            ticks += 1
            await asyncio.sleep(0.001)

    beat = asyncio.create_task(heartbeat())
    run = app.runs.start("/burst", app._run_routes["/burst"].handler, {},
                         durable=True, hooks=app.hooks)
    await asyncio.wait_for(run.task, timeout=20)
    stop = True
    beat.cancel()

    assert ticks > 5, f"loop was starved: only {ticks} ticks during the run"
    assert len(app.backend.events(run.id)) == 61   # durability intact
