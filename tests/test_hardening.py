"""Redaction, rate limiting and cross-worker fanout."""
import asyncio
import json
import os

import httpx
import pytest

from agentapi import (
    AgentAPI,
    Done,
    MockLLM,
    Principal,
    RateLimit,
    RateLimited,
    Redactor,
    StateDelta,
    Token,
    bearer_tokens,
    ctx,
    step,
)

pytestmark = pytest.mark.asyncio

REDIS_URL = os.environ.get("AGENTAPI_TEST_REDIS", "redis://localhost:56379/0")


def client_for(app, **kwargs):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", **kwargs)


# --- redaction -------------------------------------------------------------

def test_redactor_masks_known_secret_shapes():
    redactor = Redactor()
    masked = redactor.text(
        "key sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA and mail bob@corp.com")
    assert "sk-ant" not in masked and "bob@corp.com" not in masked
    assert "[redacted:ANTHROPIC_KEY]" in masked
    assert "[redacted:EMAIL]" in masked
    assert redactor.redactions == 2


def test_redactor_preserves_structure_and_drops_named_fields():
    redactor = Redactor(fields={"password"})
    out = redactor.value({"password": "hunter2",
                          "nested": {"card": "4111 1111 1111 1111"},
                          "list": ["ok", "ghp_AAAAAAAAAAAAAAAAAAAA"],
                          "count": 3})
    assert out["password"] == "[redacted:FIELD]"
    assert "4111" not in out["nested"]["card"]
    assert "[redacted:GITHUB_TOKEN]" in out["list"][1]
    assert out["list"][0] == "ok"
    assert out["count"] == 3                 # non-strings pass through


def test_redactor_never_rewrites_structural_event_fields():
    """seq/ts/type drive replay matching; rewriting them would corrupt it."""
    redactor = Redactor()
    payload = {"type": "token", "seq": 7, "ts": 1.5,
               "text": "my key is sk-ant-api03-BBBBBBBBBBBBBBBBBBBB"}
    out = redactor.event(payload)
    assert out["seq"] == 7 and out["ts"] == 1.5 and out["type"] == "token"
    assert "sk-ant" not in out["text"]


async def test_journal_is_redacted_but_live_stream_is_not(tmp_path):
    """Redaction happens on the way into the journal; the caller who sent
    the content still sees it on their own stream."""
    secret = "sk-ant-api03-CCCCCCCCCCCCCCCCCCCCCC"
    app = AgentAPI(llm=MockLLM(), durable=str(tmp_path / "r.db"),
                   redactor=Redactor())

    @step
    async def remember(value: str) -> str:
        return f"stored {value}"

    @app.run("/leaky", durability="durable")
    async def leaky(token: str):
        await remember(token)
        yield StateDelta(data={"echo": token})
        yield Done(result="ok")

    async with client_for(app) as client:
        async with client.stream("POST", "/leaky",
                                 json={"token": secret}) as response:
            run_id = response.headers["x-run-id"]
            body = (await response.aread()).decode()
    await asyncio.wait_for(app.runs.get(run_id).task, timeout=2)

    assert secret in body                      # live stream: unredacted
    stored = json.dumps(app.backend.events(run_id))
    assert secret not in stored                # journal: redacted
    assert "ANTHROPIC_KEY" in stored
    assert secret not in json.dumps(app.backend.steps(run_id))


async def test_redacted_journal_still_recovers(tmp_path):
    """A redacted journal must still be usable for crash recovery."""
    db = tmp_path / "rr.db"
    calls = []

    def build():
        app = AgentAPI(llm=MockLLM(), durable=str(db), redactor=Redactor())

        @step
        async def once(topic: str) -> str:
            calls.append(topic)
            return f"did {topic}"

        @app.run("/pipe", durability="durable")
        async def pipe(topic: str):
            result = await once(topic)
            yield StateDelta(data={"result": result})
            await ctx.pause("go", timeout="60s")
            yield Done(result=result)
        return app

    app1 = build()
    run = app1.runs.start("/pipe", app1._run_routes["/pipe"].handler,
                          {"topic": "x"}, durable=True, hooks=app1.hooks)
    for _ in range(100):
        await asyncio.sleep(0.01)
        if run.status.value == "paused":
            break
    assert calls == ["x"]
    run.durable = False
    run.task.cancel()
    try:
        await run.task
    except (asyncio.CancelledError, Exception):
        pass

    app2 = build()
    assert run.id in app2.recover()
    await asyncio.sleep(0.05)
    app2.runs.signal(run.id, "go", {})
    await asyncio.wait_for(app2.runs.get(run.id).task, timeout=2)
    assert app2.runs.get(run.id).status.value == "completed"
    assert calls == ["x"]                      # step still memoized


# --- rate limiting ---------------------------------------------------------

def test_token_bucket_refills_over_time():
    limiter = RateLimit(per_minute=60, burst=2)
    limiter.check("t:acme")
    limiter.check("t:acme")
    with pytest.raises(RateLimited) as excinfo:
        limiter.check("t:acme")
    assert 0 < excinfo.value.retry_after <= 1.01


def test_limits_are_per_key_not_global():
    limiter = RateLimit(per_minute=60, burst=1)
    limiter.check("t:acme")
    limiter.check("t:globex")                  # unaffected by acme
    with pytest.raises(RateLimited):
        limiter.check("t:acme")


def test_key_is_tenant_when_present_else_principal():
    limiter = RateLimit()
    assert limiter.key_for(Principal(id="a", tenant="acme")) == "t:acme"
    assert limiter.key_for(Principal(id="solo")) == "p:solo"


async def test_run_creation_is_rate_limited_with_retry_after():
    app = AgentAPI(llm=MockLLM(), rate_limit=RateLimit(per_minute=60, burst=2))

    @app.run("/cheap")
    async def cheap():
        yield Done(result="ok")

    async with client_for(app) as client:
        assert (await client.post("/cheap?stream=false", json={})
                ).status_code == 200
        assert (await client.post("/cheap?stream=false", json={})
                ).status_code == 200
        blocked = await client.post("/cheap?stream=false", json={})
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1


async def test_per_route_limit_overrides_the_app_default():
    app = AgentAPI(llm=MockLLM(), rate_limit=RateLimit(per_minute=600, burst=50))

    @app.run("/expensive", rate_limit=RateLimit(per_minute=60, burst=1))
    async def expensive():
        yield Done()

    @app.run("/normal")
    async def normal():
        yield Done()

    async with client_for(app) as client:
        assert (await client.post("/expensive?stream=false", json={})
                ).status_code == 200
        assert (await client.post("/expensive?stream=false", json={})
                ).status_code == 429
        # the generous app-wide limit still applies elsewhere
        assert (await client.post("/normal?stream=false", json={})
                ).status_code == 200


async def test_tenants_do_not_share_a_rate_limit_budget():
    app = AgentAPI(llm=MockLLM(), rate_limit=RateLimit(per_minute=60, burst=1))
    app.authenticator(bearer_tokens({
        "tok-a": Principal(id="a", tenant="acme"),
        "tok-b": Principal(id="b", tenant="globex"),
    }))

    @app.run("/chat")
    async def chat():
        yield Done()

    async with client_for(app) as client:
        alice = {"authorization": "Bearer tok-a"}
        bob = {"authorization": "Bearer tok-b"}
        assert (await client.post("/chat?stream=false", json={}, headers=alice)
                ).status_code == 200
        assert (await client.post("/chat?stream=false", json={}, headers=alice)
                ).status_code == 429
        assert (await client.post("/chat?stream=false", json={}, headers=bob)
                ).status_code == 200      # bob's budget is his own


# --- cross-worker fanout ---------------------------------------------------

def redis_available() -> bool:
    try:
        import redis
        redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
        return True
    except Exception:
        return False


redis_only = pytest.mark.skipif(not redis_available(),
                                reason="no redis at AGENTAPI_TEST_REDIS")


@redis_only
async def test_fanout_publishes_and_replays_by_cursor():
    from agentapi.fanout import RedisFanout

    fanout = RedisFanout(REDIS_URL)
    run_id = f"run_fan_{os.urandom(6).hex()}"
    for index in range(5):
        event = Token(text=f"t{index}")
        event.seq = index
        await fanout.publish(run_id, event)
    await fanout.publish(run_id, _stamped(Done(result="fin"), 5))

    everything = await fanout.history(run_id)
    assert [e.type for e in everything] == ["token"] * 5 + ["done"]
    later = await fanout.history(run_id, from_seq=3)
    assert [e.seq for e in later] == [3, 4, 5]
    await fanout.close()


def _stamped(event, seq):
    event.seq = seq
    return event


@redis_only
async def test_non_owning_worker_serves_the_same_stream(tmp_path):
    """The scaling fix: worker B answers for a run worker A owns."""
    from agentapi.fanout import RedisFanout

    db = str(tmp_path / "fan.db")
    suffix = os.urandom(4).hex()

    def build():
        app = AgentAPI(llm=MockLLM(), durable=db, fanout=RedisFanout(REDIS_URL))

        @app.run(f"/work-{suffix}", durability="durable")
        async def work():
            yield Token(text="alpha")
            yield Token(text="beta")
            yield Done(result="finished")
        return app

    worker_a = build()
    worker_b = build()                        # separate process, same journal

    async with client_for(worker_a) as client:
        async with client.stream("POST", f"/work-{suffix}",
                                 json={}) as response:
            run_id = response.headers["x-run-id"]
            await response.aread()
    await asyncio.wait_for(worker_a.runs.get(run_id).task, timeout=2)

    assert worker_b.runs.get(run_id) is None  # B never owned it
    async with client_for(worker_b) as client:
        async with client.stream("GET", f"/runs/{run_id}/events") as response:
            assert response.headers["x-served-by"] == "fanout"
            body = (await response.aread()).decode()

    assert "alpha" in body and "beta" in body
    assert "event: done" in body
    await worker_a.fanout.close()
    await worker_b.fanout.close()


@redis_only
async def test_fanout_failure_never_breaks_the_run():
    """Fanout is a convenience for other workers; losing it must not fail
    the run that produced the event."""
    from agentapi.fanout import RedisFanout

    broken = RedisFanout("redis://127.0.0.1:1/0")   # nothing listening
    app = AgentAPI(llm=MockLLM(), fanout=broken)

    @app.run("/resilient")
    async def resilient():
        yield Token(text="still works")
        yield Done(result="ok")

    async with client_for(app) as client:
        async with client.stream("POST", "/resilient", json={}) as response:
            body = (await response.aread()).decode()
    assert "still works" in body and '"result": "ok"' in body
    await broken.close()
