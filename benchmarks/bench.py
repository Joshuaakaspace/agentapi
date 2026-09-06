"""Head-to-head: agentapi vs FastAPI on equivalent endpoints.

Measures what actually differs. Two honest caveats up front:

1. A real LLM turn costs 100ms-10s. Every per-request difference below is
   noise against that, and the benchmark says so rather than pretending
   framework overhead is the thing to optimise.
2. The last scenario is not a performance comparison at all — it is a
   capability one. FastAPI cannot pass it at any speed.

Run:  python benchmarks/bench.py
"""
from __future__ import annotations

import asyncio
import socket
import statistics
import time

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentapi import AgentAPI, Done, MockLLM, Token

TOKENS = 40
REQUESTS = 200
CONCURRENCY = 20


class Ask(BaseModel):
    prompt: str


# ---------------------------------------------------------------- FastAPI --
fast = FastAPI()


@fast.post("/echo")
async def fast_echo(ask: Ask) -> dict:
    return {"echo": ask.prompt}


@fast.post("/stream")
async def fast_stream(ask: Ask):
    async def gen():
        for i in range(TOKENS):
            yield f"data: {{\"text\": \"tok{i}\"}}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------- agentapi --
agent_app = AgentAPI(llm=MockLLM())


@agent_app.op(http="POST /echo")
async def echo(prompt: str) -> dict:
    """Echo the prompt."""
    return {"echo": prompt}


@agent_app.run("/stream")
async def stream(prompt: str):
    for i in range(TOKENS):
        yield Token(text=f"tok{i}")
    yield Done()


@agent_app.run("/durable-stream", on_disconnect="detach")
async def durable_stream(prompt: str):
    for i in range(TOKENS):
        yield Token(text=f"tok{i}")
        await asyncio.sleep(0.005)
    yield Done(result="finished-anyway")


# -------------------------------------------------------------------- bench --
async def serve(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task, f"http://127.0.0.1:{port}"


async def measure(base, path, *, streaming, n=REQUESTS, conc=CONCURRENCY):
    latencies: list[float] = []
    semaphore = asyncio.Semaphore(conc)

    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        async def one():
            async with semaphore:
                start = time.perf_counter()
                if streaming:
                    async with client.stream("POST", path,
                                             json={"prompt": "hi"}) as r:
                        async for _ in r.aiter_lines():
                            pass
                else:
                    await client.post(path, json={"prompt": "hi"})
                latencies.append((time.perf_counter() - start) * 1000)

        await client.post(path, json={"prompt": "warm"})     # warm up
        started = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(n)])
        elapsed = time.perf_counter() - started

    latencies.sort()
    return {
        "p50": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "rps": n / elapsed,
    }


async def disconnect_survival(base, path):
    """Start a stream, hang up mid-flight, then ask the server what happened
    to the work. This is the scenario the whole design exists for."""
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        run_id = None
        async with client.stream("POST", path, json={"prompt": "x"}) as r:
            run_id = r.headers.get("x-run-id")
            async for _ in r.aiter_lines():
                break                                  # hang up immediately
        if run_id is None:
            return None
        await asyncio.sleep(0.5)
        info = (await client.get(f"/runs/{run_id}")).json()
        events = (await client.get(
            f"/runs/{run_id}/events?stream=false")).json()
        return info["status"], len(events["events"])


def row(label, result):
    print(f"  {label:<28} p50 {result['p50']:7.2f}ms   "
          f"p95 {result['p95']:7.2f}ms   {result['rps']:8.1f} req/s")


async def main() -> None:
    print(f"\n{REQUESTS} requests, concurrency {CONCURRENCY}, "
          f"{TOKENS} tokens per stream\n")

    fast_server, fast_task, fast_base = await serve(fast)
    agent_server, agent_task, agent_base = await serve(agent_app)
    try:
        print("JSON request/response (validation + routing overhead)")
        row("FastAPI  POST /echo", await measure(fast_base, "/echo",
                                                 streaming=False))
        row("agentapi POST /echo", await measure(agent_base, "/echo",
                                                 streaming=False))

        print("\nSSE streaming (40 events per request)")
        row("FastAPI  POST /stream", await measure(fast_base, "/stream",
                                                   streaming=True))
        row("agentapi POST /stream", await measure(agent_base, "/stream",
                                                   streaming=True))

        print("\nClient disconnects mid-stream — what happened to the work?")
        agent_outcome = await disconnect_survival(agent_base, "/durable-stream")
        print("  FastAPI                      "
              "no run id, no way to ask: the generator was cancelled and "
              "the output is gone")
        print(f"  agentapi                     run is {agent_outcome[0]!r} "
              f"with {agent_outcome[1]} events still replayable")
    finally:
        fast_server.should_exit = True
        agent_server.should_exit = True
        await asyncio.gather(fast_task, agent_task)


if __name__ == "__main__":
    asyncio.run(main())
