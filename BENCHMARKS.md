# agentapi vs FastAPI — measured

Reproduce with `python benchmarks/bench.py` (200 requests, concurrency 20,
40 events per stream, uvicorn on loopback, Python 3.11).

## Where they tie: JSON request/response

| | p50 | p95 | throughput |
|---|---|---|---|
| FastAPI `POST /echo` | 44.96 ms | 70.44 ms | 395.8 req/s |
| agentapi `POST /echo` | 44.43 ms | 73.69 ms | 396.7 req/s |

Same routing and Pydantic validation path, same numbers. An `@app.op` costs
what a FastAPI endpoint costs.

## Where agentapi is slower: SSE streaming

| | p50 | p95 | throughput |
|---|---|---|---|
| FastAPI `POST /stream` | 74.98 ms | 137.18 ms | 237.1 req/s |
| agentapi `POST /stream` | 86.20 ms | 175.15 ms | 196.0 req/s |

**agentapi is ~15% slower at p50 and ~17% lower throughput here, and that is
a real cost, not a rounding error.** FastAPI writes bytes straight to the
socket. agentapi appends each event to a log, assigns it a sequence number,
serialises it through Pydantic, notifies subscribers, then writes — which is
precisely what buys resumability, multiple viewers and replay.

Put it in scale: the overhead is ~11 ms per 40-event stream, about 0.3 ms per
event. A single real LLM turn costs 100 ms–10 s. If your handler talks to a
model, this is between 0.1% and 0.01% of the request.

Don't pay it where it buys nothing — `durability="ephemeral"` skips the log
for routes that are plain request/response.

## Where FastAPI cannot compete: the client hangs up

The benchmark's third scenario starts a stream, disconnects after the first
event, then asks the server what became of the work.

| | outcome |
|---|---|
| FastAPI | No run id to ask about. The generator was cancelled with the connection; the tokens you already paid for are gone. |
| agentapi | Run reports `completed`, with **41 events still replayable** from `GET /runs/{id}/events`. |

This is not a speed difference and no amount of tuning closes it — it is the
Request-vs-Run split the whole design exists for.

## Honest summary

- **Plain JSON APIs:** a tie. Use whichever you like; FastAPI has the bigger
  ecosystem and far more documentation.
- **Streaming throughput:** FastAPI wins by ~15%, and if you are serving
  high-volume streams that carry no state worth keeping, that matters.
- **Anything where losing the work costs money** — LLM generations, agent
  runs, multi-step orchestration: FastAPI has no answer, at any speed.
