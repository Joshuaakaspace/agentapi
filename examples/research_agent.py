"""A small research agent showing the pieces working together:

  * a run route with streaming, budget, deadline and detach-on-disconnect
  * ops that are simultaneously HTTP endpoints, LLM tools and MCP tools
  * a skill bundling instructions + a tool
  * journaled steps with retries
  * human-in-the-loop approval
  * an admission-control pool
  * lifecycle hooks for tracing

Run:  uvicorn examples.research_agent:app --port 8000
Then: curl -N localhost:8000/research -X POST -d '{"topic": "event logs"}'

Set ANTHROPIC_API_KEY and swap MockLLM for AnthropicLLM to go live.
"""
from pydantic import BaseModel

from agentapi import (AgentAPI, Done, MockLLM, Skill, StateDelta, Token,
                      ctx, step)

llm = MockLLM(script=[
    "Event logs make streams resumable because every consumer is just a "
    "cursor over an append-only sequence.",
    # scripted agent-loop turns for /agent below
    {"text": "Searching first.",
     "tool_use": [{"name": "search", "input": {"q": "durability"}}]},
    "Based on the passages, durability means the run outlives the process.",
])
# durable="research.db" turns on the tier-2 journal: durable routes survive
# process crashes; call app.recover() at startup to resume them.
app = AgentAPI(title="research-agent", llm=llm, durable="research.db")
backend = app.pool("mock-backend", concurrency=8)


# --- an op: one definition -> HTTP + LLM tool + MCP tool -------------------
@app.op(http="POST /tools/search")
async def search(q: str, limit: int = 3) -> list[str]:
    """Search the corpus for passages relevant to the query."""
    return [f"passage about {q} #{i}" for i in range(limit)]


# --- a skill: instructions + tools as a mountable bundle -------------------
citations = Skill("citations", description="Cite everything.",
                  instructions="Always cite passage ids in square brackets.")


@citations.op
async def cite(passage_id: str) -> str:
    """Format a citation for a passage."""
    return f"[{passage_id}]"


app.include_skill(citations)


# --- a journaled step: retried, memoized within the run --------------------
@step(retries=2, timeout="10s")
async def gather(topic: str) -> list[str]:
    return await search(topic)   # @app.op returns the plain function


class Approval(BaseModel):
    approved: bool
    note: str = ""


# --- the run -----------------------------------------------------------------
@app.run("/research", on_disconnect="detach", deadline="120s",
         budget_usd=0.50, pools=[backend])
async def research(topic: str, require_approval: bool = False):
    passages = await gather(topic)
    yield StateDelta(data={"stage": "gathered", "passages": passages})

    if require_approval:
        # Suspends the run; resume with
        #   POST /runs/{id}/signals/approval  {"approved": true}
        approval = await ctx.pause("approval", schema=Approval, timeout="24h")
        if not approval.approved:
            yield Done(result={"aborted": approval.note})
            return

    prompt = f"Summarize for topic {topic!r}: {passages}"
    async for tok in ctx.llm.stream(model="claude-opus-5", messages=[
            {"role": "user", "content": prompt}]):
        yield Token(text=tok)

    yield Done(result={"topic": topic, "passages": len(passages)})


# --- an agent loop: the model drives the ops -------------------------------
@app.run("/agent", deadline="120s", budget_usd=1.00, durability="durable")
async def agent(task: str):
    result = await app.agent(model="claude-opus-5", messages=[
        {"role": "user", "content": task}], max_turns=8)
    yield Done(result=result)


app.recover()   # resume any durable runs a previous process left unfinished


# --- hooks: tracing without touching handlers --------------------------------
@app.hook("on_run_end")
async def audit(run):
    print(f"[audit] {run.id} {run.status.value} "
          f"${run.ctx.usage.usd:.4f} {run.ctx.usage.output_tokens} tokens")
