"""A sandboxed coding agent in about twenty lines.

    uvicorn examples.agent:app --port 8000

    curl -N localhost:8000/agent -X POST -H 'content-type: application/json' \
         -d '{"task": "create hello.txt containing a haiku"}'

Everything the runtime provides applies automatically: the run survives a
client disconnect, streams resumably, is journaled for crash recovery,
enforces a dollar budget and a deadline, and pauses for human approval on
anything the policy does not allow outright.

Swap MockLLM for AnthropicLLM (and set ANTHROPIC_API_KEY) to run it live.
"""
from pathlib import Path

from agentapi import AgentAPI, MockLLM, Policy, attach_harness

HERE = Path(__file__).parent

app = AgentAPI(
    title="coding-agent",
    llm=MockLLM(script=[
        {"text": "I'll write the file.",
         "tool_use": [{"name": "write_file",
                       "input": {"path": "hello.txt",
                                 "content": "sandboxed run —\nno keys, no escape,\nthe journal remembers."}}]},
        "Done: hello.txt now holds a haiku.",
    ]),
    durable="agent.db",          # runs survive a crash or deploy
)

# Reading is free; writing and shell need a human's sign-off. Because the
# ask goes through ctx.pause(), the process may restart while an approval
# is pending and the run picks up exactly where it left off.
policy = (
    Policy(default="ask")
    .allow("read_file", "list_dir", "glob", "grep", "load_skill",
           "read_skill_resource")
    .deny("bash", where=r"\brm\s+-rf\s+/(?!\w)", reason="destructive")
)

harness = attach_harness(
    app, "/agent",
    workspace=HERE / "workspace",
    policy=policy,
    skills_dir=HERE / "skills",   # progressive disclosure: names in the
    deadline="10m",               # prompt, bodies loaded on demand
    budget_usd=2.00,
)

app.recover()   # resume anything a previous process left unfinished


@app.hook("on_run_end")
async def audit(run):
    usage = run.ctx.usage
    print(f"[audit] {run.id} {run.status.value} "
          f"${usage.usd:.4f} · {usage.tool_calls} tool calls")
