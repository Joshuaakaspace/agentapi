"""Sandbox containment, tool behaviour, permission policy and the harness."""
import asyncio
import json
import os

import httpx
import pytest

from agentapi import (
    DEFAULT_SAFE_POLICY,
    AgentAPI,
    Limits,
    MockLLM,
    PathEscape,
    PermissionDenied,
    Policy,
    Sandbox,
    SandboxError,
    Skill,
    attach_harness,
)

pytestmark = pytest.mark.asyncio


def client_for(app, **kwargs):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", **kwargs)


# --- sandbox containment ---------------------------------------------------

async def test_sandbox_runs_commands_in_the_workspace(tmp_path):
    sandbox = Sandbox(tmp_path, network=True)
    result = await sandbox.run("pwd && echo hi")
    assert result.ok
    assert str(tmp_path.resolve()) in result.stdout
    assert "hi" in result.stdout


async def test_sandbox_blocks_path_escape_including_symlinks(tmp_path):
    sandbox = Sandbox(tmp_path / "ws")
    outside = tmp_path / "secret.txt"
    outside.write_text("classified")

    with pytest.raises(PathEscape):
        sandbox.resolve("../secret.txt")
    with pytest.raises(PathEscape):
        sandbox.resolve("/etc/passwd")

    # a symlink planted inside must not become a way out
    link = sandbox.root / "escape"
    link.symlink_to(outside)
    with pytest.raises(PathEscape):
        sandbox.resolve("escape")


async def test_sandbox_kills_on_timeout_and_reaps_children(tmp_path):
    sandbox = Sandbox(tmp_path, network=True)
    result = await sandbox.run("sleep 30", limits=Limits(timeout_s=0.3))
    assert result.timed_out
    assert result.exit_code == -1
    assert result.duration_s < 5          # did not wait out the sleep


async def test_sandbox_scrubs_credentials_from_the_environment(tmp_path):
    os.environ["FAKE_SECRET_KEY"] = "sk-should-not-leak"
    try:
        sandbox = Sandbox(tmp_path, network=True)
        result = await sandbox.run("env")
        assert "sk-should-not-leak" not in result.stdout
        assert "FAKE_SECRET_KEY" not in result.stdout
        assert "PATH=" in result.stdout   # but a usable environment remains
    finally:
        del os.environ["FAKE_SECRET_KEY"]


async def test_sandbox_caps_runaway_output(tmp_path):
    sandbox = Sandbox(tmp_path, network=True,
                      limits=Limits(max_output_bytes=2_000))
    result = await sandbox.run("yes abcdefgh | head -c 200000")
    assert result.truncated
    assert len(result.stdout) < 10_000    # context window protected


async def test_sandbox_enforces_file_size_limit(tmp_path):
    sandbox = Sandbox(tmp_path, limits=Limits(max_file_mb=1))
    with pytest.raises(SandboxError):
        sandbox.write("big.txt", "x" * (2 << 20))


async def test_sandbox_read_write_round_trip(tmp_path):
    sandbox = Sandbox(tmp_path)
    sandbox.write("nested/dir/file.txt", "content here")
    assert sandbox.read("nested/dir/file.txt") == "content here"
    assert sandbox.usage_bytes() > 0


# --- permission policy -----------------------------------------------------

def test_policy_deny_beats_a_broad_allow():
    policy = Policy(default="allow").allow("*").deny(
        "bash", where=r"rm -rf", reason="destructive")
    assert policy.decide("bash", {"command": "ls"}).outcome == "allow"
    denied = policy.decide("bash", {"command": "rm -rf /tmp/x"})
    assert denied.outcome == "deny" and denied.reason == "destructive"


def test_default_safe_policy_shape():
    policy = DEFAULT_SAFE_POLICY
    assert policy.decide("read_file", {"path": "a"}).outcome == "allow"
    assert policy.decide("bash", {"command": "ls -la"}).outcome == "ask"
    assert policy.decide("bash", {"command": "rm -rf /"}).outcome == "deny"
    assert policy.decide("bash", {"command": ":(){ :|:& };:"}).outcome == "deny"


async def test_denied_tool_raises_with_a_reason_for_the_model():
    from agentapi.permissions import enforce
    policy = Policy(default="allow").deny("bash", reason="not in this run")
    with pytest.raises(PermissionDenied, match="not in this run"):
        await enforce(policy, "bash", {"command": "ls"})


async def test_ask_outside_a_run_refuses_rather_than_allowing():
    """A tool needing sign-off must not become free just because there is
    no run to ask on."""
    from agentapi.permissions import enforce
    with pytest.raises(PermissionDenied, match="no run to ask"):
        await enforce(Policy(default="ask"), "bash", {"command": "ls"})


async def test_approval_pauses_the_run_and_resumes_on_signal(tmp_path):
    """The 'ask' path routes through ctx.pause, so the whole
    human-in-the-loop machinery — including durability — applies."""
    app = AgentAPI(llm=MockLLM())
    policy = Policy(default="ask")
    sandbox = Sandbox(tmp_path, network=True)
    from agentapi.tools import register_tools
    register_tools(app, sandbox, policy, include=["bash"])

    @app.run("/needs-approval")
    async def needs_approval():
        result = await app.call_op("bash", {"command": "echo approved-run"})
        yield Done_(result)

    from agentapi import Done

    def Done_(value):
        return Done(result=value["stdout"].strip())

    run = app.runs.start("/needs-approval",
                         app._run_routes["/needs-approval"].handler, {},
                         hooks=app.hooks)
    for _ in range(200):
        await asyncio.sleep(0.01)
        if run.status.value == "paused":
            break
    assert run.status.value == "paused"

    # The tool call is announced first, then the approval request, then the
    # pause — so a reviewer sees both what was attempted and what is asked.
    events = [e.type for e in run.log.read(0)]
    assert events == ["tool_call", "state_delta", "paused"]
    asking = next(e for e in run.log.read(0)
                  if e.type == "state_delta").data["awaiting_approval"]
    assert asking["tool"] == "bash"
    assert "echo approved-run" in json.dumps(asking["arguments"])

    app.runs.signal(run.id, "approval", {"approved": True})
    await asyncio.wait_for(run.task, timeout=5)
    assert run.status.value == "completed"
    assert run.log.read(0)[-1].result == "approved-run"


async def test_declined_approval_stops_the_tool(tmp_path):
    app = AgentAPI(llm=MockLLM())
    sandbox = Sandbox(tmp_path, network=True)
    from agentapi.tools import register_tools
    register_tools(app, sandbox, Policy(default="ask"), include=["bash"])
    marker = tmp_path / "should-not-exist"

    @app.run("/declined")
    async def declined():
        from agentapi import Done
        try:
            await app.call_op("bash", {"command": f"touch {marker}"})
        except PermissionDenied as exc:
            yield Done(result={"denied": str(exc)})
            return
        yield Done(result={"denied": None})

    run = app.runs.start("/declined", app._run_routes["/declined"].handler,
                         {}, hooks=app.hooks)
    for _ in range(200):
        await asyncio.sleep(0.01)
        if run.status.value == "paused":
            break
    app.runs.signal(run.id, "approval",
                    {"approved": False, "reason": "too risky"})
    await asyncio.wait_for(run.task, timeout=5)
    assert "too risky" in run.log.read(0)[-1].result["denied"]
    assert not marker.exists()            # the command never ran


# --- tools -----------------------------------------------------------------

def tools_app(tmp_path, policy=None, include=None):
    app = AgentAPI(llm=MockLLM())
    sandbox = Sandbox(tmp_path, network=True)
    from agentapi.tools import register_tools
    names = register_tools(app, sandbox,
                           policy or Policy(default="allow"), include=include)
    return app, sandbox, names


async def test_tools_are_registered_on_every_surface(tmp_path):
    app, _, names = tools_app(tmp_path)
    assert {"bash", "read_file", "write_file", "edit_file", "grep",
            "glob", "list_dir"} <= set(names)
    async with client_for(app) as client:
        llm_tools = (await client.get("/llm/tools")).json()["tools"]
        mcp = (await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list"})).json()
    exposed = {t["name"] for t in llm_tools}
    assert "bash" in exposed and "edit_file" in exposed
    assert "bash" in {t["name"] for t in mcp["result"]["tools"]}


async def test_file_tools_round_trip(tmp_path):
    app, sandbox, _ = tools_app(tmp_path)
    await app.call_op("write_file", {"path": "a/b.txt", "content": "one\ntwo"})
    read = await app.call_op("read_file", {"path": "a/b.txt"})
    assert "one" in read["content"] and read["lines"] == 2
    assert "     1\t" in read["content"]        # numbered for the model

    listing = await app.call_op("list_dir", {"path": "a"})
    assert listing["entries"][0]["name"] == "b.txt"

    found = await app.call_op("glob", {"pattern": "**/*.txt"})
    assert "a/b.txt" in found["matches"]

    hits = await app.call_op("grep", {"pattern": "two"})
    assert hits["results"][0]["line"] == 2


async def test_read_file_windows_large_files(tmp_path):
    app, sandbox, _ = tools_app(tmp_path)
    sandbox.write("big.txt", "\n".join(f"line{i}" for i in range(1000)))
    page = await app.call_op("read_file",
                             {"path": "big.txt", "offset": 10, "limit": 5})
    assert page["truncated"] is True
    assert "line10" in page["content"] and "line99" not in page["content"]


async def test_edit_file_refuses_ambiguous_matches(tmp_path):
    app, sandbox, _ = tools_app(tmp_path)
    sandbox.write("dup.txt", "target\nmiddle\ntarget\n")

    with pytest.raises(SandboxError, match="2 matches"):
        await app.call_op("edit_file",
                          {"path": "dup.txt", "old": "target", "new": "x"})

    result = await app.call_op("edit_file", {
        "path": "dup.txt", "old": "target", "new": "x", "replace_all": True})
    assert result["replacements"] == 2
    assert sandbox.read("dup.txt") == "x\nmiddle\nx\n"


async def test_edit_file_reports_a_missing_match_usefully(tmp_path):
    app, sandbox, _ = tools_app(tmp_path)
    sandbox.write("f.txt", "hello")
    with pytest.raises(SandboxError, match="read the file first"):
        await app.call_op("edit_file",
                          {"path": "f.txt", "old": "nope", "new": "x"})


async def test_tools_cannot_escape_the_workspace(tmp_path):
    app, _, _ = tools_app(tmp_path / "ws")
    (tmp_path / "outside.txt").write_text("secret")
    with pytest.raises(PathEscape):
        await app.call_op("read_file", {"path": "../outside.txt"})
    with pytest.raises(PathEscape):
        await app.call_op("write_file", {"path": "/etc/evil", "content": "x"})


async def test_tool_calls_appear_in_the_run_event_log(tmp_path):
    app, _, _ = tools_app(tmp_path)
    from agentapi import Done

    @app.run("/uses-tools")
    async def uses_tools():
        await app.call_op("write_file", {"path": "x.txt", "content": "hi"})
        yield Done(result="done")

    async with client_for(app) as client:
        async with client.stream("POST", "/uses-tools", json={}) as response:
            body = (await response.aread()).decode()
    assert "event: tool_call" in body and "event: tool_result" in body


# --- skills: discovery and progressive disclosure --------------------------

def write_skill(root, name, description, body, extra=None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"name: {name}\ndescription: {description}\n---\n{body}")
    for filename, content in (extra or {}).items():
        (directory / filename).write_text(content)
    return directory


def test_skill_discovery_loads_a_directory_tree(tmp_path):
    write_skill(tmp_path, "research", "Research things.", "Cite sources.")
    write_skill(tmp_path, "writing", "Write well.", "Be concise.",
                {"template.md": "# Title"})
    (tmp_path / "broken").mkdir()          # no SKILL.md: skipped, not fatal

    skills = Skill.discover(tmp_path)
    assert [s.name for s in skills] == ["research", "writing"]
    assert skills[1].resources() == ["template.md"]
    assert skills[1].resource("template.md") == "# Title"


def test_skill_resource_cannot_escape_its_directory(tmp_path):
    directory = write_skill(tmp_path, "s", "d", "b")
    (tmp_path / "outside.txt").write_text("secret")
    skill = Skill.from_dir(directory)
    with pytest.raises(FileNotFoundError):
        skill.resource("../outside.txt")


async def test_skills_are_disclosed_progressively(tmp_path):
    """The prompt carries names and descriptions; bodies load on demand."""
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "research", "Research things.",
                "The full instructions are long and detailed.")
    app = AgentAPI(llm=MockLLM())
    harness = attach_harness(app, "/agent", workspace=tmp_path / "ws",
                             skills_dir=skills_dir, durability="resumable")

    prompt = harness.system_prompt()
    assert "research: Research things." in prompt
    assert "full instructions are long" not in prompt   # body withheld

    loaded = await app.call_op("load_skill", {"name": "research"})
    assert "full instructions are long" in loaded["instructions"]

    with pytest.raises(ValueError, match="unknown skill"):
        await app.call_op("load_skill", {"name": "nope"})


# --- the harness -----------------------------------------------------------

async def test_harness_runs_an_agent_over_sandboxed_tools(tmp_path):
    """End to end: the model calls a tool, it executes in the sandbox, and
    the result comes back through the run's event log."""
    app = AgentAPI(llm=MockLLM(script=[
        {"text": "Writing the file.",
         "tool_use": [{"name": "write_file",
                       "input": {"path": "hello.txt", "content": "from agent"}}]},
        "Created hello.txt as requested.",
    ]))
    harness = attach_harness(
        app, "/agent", workspace=tmp_path / "ws",
        policy=Policy(default="allow"), durability="resumable")

    async with client_for(app) as client:
        async with client.stream("POST", "/agent",
                                 json={"task": "create hello.txt"}) as response:
            body = (await response.aread()).decode()

    assert "event: tool_call" in body
    assert (harness.sandbox.root / "hello.txt").read_text() == "from agent"
    assert '"status": "ok"' in body


async def test_harness_surfaces_a_denial_as_a_clean_outcome(tmp_path):
    app = AgentAPI(llm=MockLLM(script=[
        {"tool_use": [{"name": "bash", "input": {"command": "rm -rf /"}}]},
    ]))
    attach_harness(app, "/agent", workspace=tmp_path / "ws",
                   policy=DEFAULT_SAFE_POLICY, durability="resumable")

    async with client_for(app) as client:
        async with client.stream("POST", "/agent",
                                 json={"task": "wipe the disk"}) as response:
            body = (await response.aread()).decode()

    # the model sees the refusal as a tool error and the run still ends well
    assert "not permitted" in body
    assert "event: done" in body


async def test_harness_defaults_to_durable_when_a_journal_exists(tmp_path):
    app = AgentAPI(llm=MockLLM(), durable=str(tmp_path / "h.db"))
    attach_harness(app, "/agent", workspace=tmp_path / "ws")
    assert app._run_routes["/agent"].durability == "durable"
    assert app._run_routes["/agent"].budget_usd == 5.0


async def test_harness_degrades_when_no_journal_is_configured(tmp_path):
    app = AgentAPI(llm=MockLLM())          # no durable= configured
    attach_harness(app, "/agent", workspace=tmp_path / "ws")
    assert app._run_routes["/agent"].durability == "resumable"
