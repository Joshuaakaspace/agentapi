"""MCP client: real subprocess servers, HTTP servers, and policy gating."""
import asyncio
import os
import socket
import sys
from pathlib import Path

import httpx
import pytest
import uvicorn

from agentapi import AgentAPI, Done, MockLLM, PermissionDenied, Policy

pytestmark = pytest.mark.asyncio

SERVER = str(Path(__file__).parent / "fixtures" / "echo_mcp_server.py")
STDIO = [sys.executable, SERVER]


def client_for(app, **kwargs):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", **kwargs)


# --- stdio transport against a real subprocess -----------------------------

async def test_connects_and_adopts_remote_tools():
    app = AgentAPI(llm=MockLLM())
    connection = await app.connect_mcp("echo", command=STDIO)
    try:
        assert connection.server_info["name"] == "echo-server"
        assert {t["name"] for t in connection.tools} >= {"echo", "add"}

        # tools arrive namespaced, and on every surface
        assert app.ops.get("echo__echo") is not None
        async with client_for(app) as http:
            llm_tools = {t["name"] for t in
                         (await http.get("/llm/tools")).json()["tools"]}
            listed = (await http.post("/mcp", json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/list"})).json()
            servers = (await http.get("/mcp/servers")).json()["servers"]
        assert "echo__echo" in llm_tools
        assert "echo__add" in {t["name"] for t in listed["result"]["tools"]}
        assert servers[0]["name"] == "echo"
    finally:
        await app.disconnect_mcp("echo")


async def test_calls_a_remote_tool_and_gets_structured_data():
    app = AgentAPI(llm=MockLLM())
    await app.connect_mcp("echo", command=STDIO)
    try:
        echoed = await app.call_op("echo__echo", {"message": "hello"})
        assert echoed == {"echoed": "hello"}
        summed = await app.call_op("echo__add", {"a": 2, "b": 40})
        assert summed == {"sum": 42}
    finally:
        await app.disconnect_mcp("echo")


async def test_remote_schema_is_preserved_and_enforced():
    """The server knows its own arguments better than we could infer."""
    app = AgentAPI(llm=MockLLM())
    await app.connect_mcp("echo", command=STDIO)
    try:
        op = app.ops.get("echo__add")
        assert op.args_schema["required"] == ["a", "b"]
        assert op.args_schema["properties"]["a"]["type"] == "number"
        assert op.description == "Add two numbers."

        with pytest.raises(Exception, match="missing required argument"):
            await app.call_op("echo__add", {"a": 1})
    finally:
        await app.disconnect_mcp("echo")


async def test_remote_tool_failure_is_a_tool_error_not_a_crash():
    app = AgentAPI(llm=MockLLM())
    await app.connect_mcp("echo", command=STDIO)
    try:
        @app.run("/uses-mcp")
        async def uses_mcp():
            try:
                await app.call_op("echo__explode", {})
            except RuntimeError as exc:
                yield Done(result={"handled": str(exc)})
                return
            yield Done(result={"handled": None})

        async with client_for(app) as http:
            async with http.stream("POST", "/uses-mcp", json={}) as response:
                run_id = response.headers["x-run-id"]
                body = (await response.aread()).decode()
        await asyncio.wait_for(app.runs.get(run_id).task, timeout=5)
        assert "tool blew up" in body
        assert "event: done" in body          # the run survived
    finally:
        await app.disconnect_mcp("echo")


async def test_mcp_subprocess_does_not_inherit_host_secrets():
    """A third-party MCP server is untrusted code; it must not be handed
    the API keys sitting in the host's environment."""
    os.environ["HOST_ONLY_SECRET"] = "sk-must-not-leak"
    try:
        app = AgentAPI(llm=MockLLM())
        await app.connect_mcp("echo", command=STDIO)
        try:
            result = await app.call_op("echo__leak_env", {})
            assert result == {"secret": "<absent>"}
        finally:
            await app.disconnect_mcp("echo")
    finally:
        del os.environ["HOST_ONLY_SECRET"]


async def test_explicit_env_is_passed_through():
    app = AgentAPI(llm=MockLLM())
    await app.connect_mcp("echo", command=STDIO,
                          env={"HOST_ONLY_SECRET": "granted"})
    try:
        assert await app.call_op("echo__leak_env", {}) == {"secret": "granted"}
    finally:
        await app.disconnect_mcp("echo")


async def test_missing_server_command_fails_clearly():
    from agentapi.mcp_client import MCPError
    app = AgentAPI(llm=MockLLM())
    with pytest.raises(MCPError, match="not found"):
        await app.connect_mcp("nope", command=["definitely-not-a-real-binary"])


async def test_hanging_server_times_out():
    from agentapi.mcp_client import MCPError
    app = AgentAPI(llm=MockLLM())
    with pytest.raises(MCPError, match="timed out"):
        await app.connect_mcp(
            "sleeper", command=["bash", "-c", "sleep 30"], timeout=0.5)


async def test_duplicate_and_invalid_connections_are_rejected():
    app = AgentAPI(llm=MockLLM())
    await app.connect_mcp("echo", command=STDIO)
    try:
        with pytest.raises(ValueError, match="already connected"):
            await app.connect_mcp("echo", command=STDIO)
        with pytest.raises(ValueError, match="exactly one"):
            await app.connect_mcp("both", command=STDIO, url="http://x/mcp")
    finally:
        await app.disconnect_mcp("echo")


# --- policy applies to remote tools too ------------------------------------

async def test_remote_tools_are_policy_gated():
    app = AgentAPI(llm=MockLLM())
    policy = Policy(default="allow").deny(
        "echo__explode", reason="known to be broken")
    await app.connect_mcp("echo", command=STDIO, policy=policy)
    try:
        assert await app.call_op("echo__echo", {"message": "fine"}) == {
            "echoed": "fine"}
        with pytest.raises(PermissionDenied, match="known to be broken"):
            await app.call_op("echo__explode", {})
    finally:
        await app.disconnect_mcp("echo")


async def test_remote_tool_can_require_human_approval():
    """A third-party tool escalates through the same pause/approve path as
    a local one — and so survives a process restart on a durable run."""
    app = AgentAPI(llm=MockLLM())
    await app.connect_mcp("echo", command=STDIO, policy=Policy(default="ask"))
    try:
        @app.run("/gated")
        async def gated():
            result = await app.call_op("echo__echo", {"message": "sensitive"})
            yield Done(result=result)

        run = app.runs.start("/gated", app._run_routes["/gated"].handler, {},
                             hooks=app.hooks)
        for _ in range(300):
            await asyncio.sleep(0.01)
            if run.status.value == "paused":
                break
        assert run.status.value == "paused"
        asking = next(e for e in run.log.read(0)
                      if e.type == "state_delta").data["awaiting_approval"]
        assert asking["tool"] == "echo__echo"

        app.runs.signal(run.id, "approval", {"approved": True})
        await asyncio.wait_for(run.task, timeout=5)
        assert run.log.read(0)[-1].result == {"echoed": "sensitive"}
    finally:
        await app.disconnect_mcp("echo")


async def test_remote_tool_calls_land_in_the_event_log():
    app = AgentAPI(llm=MockLLM())
    await app.connect_mcp("echo", command=STDIO)
    try:
        @app.run("/logged")
        async def logged():
            await app.call_op("echo__add", {"a": 1, "b": 2})
            yield Done()

        async with client_for(app) as http:
            async with http.stream("POST", "/logged", json={}) as response:
                body = (await response.aread()).decode()
        assert "event: tool_call" in body and "echo__add" in body
        assert '"sum": 3' in body
    finally:
        await app.disconnect_mcp("echo")


# --- HTTP transport, against agentapi's own MCP server ---------------------

async def test_http_transport_against_another_agentapi():
    """Symmetry check: agentapi's MCP client talking to agentapi's MCP
    server, over the real network."""
    provider = AgentAPI(llm=MockLLM(), title="provider")

    @provider.op
    async def multiply(x: int, y: int) -> dict:
        """Multiply two integers."""
        return {"product": x * y}

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(provider, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)

    consumer = AgentAPI(llm=MockLLM(), title="consumer")
    try:
        connection = await consumer.connect_mcp(
            "remote", url=f"http://127.0.0.1:{port}/mcp")
        assert "multiply" in {t["name"] for t in connection.tools}
        result = await consumer.call_op("remote__multiply", {"x": 6, "y": 7})
        assert result == {"product": 42}
    finally:
        await consumer.disconnect_mcp("remote")
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


async def test_http_transport_reports_a_dead_server():
    from agentapi.mcp_client import MCPError
    app = AgentAPI(llm=MockLLM())
    with pytest.raises(MCPError):
        await app.connect_mcp("dead", url="http://127.0.0.1:1/mcp",
                              timeout=2)


# --- sandboxed skill scripts -----------------------------------------------

async def test_skill_scripts_run_inside_the_sandbox(tmp_path):
    from agentapi import attach_harness

    skills = tmp_path / "skills" / "tooling"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "name: tooling\ndescription: Bundled helpers.\n---\nUse count.py.")
    (skills / "count.py").write_text(
        "import sys, os\n"
        "print('cwd', os.getcwd())\n"
        "print('secret', os.environ.get('HOST_ONLY_SECRET', '<absent>'))\n"
        "print('args', sys.argv[1:])\n")

    os.environ["HOST_ONLY_SECRET"] = "sk-must-not-leak"
    try:
        app = AgentAPI(llm=MockLLM())
        harness = attach_harness(app, "/agent", workspace=tmp_path / "ws",
                                 skills_dir=tmp_path / "skills",
                                 policy=Policy(default="allow"),
                                 durability="resumable", network=True)
        assert "run_skill_script" in harness.tool_names

        result = await app.call_op("run_skill_script", {
            "skill": "tooling", "script": "count.py", "args": ["a", "b"]})
    finally:
        del os.environ["HOST_ONLY_SECRET"]

    assert result["exit_code"] == 0
    assert str((tmp_path / "ws").resolve()) in result["stdout"]  # workspace cwd
    assert "secret <absent>" in result["stdout"]                 # env scrubbed
    assert "args ['a', 'b']" in result["stdout"]


async def test_skill_script_cannot_escape_its_skill_directory(tmp_path):
    from agentapi import attach_harness

    skills = tmp_path / "skills" / "s"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("name: s\ndescription: d\n---\nbody")
    (tmp_path / "evil.py").write_text("print('should never run')")

    app = AgentAPI(llm=MockLLM())
    attach_harness(app, "/agent", workspace=tmp_path / "ws",
                   skills_dir=tmp_path / "skills",
                   policy=Policy(default="allow"), durability="resumable")

    with pytest.raises(FileNotFoundError):
        await app.call_op("run_skill_script",
                          {"skill": "s", "script": "../../evil.py"})


async def test_skill_script_is_policy_gated(tmp_path):
    from agentapi import attach_harness

    skills = tmp_path / "skills" / "s"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("name: s\ndescription: d\n---\nbody")
    (skills / "go.sh").write_text("echo ran")

    app = AgentAPI(llm=MockLLM())
    attach_harness(app, "/agent", workspace=tmp_path / "ws",
                   skills_dir=tmp_path / "skills",
                   policy=Policy(default="allow").deny(
                       "run_skill_script", reason="scripts are off"),
                   durability="resumable")

    with pytest.raises(PermissionDenied, match="scripts are off"):
        await app.call_op("run_skill_script",
                          {"skill": "s", "script": "go.sh"})


async def test_skill_script_timeout_is_contained(tmp_path):
    from agentapi import Limits, attach_harness

    skills = tmp_path / "skills" / "s"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("name: s\ndescription: d\n---\nbody")
    (skills / "hang.sh").write_text("sleep 30")

    app = AgentAPI(llm=MockLLM())
    attach_harness(app, "/agent", workspace=tmp_path / "ws",
                   skills_dir=tmp_path / "skills",
                   policy=Policy(default="allow"),
                   limits=Limits(timeout_s=0.4), network=True,
                   durability="resumable")

    result = await app.call_op("run_skill_script",
                               {"skill": "s", "script": "hang.sh"})
    assert result["timed_out"] is True
