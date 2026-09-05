"""The harness: a working agent in one call.

Everything needed for a useful agent already exists in this package —
sandbox, tools, policy, skills, the model loop, durability, budgets — but
wiring them together is a page of boilerplate every project would copy.

    app.harness("/agent", workspace="./work", policy=DEFAULT_SAFE_POLICY)

That single call gives a durable run route whose handler runs a real agent
loop over sandboxed tools, gated by policy, with skills disclosed
progressively, budgets and deadlines enforced, every tool call in the event
log, and human approvals that survive a process restart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .events import Done, Message, StateDelta
from .permissions import DEFAULT_SAFE_POLICY, PermissionDenied, Policy
from .sandbox import Limits, Sandbox
from .skills import Skill
from .tools import register_tools

# Scripts run through an explicit interpreter rather than relying on the
# executable bit: a skill copied out of a zip or a git checkout on Windows
# routinely arrives without it, and a confusing "permission denied" is a
# worse outcome than choosing the interpreter from the extension.
_INTERPRETERS = {".py": ["python3"], ".sh": ["bash"], ".js": ["node"],
                 ".rb": ["ruby"]}


def _interpreter_for(path: Path) -> list[str]:
    return _INTERPRETERS.get(path.suffix, ["bash"])

SYSTEM_PROMPT = """You are an agent working inside a sandboxed workspace.

Rules that matter:
- Every path you use is relative to the workspace. You cannot read or write
  outside it, and the environment holds no credentials.
- Read a file before editing it, and make `old` text unique enough to match
  exactly once.
- Some tools need a human's approval. If a call is declined, do not retry it
  in a different shape — explain what you need and why.
- Prefer the smallest action that makes progress, then check the result.
"""


class Harness:
    """A configured agent: workspace + tools + policy + skills."""

    def __init__(self, app: Any, *, workspace: str | Path,
                 policy: Policy | None = None,
                 limits: Limits | None = None,
                 tools: list[str] | None = None,
                 skills_dir: str | Path | None = None,
                 network: bool = False,
                 system: str | None = None,
                 approval_timeout: str = "1h") -> None:
        self.app = app
        self.sandbox = Sandbox(workspace, limits=limits, network=network)
        self.policy = policy or DEFAULT_SAFE_POLICY
        self.system = system or SYSTEM_PROMPT
        self.approval_timeout = approval_timeout
        self.tool_names = register_tools(
            app, self.sandbox, self.policy, include=tools,
            approval_timeout=approval_timeout)

        if skills_dir is not None:
            for skill in Skill.discover(skills_dir):
                app.include_skill(skill)
            self._register_skill_loader()

    async def _gate(self, tool: str, arguments: dict) -> None:
        from .permissions import enforce
        await enforce(self.policy, tool, arguments,
                      timeout=self.approval_timeout)

    def _register_skill_loader(self) -> None:
        """Expose skills through a tool rather than the system prompt.

        Progressive disclosure: the prompt lists names and descriptions, and
        the body is fetched only when a task needs it — a dozen skills'
        full text would otherwise crowd out the actual conversation.
        """
        app = self.app

        @app.op(name="load_skill")
        async def load_skill(name: str) -> dict:
            """Load a skill's full instructions and list its bundled files."""
            skill = app.skills.get(name)
            if skill is None:
                available = [s.name for s in app.skills.all()]
                raise ValueError(f"unknown skill {name!r}; available: {available}")
            return {"name": skill.name, "instructions": skill.instructions,
                    "resources": skill.resources()}

        @app.op(name="read_skill_resource")
        async def read_skill_resource(skill: str, path: str) -> dict:
            """Read a file bundled with a skill (a template, script or data)."""
            found = app.skills.get(skill)
            if found is None:
                raise ValueError(f"unknown skill {skill!r}")
            return {"skill": skill, "path": path,
                    "content": found.resource(path)}

        harness = self

        @app.op(name="run_skill_script")
        async def run_skill_script(skill: str, script: str,
                                   args: list[str] | None = None) -> dict:
            """Run an executable script bundled with a skill, sandboxed.

            Skills often ship a helper — a linter, a formatter, a converter —
            that is far better executed than reimplemented by the model each
            time. The script runs under the same containment as every other
            command: resource limits, a scrubbed environment and the
            workspace as its working directory.
            """
            await harness._gate("run_skill_script",
                                {"skill": skill, "script": script})
            found = app.skills.get(skill)
            if found is None:
                raise ValueError(f"unknown skill {skill!r}")
            # resource() refuses paths outside the skill directory.
            found.resource(script)
            source = (found.path / script).resolve()
            interpreter = _interpreter_for(source)
            result = await harness.sandbox.run(
                [*interpreter, str(source), *(args or [])])
            return {"skill": skill, "script": script, **result.as_dict()}

        self.tool_names += ["load_skill", "read_skill_resource",
                            "run_skill_script"]

    async def connect_mcp(self, name: str, **kwargs: Any) -> Any:
        """Attach an MCP server's tools to this harness, policy-gated."""
        connection = await self.app.connect_mcp(name, policy=self.policy,
                                                **kwargs)
        self.tool_names += [f"{name}__{t['name']}" for t in connection.tools
                            if t.get("name")]
        return connection

    def system_prompt(self) -> str:
        catalog = self.app.skills.catalog()
        return f"{self.system}\n\n{catalog}" if catalog else self.system

    async def run(self, task: str, *, model: str = "claude-opus-5",
                  max_turns: int = 20, **params: Any):
        """Drive the agent loop for one task, yielding run events."""
        yield StateDelta(data={"task": task, "workspace": str(self.sandbox.root),
                               "tools": self.tool_names})
        try:
            result = await self.app.agent(
                model=model,
                messages=[{"role": "user", "content": task}],
                system=self.system_prompt(),
                tools=self.tool_names,
                max_turns=max_turns,
                **params)
        except PermissionDenied as exc:
            # A refusal is an outcome, not a crash: the run ends cleanly with
            # what was refused, so the caller can decide what to do about it.
            yield Message(role="assistant", content=str(exc))
            yield Done(result={"status": "denied", "detail": str(exc)})
            return
        yield Done(result={"status": "ok", **result})


def attach_harness(app: Any, path: str, *, workspace: str | Path,
                   durability: str = "durable",
                   deadline: str = "30m",
                   budget_usd: float | None = 5.0,
                   **kwargs: Any) -> Harness:
    """Build a Harness and mount it as a run route.

    Durable by default: an agent that loses an hour of work to a deploy is
    the failure this whole runtime exists to prevent.
    """
    harness = Harness(app, workspace=workspace, **kwargs)

    if durability == "durable" and app.backend is None:
        durability = "resumable"        # no journal configured; degrade

    @app.run(path, durability=durability, deadline=deadline,
             budget_usd=budget_usd, on_disconnect="detach")
    async def agent_route(task: str, model: str = "claude-opus-5",
                          max_turns: int = 20):
        async for event in harness.run(task, model=model, max_turns=max_turns):
            yield event

    return harness
