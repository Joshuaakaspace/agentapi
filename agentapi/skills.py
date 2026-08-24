"""Skills: packaged capabilities an app (or its agents) can mount.

A skill bundles:
  * instructions — markdown the agent injects into its system prompt,
  * ops — tools exposed on every surface (HTTP, LLM tool, MCP),
  * hooks — lifecycle instrumentation the skill needs.

Skills can be declared in code or loaded from a directory containing a
``SKILL.md`` (name/description header lines, then instructions), mirroring
the on-disk convention used by Claude Code skills.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Optional

from .hooks import Hooks
from .ops import Op, OpRegistry


class Skill:
    def __init__(self, name: str, *, description: str = "",
                 instructions: str = "") -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self.ops = OpRegistry()
        self.hooks = Hooks()

    def op(self, fn: Optional[Callable[..., Any]] = None, *,
           name: Optional[str] = None, description: Optional[str] = None,
           llm_tool: bool = True, mcp: bool = True) -> Any:
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            self.ops.register(func, name=name, description=description,
                              llm_tool=llm_tool, mcp=mcp)
            return func
        return decorate if fn is None else decorate(fn)

    def hook(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            self.hooks.register(event, func)
            return func
        return decorate

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "ops": [op.name for op in self.ops.all()],
        }

    @classmethod
    def from_dir(cls, path: str | Path) -> "Skill":
        """Load a skill from a directory containing SKILL.md.

        SKILL.md format (frontmatter-lite):
            name: web-research
            description: Search and summarize the web.
            ---
            <markdown instructions injected into the agent's prompt>
        """
        path = Path(path)
        text = (path / "SKILL.md").read_text()
        meta: dict[str, str] = {}
        body = text
        if "---" in text:
            head, _, body = text.partition("---")
            for line in head.strip().splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip().lower()] = value.strip()
        return cls(meta.get("name", path.name),
                   description=meta.get("description", ""),
                   instructions=body.strip())


class SkillSet:
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> Skill:
        if skill.name in self.skills:
            raise ValueError(f"skill {skill.name!r} already mounted")
        self.skills[skill.name] = skill
        return skill

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def all(self) -> list[Skill]:
        return list(self.skills.values())

    def combined_instructions(self) -> str:
        parts = [f"## Skill: {s.name}\n{s.instructions}"
                 for s in self.skills.values() if s.instructions]
        return "\n\n".join(parts)
