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

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .hooks import Hooks
from .ops import OpRegistry


class Skill:
    def __init__(self, name: str, *, description: str = "",
                 instructions: str = "", path: Path | None = None) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self.path = path            # on-disk root, when loaded from a dir
        self.ops = OpRegistry()
        self.hooks = Hooks()

    # -- progressive disclosure ---------------------------------------------
    @property
    def summary(self) -> str:
        """One line for the system prompt.

        A dozen skills' full instructions would crowd out the conversation,
        so the prompt carries names and descriptions, and the agent calls
        ``load_skill`` for the body only when a task actually needs it.
        """
        return f"- {self.name}: {self.description}"

    def resources(self) -> list[str]:
        """Files bundled alongside SKILL.md (templates, scripts, data)."""
        if self.path is None:
            return []
        return sorted(
            str(p.relative_to(self.path)) for p in self.path.rglob("*")
            if p.is_file() and p.name != "SKILL.md")

    def resource(self, name: str) -> str:
        """Read one bundled resource, refusing paths outside the skill."""
        if self.path is None:
            raise FileNotFoundError(f"skill {self.name!r} has no files")
        target = (self.path / name).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise FileNotFoundError(f"{name} is outside skill {self.name!r}")
        return target.read_text()

    def op(self, fn: Callable[..., Any] | None = None, *,
           name: str | None = None, description: str | None = None,
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
            "resources": self.resources(),
        }

    @classmethod
    def from_dir(cls, path: str | Path) -> Skill:
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
                   instructions=body.strip(),
                   path=path)

    @classmethod
    def discover(cls, root: str | Path) -> list[Skill]:
        """Load every skill under ``root`` (each a dir holding SKILL.md).

        Mirrors the convention Claude Code uses, so a skills directory can
        be shared between the two.
        """
        root = Path(root)
        if not root.is_dir():
            return []
        found = []
        for skill_file in sorted(root.glob("*/SKILL.md")):
            try:
                found.append(cls.from_dir(skill_file.parent))
            except Exception:  # noqa: BLE001 - one bad skill must not
                continue       # stop the rest from loading
        return found


class SkillSet:
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> Skill:
        if skill.name in self.skills:
            raise ValueError(f"skill {skill.name!r} already mounted")
        self.skills[skill.name] = skill
        return skill

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def all(self) -> list[Skill]:
        return list(self.skills.values())

    def combined_instructions(self) -> str:
        parts = [f"## Skill: {s.name}\n{s.instructions}"
                 for s in self.skills.values() if s.instructions]
        return "\n\n".join(parts)

    def catalog(self) -> str:
        """The names-and-descriptions form for the system prompt."""
        if not self.skills:
            return ""
        listing = "\n".join(s.summary for s in self.skills.values())
        return ("Available skills (call load_skill for the full "
                f"instructions before using one):\n{listing}")
