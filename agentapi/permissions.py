"""Tool permission policy, with approval that survives a crash.

Sandboxing bounds what a tool *can* do. Policy decides what it *may* do —
and for the middle ground, asks a human.

    policy = Policy(default="ask")
    policy.allow("read_file", "list_dir", "grep")
    policy.deny("bash", where=r"\\brm\\s+-rf\\b")
    policy.ask("bash", "write_file")

Three outcomes: ``allow`` runs it, ``deny`` refuses with a reason the model
can read and route around, ``ask`` escalates to a human.

The escalation is the interesting part: it goes through ``ctx.pause()``, so
on a durable run the process may **exit** while the approval is pending and
resume where it left off when the answer arrives. Approval prompts that
expire because a worker restarted are the usual reason teams give up on
human-in-the-loop; here that failure mode does not exist.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from re import Pattern
from typing import Any

from pydantic import BaseModel


class PermissionDenied(Exception):
    """Policy refused the call. The message is written for the model."""


class Decision(BaseModel):
    """A human's answer to an approval request."""

    approved: bool
    reason: str = ""
    remember: bool = False       # apply to later calls of this tool


@dataclass
class Rule:
    tool: str                    # exact name, or "*"
    outcome: str                 # allow | deny | ask
    where: Pattern[str] | None = None   # matched against the arguments
    reason: str = ""

    def matches(self, tool: str, arguments: dict[str, Any]) -> bool:
        if self.tool not in ("*", tool):
            return False
        if self.where is None:
            return True
        return bool(self.where.search(_argument_text(arguments)))


def _argument_text(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(arguments, default=str, sort_keys=True)
    except TypeError:
        return repr(arguments)


@dataclass
class Policy:
    """Ordered rules; the first match wins, else ``default``."""

    default: str = "ask"
    rules: list[Rule] = field(default_factory=list)
    _remembered: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default not in ("allow", "deny", "ask"):
            raise ValueError("default must be allow|deny|ask")

    # -- rule building ------------------------------------------------------
    def _add(self, outcome: str, tools: Iterable[str], where: str | None,
             reason: str) -> Policy:
        pattern = re.compile(where) if where else None
        for tool in tools:
            self.rules.append(Rule(tool, outcome, pattern, reason))
        return self

    def allow(self, *tools: str, where: str | None = None,
              reason: str = "") -> Policy:
        return self._add("allow", tools, where, reason)

    def deny(self, *tools: str, where: str | None = None,
             reason: str = "") -> Policy:
        return self._add("deny", tools, where, reason)

    def ask(self, *tools: str, where: str | None = None,
            reason: str = "") -> Policy:
        return self._add("ask", tools, where, reason)

    # -- evaluation ---------------------------------------------------------
    def decide(self, tool: str, arguments: dict[str, Any]) -> Rule:
        """First matching rule, else the default. Deny rules are checked
        first so a broad ``allow("*")`` cannot accidentally outrank a
        specific ``deny``."""
        matching = [rule for rule in self.rules
                    if rule.matches(tool, arguments)]
        for rule in matching:
            if rule.outcome == "deny":
                return rule
        if matching:
            return matching[0]
        if self._remembered.get(tool):
            return Rule(tool, "allow", reason="remembered approval")
        return Rule(tool, self.default)

    def remember(self, tool: str) -> None:
        self._remembered[tool] = True


DEFAULT_SAFE_POLICY = (
    Policy(default="ask")
    .allow("read_file", "list_dir", "glob", "grep")
    .deny("bash", where=r"\brm\s+-rf\s+/(?!\w)", reason="refuses to wipe the filesystem root")
    .deny("bash", where=r":\(\)\s*\{.*\};\s*:", reason="fork bomb")
)


async def enforce(policy: Policy, tool: str, arguments: dict[str, Any], *,
                  signal_name: str = "approval",
                  timeout: str = "1h") -> None:
    """Apply the policy, escalating to a human when the rule says ``ask``.

    Raises ``PermissionDenied`` on refusal; returns None when the call may
    proceed. Import-time cost is nil when no run is active, so this is safe
    to call from tools used outside a run.
    """
    rule = policy.decide(tool, arguments)
    if rule.outcome == "allow":
        return
    if rule.outcome == "deny":
        raise PermissionDenied(
            f"tool {tool!r} is not permitted"
            + (f": {rule.reason}" if rule.reason else ""))

    from .context import get_ctx
    try:
        ctx = get_ctx()
    except RuntimeError:
        # No run to pause: refuse rather than silently allowing. A tool that
        # would have needed sign-off must not become free just because it
        # was invoked outside a run.
        raise PermissionDenied(
            f"tool {tool!r} requires approval, but there is no run to ask on"
        ) from None

    # Say what is being approved before pausing: ctx.pause emits the schema
    # of the answer, not the question, and a reviewer needs the question.
    from .events import StateDelta
    await ctx.emit(StateDelta(data={
        "awaiting_approval": {"tool": tool, "arguments": arguments,
                              "reason": rule.reason}}))
    decision = await ctx.pause(signal_name, schema=Decision, timeout=timeout)
    if not decision.approved:
        raise PermissionDenied(
            f"a human declined {tool!r}"
            + (f": {decision.reason}" if decision.reason else ""))
    if decision.remember:
        policy.remember(tool)
