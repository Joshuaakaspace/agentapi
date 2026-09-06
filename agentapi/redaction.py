"""Redaction for the journal.

The journal is the most useful thing in this system and the most dangerous:
it stores prompts, tool arguments, model output and step results verbatim,
on disk, indefinitely. That is a retention and compliance problem long
before it is an engineering one, so redaction is applied on the way *in* —
what never reaches the journal cannot leak from it.

    app = AgentAPI(durable="runs.db", redactor=Redactor())

A ``Redactor`` rewrites event payloads, step results and recorded LLM calls
before they are persisted. The live event stream is untouched: attached
clients still see real text, because they are the caller who supplied it.

Defaults cover the credentials that most often end up pasted into prompts.
Add your own with ``pattern()``, or subclass for structured rules.

**Replay still works.** Redaction is deliberately *shape-preserving*: a
redacted string stays a string, so a replayed run takes the same branches.
What it cannot do is reproduce output that depended on the secret's actual
value — a redacted journal is for recovery and audit, not for bit-exact
reproduction of a run that echoed a key.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from re import Pattern
from typing import Any

# Ordered most-specific first: a token that also looks like a generic
# hex blob should be labelled by its provider, not by its shape.
DEFAULT_PATTERNS: list[tuple[str, str]] = [
    (r"sk-ant-[A-Za-z0-9_\-]{16,}", "ANTHROPIC_KEY"),
    (r"sk-[A-Za-z0-9]{20,}", "OPENAI_KEY"),
    (r"gh[pousr]_[A-Za-z0-9]{16,}", "GITHUB_TOKEN"),
    (r"AKIA[0-9A-Z]{16}", "AWS_ACCESS_KEY"),
    (r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}", "JWT"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
     "PRIVATE_KEY"),
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "EMAIL"),
    (r"\b(?:\d[ \-]*?){13,16}\b", "CARD_NUMBER"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
]


class Redactor:
    """Rewrites sensitive substrings before they reach the journal."""

    def __init__(self, *, patterns: Iterable[tuple[str, str]] | None = None,
                 use_defaults: bool = True,
                 fields: Iterable[str] = (),
                 mask: Callable[[str, str], str] | None = None) -> None:
        self._patterns: list[tuple[Pattern[str], str]] = []
        if use_defaults:
            for expression, label in DEFAULT_PATTERNS:
                self._patterns.append((re.compile(expression), label))
        for expression, label in (patterns or ()):
            self._patterns.append((re.compile(expression), label))
        # Whole fields to drop regardless of content (e.g. "password").
        self.fields = set(fields)
        self._mask = mask or (lambda label, _value: f"[redacted:{label}]")
        self.redactions = 0

    def pattern(self, expression: str, label: str) -> Redactor:
        """Add a rule. Returns self so rules can be chained."""
        self._patterns.append((re.compile(expression), label))
        return self

    # -- core ---------------------------------------------------------------
    def text(self, value: str) -> str:
        for expression, label in self._patterns:
            value, count = expression.subn(
                lambda match, label=label: self._mask(label, match.group(0)),
                value)
            self.redactions += count
        return value

    def value(self, value: Any, *, key: str | None = None) -> Any:
        """Redact any JSON-shaped value, preserving its structure."""
        if key is not None and key.lower() in self.fields:
            self.redactions += 1
            return self._mask("FIELD", str(value))
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.value(v, key=k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        return value

    # -- journal hooks ------------------------------------------------------
    def event(self, payload: dict[str, Any]) -> dict[str, Any]:
        redacted = self.value(payload)
        # seq/ts/type are structural: never rewrite them, or replay's
        # fingerprint matching breaks in confusing ways.
        for structural in ("seq", "ts", "type"):
            if structural in payload:
                redacted[structural] = payload[structural]
        return redacted

    def step(self, result: Any) -> Any:
        return self.value(result)

    def llm_call(self, request: dict[str, Any],
                 response: dict[str, Any]) -> tuple[dict, dict]:
        return self.value(request), self.value(response)


class NullRedactor:
    """Explicitly store everything. Named so that choosing it is visible in
    code review, rather than being the silent default."""

    redactions = 0

    def event(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def step(self, result: Any) -> Any:
        return result

    def llm_call(self, request: dict[str, Any],
                 response: dict[str, Any]) -> tuple[dict, dict]:
        return request, response
