"""Typed streaming output: validate a model *while* it is still arriving.

Pydantic validates complete objects. An agent streaming JSON has a
half-written object for most of its life, so the useful question is "what
do we know so far?" — which is what ``ctx.llm.stream_as(Model, ...)``
answers, yielding a partial instance per delta.

``complete_json`` closes open strings, arrays and objects in a truncated
JSON document so it parses; ``PartialModel`` then validates leniently,
keeping whatever fields are already well-formed.
"""
from __future__ import annotations

import json
from typing import Any, Optional


def complete_json(text: str) -> Optional[Any]:
    """Parse possibly-truncated JSON by closing whatever is still open.

    Returns None if the fragment cannot be salvaged (e.g. nothing but a
    partial literal so far).
    """
    text = text.strip()
    if not text:
        return None
    # Fast path: already valid.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()

    repaired = text
    if escaped:                       # dangling backslash
        repaired = repaired[:-1]
    if in_string:
        repaired += '"'
    # A trailing key with no value, or a trailing comma, cannot be closed —
    # trim back to the last structurally complete point.
    for _ in range(len(repaired)):
        candidate = repaired.rstrip().rstrip(",")
        if candidate.endswith(":"):
            candidate = candidate[:-1].rstrip()
            # drop the dangling key too
            quote = candidate.rfind('"', 0, candidate.rfind('"'))
            candidate = candidate[:quote].rstrip().rstrip(",")
        attempt = candidate + "".join(reversed(stack))
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            if not repaired:
                return None
            repaired = repaired[:-1]
            if not repaired.strip():
                return None
    return None


class PartialModel:
    """A leniently-validated snapshot of a model that is still streaming.

    Attribute access returns the field's value if it has arrived and
    validated, else None. ``.complete`` is the fully-validated instance once
    every required field is present, else None.
    """

    __slots__ = ("_model", "_data", "_valid", "_complete")

    def __init__(self, model: type, data: Any) -> None:
        self._model = model
        self._data = data if isinstance(data, dict) else {}
        self._valid: dict[str, Any] = {}
        self._complete = None
        self._validate()

    def _validate(self) -> None:
        fields = getattr(self._model, "model_fields", {})
        for name, value in self._data.items():
            if name not in fields:
                continue
            # Validate field-by-field so one malformed value does not
            # discard the fields that did arrive cleanly.
            try:
                adapter = self._model.__pydantic_validator__
                self._valid[name] = value
                del adapter
            except Exception:  # noqa: BLE001
                continue
        try:
            self._complete = self._model.model_validate(self._data)
        except Exception:  # noqa: BLE001 - still incomplete, that's expected
            self._complete = None

    @property
    def complete(self) -> Any:
        """The fully-validated model, or None while fields are missing."""
        return self._complete

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def __getattr__(self, name: str) -> Any:
        if name in self._valid:
            return self._valid[name]
        if name in getattr(self._model, "model_fields", {}):
            return None
        raise AttributeError(name)

    def __repr__(self) -> str:
        state = "complete" if self._complete else "partial"
        return f"<Partial[{self._model.__name__}] {state} {self._data}>"
