"""Ops: one function definition, many surfaces.

An op declared once becomes, from a single signature + docstring:
  * an HTTP endpoint (with OpenAPI-ish schema),
  * an LLM tool definition (JSON schema the model sees),
  * an MCP tool (tools/list + tools/call),
  * a plain awaitable Python function for in-process agent use.

This kills the silent drift between "the tool the model sees" and "the
function that actually runs".
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from .context import _current
from .events import ToolCall, ToolResult


def _schema_from_signature(fn: Callable[..., Any]) -> tuple[type[BaseModel], dict]:
    """Build a pydantic model (and JSON schema) from a function signature."""
    hints = get_type_hints(fn)
    hints.pop("return", None)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = hints.get(name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)
    model = create_model(f"{fn.__name__}_args", **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)
    return model, schema


@dataclass
class Op:
    name: str
    fn: Callable[..., Any]
    description: str = ""
    args_model: type[BaseModel] = None            # type: ignore[assignment]
    args_schema: dict[str, Any] = field(default_factory=dict)
    http: str | None = None                    # e.g. "POST /tools/search"
    llm_tool: bool = True
    mcp: bool = True

    async def call(self, arguments: dict[str, Any], *, hooks: Any = None,
                   call_id: str | None = None) -> Any:
        """Validated invocation used by every surface. Emits ToolCall /
        ToolResult events when running inside a run context."""
        parsed = self.args_model.model_validate(arguments or {})
        kwargs = {k: getattr(parsed, k) for k in type(parsed).model_fields}
        context = _current.get()
        if context is not None:
            await context.emit(ToolCall(name=self.name, arguments=arguments or {},
                                        call_id=call_id))
            context.charge(tool_calls=1)
        if hooks is not None:
            await hooks.fire("on_tool_call", self.name, kwargs)
        try:
            result = self.fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            if context is not None:
                await context.emit(ToolResult(name=self.name, call_id=call_id,
                                              error=f"{type(exc).__name__}: {exc}"))
            raise
        plain = result.model_dump() if hasattr(result, "model_dump") else result
        if context is not None:
            await context.emit(ToolResult(name=self.name, result=plain,
                                          call_id=call_id))
        return result

    # -- surface exports ----------------------------------------------------
    def llm_tool_def(self) -> dict[str, Any]:
        """Anthropic-style tool definition (name/description/input_schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_schema,
        }

    def openai_tool_def(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema,
            },
        }

    def mcp_tool_def(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.args_schema,
        }


class OpRegistry:
    def __init__(self) -> None:
        self._ops: dict[str, Op] = {}

    def register(self, fn: Callable[..., Any], *, name: str | None = None,
                 description: str | None = None, http: str | None = None,
                 llm_tool: bool = True, mcp: bool = True) -> Op:
        op_name = name or fn.__name__
        if op_name in self._ops:
            raise ValueError(f"op {op_name!r} already registered")
        args_model, args_schema = _schema_from_signature(fn)
        op = Op(
            name=op_name,
            fn=fn,
            description=(description or inspect.getdoc(fn) or "").strip(),
            args_model=args_model,
            args_schema=args_schema,
            http=http,
            llm_tool=llm_tool,
            mcp=mcp,
        )
        self._ops[op_name] = op
        return op

    def get(self, name: str) -> Op | None:
        return self._ops.get(name)

    def all(self) -> list[Op]:
        return list(self._ops.values())

    def llm_tools(self, style: str = "anthropic") -> list[dict[str, Any]]:
        ops = [op for op in self._ops.values() if op.llm_tool]
        if style == "openai":
            return [op.openai_tool_def() for op in ops]
        return [op.llm_tool_def() for op in ops]

    def mcp_tools(self) -> list[dict[str, Any]]:
        return [op.mcp_tool_def() for op in self._ops.values() if op.mcp]
