"""The agent loop: model <-> ops until the model stops asking for tools.

``app.agent(...)`` runs an Anthropic-format tool loop inside the current
run. Every op the app (and its skills) exposes is offered to the model;
tool dispatch goes through ``Op.call`` so ToolCall/ToolResult events land
in the run's event log automatically, usage is metered, and budgets and
deadlines apply to every model turn.
"""
from __future__ import annotations

import json
from typing import Any

from .context import get_ctx
from .events import Message


async def agent_loop(app: Any, *, model: str,
                     messages: list[dict[str, Any]],
                     system: str | None = None,
                     tools: list[str] | None = None,
                     max_turns: int = 10,
                     **params: Any) -> dict[str, Any]:
    """Run the tool loop. ``tools`` is a list of op names (None = all ops
    marked llm_tool, including skill ops). Returns the final assistant text
    plus turn/usage accounting."""
    ctx = get_ctx()
    tool_defs = app.llm_tools("anthropic")
    if tools is not None:
        wanted = set(tools)
        tool_defs = [t for t in tool_defs if t["name"] in wanted]
    if system is None and app.skills.all():
        instructions = app.skills.combined_instructions()
        system = instructions or None

    conversation = list(messages)
    final_text = ""
    turns = 0
    for turns in range(1, max_turns + 1):
        ctx.check()
        request: dict[str, Any] = {"model": model, "messages": conversation}
        if system:
            request["system"] = system
        if tool_defs:
            request["tools"] = tool_defs
        response = await ctx.llm.complete(**request, **params)

        content = response.get("content", [])
        text = "".join(block.get("text", "") for block in content
                       if block.get("type") == "text")
        if text:
            final_text = text
            await ctx.emit(Message(role="assistant", content=text))

        tool_uses = [block for block in content
                     if block.get("type") == "tool_use"]
        if not tool_uses:
            return {"text": final_text, "turns": turns,
                    "stop_reason": response.get("stop_reason", "end_turn")}

        conversation.append({"role": "assistant", "content": content})
        results = []
        for use in tool_uses:
            call_id = use.get("id")
            try:
                out = await app.call_op(use["name"], use.get("input") or {},
                                        call_id=call_id)
                if hasattr(out, "model_dump"):
                    out = out.model_dump()
                payload = out if isinstance(out, str) else json.dumps(
                    out, default=str)
                results.append({"type": "tool_result",
                                "tool_use_id": call_id, "content": payload})
            except Exception as exc:  # noqa: BLE001 - model sees the error
                results.append({"type": "tool_result",
                                "tool_use_id": call_id,
                                "content": f"{type(exc).__name__}: {exc}",
                                "is_error": True})
        conversation.append({"role": "user", "content": results})

    return {"text": final_text, "turns": turns, "stop_reason": "max_turns"}
