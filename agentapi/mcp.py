"""MCP surface: expose the app's ops as an MCP server.

Implements the Streamable HTTP transport's POST half of MCP (JSON-RPC 2.0):
``initialize``, ``tools/list``, ``tools/call``, ``ping``. Any MCP client
(Claude Code/Desktop, an agent framework) can mount this app at ``/mcp``
and call the same ops the HTTP and LLM surfaces use — one definition,
zero drift.
"""
from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = "2025-06-18"


class MCPServer:
    def __init__(self, name: str, ops: Any, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self.ops = ops

    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch a single JSON-RPC message; returns the response object
        (or None for notifications)."""
        msg_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params") or {}

        if method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": params.get("protocolVersion",
                                                  PROTOCOL_VERSION),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": self.version},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.ops.mcp_tools()}
            elif method == "tools/call":
                result = await self._call_tool(params)
            else:
                return _error(msg_id, -32601, f"method not found: {method}")
        except Exception as exc:  # noqa: BLE001 - surfaced as JSON-RPC error
            return _error(msg_id, -32603, f"{type(exc).__name__}: {exc}")

        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        op = self.ops.get(name)
        if op is None or not op.mcp:
            raise ValueError(f"unknown tool: {name}")
        try:
            result = await op.call(params.get("arguments") or {})
        except Exception as exc:  # noqa: BLE001 - MCP tool errors are content
            return {
                "content": [{"type": "text",
                             "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        text = result if isinstance(result, str) else __import__("json").dumps(
            result, default=str)
        return {"content": [{"type": "text", "text": text}], "isError": False}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}
