"""MCP client: borrow tools from other servers.

agentapi already *serves* MCP. This is the other half — connecting out to
external MCP servers and folding their tools into an agent's toolset, so a
harness can use a filesystem server, a database server, or your own
internal one without wrapping each by hand.

    await app.connect_mcp("github", command=["npx", "-y",
                                             "@modelcontextprotocol/server-github"])
    await app.connect_mcp("internal", url="https://tools.internal/mcp")

Discovered tools become ordinary ops named ``server__tool``, which means
they inherit everything the rest of the runtime gives a tool: permission
policy (including human approval), event-log entries, usage accounting,
budgets and deadlines. A third-party tool is not more trusted than a local
one — arguably less — so it goes through the same gate.

Two transports: **stdio**, which launches the server as a subprocess under
resource limits, and **HTTP**, which POSTs JSON-RPC. Both isolate failure:
a server that dies, hangs or returns garbage degrades to a tool error the
model can read, never an exception that kills the run.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import Sequence
from typing import Any

from .sandbox import Limits

logger = logging.getLogger("agentapi.mcp_client")

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TIMEOUT = 30.0


class MCPError(Exception):
    """The remote server failed, timed out, or refused the call."""


class MCPTransport:
    async def request(self, method: str, params: dict[str, Any] | None,
                      *, timeout: float) -> Any:
        raise NotImplementedError

    async def notify(self, method: str,
                     params: dict[str, Any] | None = None) -> None:
        return None

    async def close(self) -> None:
        return None


class StdioTransport(MCPTransport):
    """Newline-delimited JSON-RPC over a subprocess's stdin/stdout.

    The subprocess is launched with a scrubbed environment and resource
    limits: an MCP server is third-party code, and it should not inherit
    the API keys sitting in the host's environment.
    """

    def __init__(self, command: Sequence[str], *,
                 env: dict[str, str] | None = None,
                 cwd: str | None = None,
                 limits: Limits | None = None,
                 inherit_env: Sequence[str] = ("PATH", "HOME", "LANG")) -> None:
        self.command = list(command)
        self.cwd = cwd
        self.limits = limits or Limits(timeout_s=DEFAULT_TIMEOUT)
        self._env = {name: os.environ[name] for name in inherit_env
                     if name in os.environ}
        self._env.update(env or {})
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if shutil.which(self.command[0]) is None:
            raise MCPError(f"MCP server command not found: {self.command[0]}")

        def preexec() -> None:  # pragma: no cover - child process
            import resource
            os.setsid()
            for which, values in self.limits.as_rlimits():
                try:
                    resource.setrlimit(which, values)
                except (ValueError, OSError):
                    pass

        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env, cwd=self.cwd, preexec_fn=preexec)

    async def request(self, method: str, params: dict[str, Any] | None,
                      *, timeout: float = DEFAULT_TIMEOUT) -> Any:
        if self._process is None or self._process.returncode is not None:
            raise MCPError("MCP server is not running")
        # One in-flight request at a time: the stdio transport is a single
        # pipe, and interleaving replies would require correlating ids
        # across concurrent readers for no practical gain here.
        async with self._lock:
            self._next_id += 1
            message = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
            if params is not None:
                message["params"] = params
            payload = (json.dumps(message) + "\n").encode()
            self._process.stdin.write(payload)
            await self._process.stdin.drain()
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=timeout)
            except TimeoutError as exc:
                raise MCPError(
                    f"MCP server timed out after {timeout}s on {method}") from exc
        if not line:
            raise MCPError("MCP server closed the connection")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MCPError(f"MCP server sent invalid JSON: {line[:200]!r}") from exc
        if "error" in response:
            raise MCPError(str(response["error"].get("message", response["error"])))
        return response.get("result")

    async def notify(self, method: str,
                     params: dict[str, Any] | None = None) -> None:
        if self._process is None or self._process.stdin.is_closing():
            return
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._process.stdin.write((json.dumps(message) + "\n").encode())
        await self._process.stdin.drain()

    async def close(self) -> None:
        if self._process is None:
            return
        try:
            self._process.stdin.close()
        except (BrokenPipeError, AttributeError):
            pass
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except (TimeoutError, ProcessLookupError):
            try:
                os.killpg(os.getpgid(self._process.pid), 9)
            except (ProcessLookupError, PermissionError, OSError):
                self._process.kill()
        self._process = None


class HTTPTransport(MCPTransport):
    """JSON-RPC over the MCP Streamable HTTP transport's POST half."""

    def __init__(self, url: str, *,
                 headers: dict[str, str] | None = None) -> None:
        self.url = url
        self.headers = {"content-type": "application/json",
                        "accept": "application/json, text/event-stream",
                        **(headers or {})}
        self._next_id = 0
        self._client = None
        self._session: str | None = None

    async def _http(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        return self._client

    async def request(self, method: str, params: dict[str, Any] | None,
                      *, timeout: float = DEFAULT_TIMEOUT) -> Any:
        client = await self._http()
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id,
                                   "method": method}
        if params is not None:
            message["params"] = params
        headers = dict(self.headers)
        if self._session:
            headers["mcp-session-id"] = self._session
        try:
            response = await client.post(self.url, json=message,
                                         headers=headers, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - network failures are data
            raise MCPError(f"MCP request failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise MCPError(f"MCP server returned HTTP {response.status_code}")
        session = response.headers.get("mcp-session-id")
        if session:
            self._session = session
        body = _decode_http_body(response)
        if body is None:
            return None
        if "error" in body:
            raise MCPError(str(body["error"].get("message", body["error"])))
        return body.get("result")

    async def notify(self, method: str,
                     params: dict[str, Any] | None = None) -> None:
        client = await self._http()
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        headers = dict(self.headers)
        if self._session:
            headers["mcp-session-id"] = self._session
        try:
            await client.post(self.url, json=message, headers=headers)
        except Exception:  # noqa: BLE001 - notifications are best effort
            logger.debug("MCP notify %s failed", method, exc_info=True)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _decode_http_body(response: Any) -> dict[str, Any] | None:
    """Accept either a JSON body or a one-message SSE stream, since servers
    are free to answer a POST with either."""
    content_type = response.headers.get("content-type", "")
    text = response.text.strip()
    if not text:
        return None
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        return None
    return json.loads(text)


class MCPConnection:
    """One connected MCP server and the tools it offers."""

    def __init__(self, name: str, transport: MCPTransport, *,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.name = name
        self.transport = transport
        self.timeout = timeout
        self.server_info: dict[str, Any] = {}
        self.tools: list[dict[str, Any]] = []

    async def connect(self) -> MCPConnection:
        start = getattr(self.transport, "start", None)
        if start is not None:
            await start()
        result = await self.transport.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "agentapi", "version": "0.1.0"},
        }, timeout=self.timeout) or {}
        self.server_info = result.get("serverInfo", {})
        await self.transport.notify("notifications/initialized")
        await self.refresh_tools()
        return self

    async def refresh_tools(self) -> list[dict[str, Any]]:
        result = await self.transport.request("tools/list", {},
                                              timeout=self.timeout) or {}
        self.tools = result.get("tools", [])
        return self.tools

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        result = await self.transport.request(
            "tools/call", {"name": tool, "arguments": arguments},
            timeout=self.timeout) or {}
        content = result.get("content", [])
        text = "\n".join(block.get("text", "") for block in content
                         if block.get("type") == "text")
        if result.get("isError"):
            raise MCPError(text or f"{tool} failed")
        # Hand back structured data when the server sent JSON, since the
        # model handles objects better than a stringified blob.
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    async def close(self) -> None:
        await self.transport.close()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "server": self.server_info,
                "tools": [t.get("name") for t in self.tools]}
