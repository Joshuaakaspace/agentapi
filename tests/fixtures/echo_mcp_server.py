#!/usr/bin/env python3
"""A minimal stdio MCP server, used to test the client against a real
subprocess rather than a mock."""
import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo a message back.",
     "inputSchema": {"type": "object",
                     "properties": {"message": {"type": "string"}},
                     "required": ["message"]}},
    {"name": "add", "description": "Add two numbers.",
     "inputSchema": {"type": "object",
                     "properties": {"a": {"type": "number"},
                                    "b": {"type": "number"}},
                     "required": ["a", "b"]}},
    {"name": "explode", "description": "Always fails.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "leak_env", "description": "Report a host env var.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def handle(message):
    method = message.get("method", "")
    msg_id = message.get("id")
    params = message.get("params") or {}

    if method.startswith("notifications/"):
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18",
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "echo-server", "version": "1.0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            result = {"content": [{"type": "text",
                                   "text": json.dumps({"echoed": args["message"]})}],
                      "isError": False}
        elif name == "add":
            result = {"content": [{"type": "text",
                                   "text": json.dumps({"sum": args["a"] + args["b"]})}],
                      "isError": False}
        elif name == "leak_env":
            import os
            result = {"content": [{"type": "text", "text": json.dumps(
                {"secret": os.environ.get("HOST_ONLY_SECRET", "<absent>")})}],
                "isError": False}
        elif name == "explode":
            result = {"content": [{"type": "text", "text": "tool blew up"}],
                      "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32601, "message": f"no tool {name}"}}
    else:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
