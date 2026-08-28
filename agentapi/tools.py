"""The standard agent toolset, sandboxed and policy-gated.

Every agent harness re-implements the same handful of tools — run a
command, read a file, edit a file, search — and every one re-implements the
same mistakes: unbounded output that blows the context window, paths that
escape the workspace, edits that silently match in three places.

``register_tools(app, sandbox, policy)`` installs them as ordinary
``@app.op``s, so each one is simultaneously an HTTP endpoint, an LLM tool
definition and an MCP tool, with sandbox containment and permission checks
applied uniformly.
"""
from __future__ import annotations

import fnmatch
import re
from typing import Any

from .permissions import Policy, enforce
from .sandbox import Sandbox, SandboxError

MAX_MATCHES = 200
MAX_LISTING = 500


def register_tools(app: Any, sandbox: Sandbox, policy: Policy, *,
                   include: list[str] | None = None,
                   approval_timeout: str = "1h") -> list[str]:
    """Register the toolset on ``app``. Returns the tool names registered."""

    async def gate(tool: str, arguments: dict[str, Any]) -> None:
        await enforce(policy, tool, arguments, timeout=approval_timeout)

    registered: list[str] = []

    def wanted(name: str) -> bool:
        return include is None or name in include

    # -- shell ---------------------------------------------------------------
    if wanted("bash"):
        @app.op(name="bash")
        async def bash(command: str, timeout_s: float = 30.0) -> dict:
            """Run a shell command inside the agent's workspace.

            The workspace is the only writable location and the environment
            carries no credentials. Output is truncated if very large.
            """
            await gate("bash", {"command": command})
            from .sandbox import Limits
            limits = Limits(**{**vars(sandbox.limits), "timeout_s": timeout_s})
            result = await sandbox.run(command, limits=limits)
            return result.as_dict()
        registered.append("bash")

    # -- reading -------------------------------------------------------------
    if wanted("read_file"):
        @app.op(name="read_file")
        async def read_file(path: str, offset: int = 0,
                            limit: int = 400) -> dict:
            """Read a text file from the workspace, with line numbers.

            Reads a window of ``limit`` lines from ``offset`` so a large
            file cannot flood the context.
            """
            await gate("read_file", {"path": path})
            text = sandbox.read(path)
            lines = text.splitlines()
            window = lines[offset:offset + limit]
            numbered = "\n".join(f"{offset + i + 1:>6}\t{line}"
                                 for i, line in enumerate(window))
            return {"path": path, "content": numbered,
                    "lines": len(lines),
                    "truncated": offset + limit < len(lines)}
        registered.append("read_file")

    if wanted("list_dir"):
        @app.op(name="list_dir")
        async def list_dir(path: str = ".") -> dict:
            """List a directory in the workspace."""
            await gate("list_dir", {"path": path})
            target = sandbox.resolve(path, must_exist=True)
            if not target.is_dir():
                raise SandboxError(f"{path} is not a directory")
            entries = []
            for child in sorted(target.iterdir())[:MAX_LISTING]:
                entries.append({
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "bytes": child.stat().st_size if child.is_file() else None,
                })
            return {"path": path, "entries": entries}
        registered.append("list_dir")

    if wanted("glob"):
        @app.op(name="glob")
        async def glob(pattern: str, path: str = ".") -> dict:
            """Find files matching a glob pattern (e.g. '**/*.py')."""
            await gate("glob", {"pattern": pattern, "path": path})
            root = sandbox.resolve(path, must_exist=True)
            matches = [sandbox.relative(p) for p in sorted(root.glob(pattern))
                       if p.is_file()][:MAX_MATCHES]
            return {"pattern": pattern, "matches": matches,
                    "count": len(matches)}
        registered.append("glob")

    if wanted("grep"):
        @app.op(name="grep")
        async def grep(pattern: str, path: str = ".",
                       glob: str = "*", max_results: int = 50) -> dict:
            """Search file contents with a regular expression."""
            await gate("grep", {"pattern": pattern, "path": path})
            try:
                expression = re.compile(pattern)
            except re.error as exc:
                raise SandboxError(f"invalid regex: {exc}") from exc
            root = sandbox.resolve(path, must_exist=True)
            candidates = ([root] if root.is_file()
                          else [p for p in root.rglob("*") if p.is_file()])
            results = []
            for candidate in candidates:
                if not fnmatch.fnmatch(candidate.name, glob):
                    continue
                try:
                    content = candidate.read_text(errors="replace")
                except OSError:
                    continue
                for number, line in enumerate(content.splitlines(), 1):
                    if expression.search(line):
                        results.append({"file": sandbox.relative(candidate),
                                        "line": number,
                                        "text": line.strip()[:300]})
                        if len(results) >= max_results:
                            return {"pattern": pattern, "results": results,
                                    "truncated": True}
            return {"pattern": pattern, "results": results, "truncated": False}
        registered.append("grep")

    # -- writing -------------------------------------------------------------
    if wanted("write_file"):
        @app.op(name="write_file")
        async def write_file(path: str, content: str) -> dict:
            """Create or overwrite a file in the workspace."""
            await gate("write_file", {"path": path})
            target = sandbox.write(path, content)
            return {"path": sandbox.relative(target),
                    "bytes": len(content.encode())}
        registered.append("write_file")

    if wanted("edit_file"):
        @app.op(name="edit_file")
        async def edit_file(path: str, old: str, new: str,
                            replace_all: bool = False) -> dict:
            """Replace exact text in a file.

            Refuses when ``old`` appears more than once unless
            ``replace_all`` is set: an ambiguous edit that silently picks the
            first match is how agents corrupt files.
            """
            await gate("edit_file", {"path": path})
            target = sandbox.resolve(path, must_exist=True)
            content = target.read_text()
            occurrences = content.count(old)
            if occurrences == 0:
                raise SandboxError(
                    f"text not found in {path}; read the file first and "
                    f"match it exactly")
            if occurrences > 1 and not replace_all:
                raise SandboxError(
                    f"{occurrences} matches in {path}; include more context "
                    f"to make it unique, or pass replace_all=true")
            updated = (content.replace(old, new) if replace_all
                       else content.replace(old, new, 1))
            sandbox.write(sandbox.relative(target), updated)
            return {"path": sandbox.relative(target),
                    "replacements": occurrences if replace_all else 1}
        registered.append("edit_file")

    return registered
