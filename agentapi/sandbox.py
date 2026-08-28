"""Sandboxed execution for agent tools.

An agent runs commands and writes files chosen by a model, sometimes acting
on text it read from the internet. Treat every such action as untrusted
input, because it is: the containment has to be in the runtime, not in the
prompt.

``Sandbox`` gives each execution:

* a **workspace jail** — every path is resolved (symlinks included) and
  must land inside the workspace root, so ``../../etc/passwd`` and a
  symlink pointing outside are both refused;
* **resource limits** via POSIX rlimits — CPU seconds, address space,
  process count, file size — so a fork bomb or a runaway allocation dies
  instead of taking the host with it;
* a **wall-clock timeout** that kills the whole process group, since a
  child that outlives its parent is the usual way timeouts leak;
* a **scrubbed environment**, so API keys in the server's environment are
  not readable by a model-authored command;
* optional **network isolation** via ``unshare -n`` where available.

This is defence in depth for a cooperative agent, not a security boundary
against a determined attacker sharing your kernel. For genuinely hostile
code, run the whole server in a container or microVM — this class narrows
blast radius, it does not replace isolation.
"""
from __future__ import annotations

import asyncio
import os
import resource
import shutil
import signal
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Environment variables safe to pass through. Everything else is dropped,
# which notably includes every *_API_KEY, *_TOKEN and cloud credential.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TZ", "TERM", "HOME")


class SandboxError(Exception):
    """The request was refused before anything ran."""


class PathEscape(SandboxError):
    """A path resolved outside the workspace."""


@dataclass
class Limits:
    """Per-execution resource ceilings."""

    timeout_s: float = 30.0
    cpu_s: int = 10
    memory_mb: int = 512
    max_processes: int = 64
    max_file_mb: int = 64
    max_output_bytes: int = 256_000

    def as_rlimits(self) -> list[tuple[int, tuple[int, int]]]:
        limits = [
            (resource.RLIMIT_CPU, (self.cpu_s, self.cpu_s + 1)),
            (resource.RLIMIT_FSIZE,
             (self.max_file_mb << 20, self.max_file_mb << 20)),
            (resource.RLIMIT_CORE, (0, 0)),
        ]
        if hasattr(resource, "RLIMIT_NPROC"):
            limits.append((resource.RLIMIT_NPROC,
                           (self.max_processes, self.max_processes)))
        if hasattr(resource, "RLIMIT_AS") and self.memory_mb:
            size = self.memory_mb << 20
            limits.append((resource.RLIMIT_AS, (size, size)))
        return limits


@dataclass
class Result:
    """Outcome of one sandboxed execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    truncated: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def as_dict(self) -> dict:
        return {
            "stdout": self.stdout, "stderr": self.stderr,
            "exit_code": self.exit_code, "timed_out": self.timed_out,
            "truncated": self.truncated, "duration_s": round(self.duration_s, 3),
        }


class Sandbox:
    """Runs commands and mediates file access inside one workspace."""

    def __init__(self, workspace: str | Path, *,
                 limits: Limits | None = None,
                 network: bool = False,
                 env: dict[str, str] | None = None,
                 create: bool = True) -> None:
        self.root = Path(workspace).resolve()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise SandboxError(f"workspace {self.root} is not a directory")
        self.limits = limits or Limits()
        self.network = network
        self.extra_env = env or {}

    # -- path jail ----------------------------------------------------------
    def resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve a caller-supplied path inside the workspace.

        Resolution follows symlinks *before* the containment check, so a
        symlink planted inside the workspace cannot be used to read or write
        outside it.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PathEscape(
                f"path {path!r} resolves outside the workspace") from None
        if must_exist and not resolved.exists():
            raise SandboxError(f"no such file: {path}")
        return resolved

    def relative(self, path: Path) -> str:
        """Workspace-relative form, for messages the model will read."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    # -- environment --------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = {name: os.environ[name] for name in _ENV_ALLOWLIST
               if name in os.environ}
        env["HOME"] = str(self.root)
        env["PWD"] = str(self.root)
        env.update(self.extra_env)
        return env

    def _wrap(self, argv: Sequence[str]) -> list[str]:
        if self.network:
            return list(argv)
        unshare = shutil.which("unshare")
        if unshare is None:
            return list(argv)          # best effort; documented as such
        # -r maps the current user to root inside the namespace so an
        # unprivileged process may create the (empty) network namespace.
        return [unshare, "-r", "-n", "--", *argv]

    # -- execution ----------------------------------------------------------
    async def run(self, argv: Sequence[str] | str, *,
                  cwd: str | Path | None = None,
                  stdin: str = "",
                  limits: Limits | None = None) -> Result:
        """Execute a command under the sandbox's limits.

        A string is run through ``bash -lc`` (agents write shell, not argv
        arrays); a sequence is executed directly with no shell involved.
        """
        active = limits or self.limits
        shell = isinstance(argv, str)
        command = ["bash", "-lc", argv] if shell else list(argv)
        workdir = self.resolve(cwd) if cwd else self.root

        def preexec() -> None:                       # pragma: no cover
            os.setsid()                              # own process group
            for which, values in active.as_rlimits():
                try:
                    resource.setrlimit(which, values)
                except (ValueError, OSError):
                    pass

        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            process = await asyncio.create_subprocess_exec(
                *self._wrap(command),
                cwd=str(workdir),
                env=self._env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec,
                start_new_session=False,
            )
        except FileNotFoundError as exc:
            raise SandboxError(f"command not found: {command[0]}") from exc

        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin.encode()), timeout=active.timeout_s)
        except TimeoutError:
            timed_out = True
            _kill_group(process)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=5)
            except (TimeoutError, ProcessLookupError):
                stdout, stderr = b"", b""

        out, truncated_out = _cap(stdout, active.max_output_bytes)
        err, truncated_err = _cap(stderr, active.max_output_bytes)
        return Result(
            stdout=out, stderr=err,
            exit_code=-1 if timed_out else (process.returncode or 0),
            timed_out=timed_out,
            truncated=truncated_out or truncated_err,
            duration_s=loop.time() - started,
        )

    # -- file helpers -------------------------------------------------------
    def read(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        target = self.resolve(path, must_exist=True)
        if target.is_dir():
            raise SandboxError(f"{path} is a directory")
        data = target.read_bytes()[:max_bytes + 1]
        if len(data) > max_bytes:
            return data[:max_bytes].decode("utf-8", "replace") + "\n...[truncated]"
        return data.decode("utf-8", "replace")

    def write(self, path: str, content: str) -> Path:
        target = self.resolve(path)
        if len(content.encode()) > (self.limits.max_file_mb << 20):
            raise SandboxError("content exceeds the sandbox file size limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())


def _kill_group(process: asyncio.subprocess.Process) -> None:
    """Kill the process *group*: a timed-out shell usually has children,
    and killing only the parent leaves them running."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _cap(raw: bytes, limit: int) -> tuple[str, bool]:
    text = raw.decode("utf-8", "replace")
    if len(raw) <= limit:
        return text, False
    return text[:limit] + "\n...[output truncated]", True
