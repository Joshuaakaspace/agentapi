"""AgentAPI: the application object and its ASGI surface.

FastAPI-shaped ergonomics, run-shaped semantics:

    app = AgentAPI(llm=AnthropicLLM())

    @app.run("/chat", on_disconnect="detach", deadline="120s", budget_usd=0.50)
    async def chat(prompt: str):
        async for tok in ctx.llm.stream(model="claude-opus-5",
                                        messages=[{"role": "user",
                                                   "content": prompt}]):
            yield Token(text=tok)

Surfaces mounted automatically:
    POST /{route}                 create a run (Idempotency-Key honoured)
    GET  /runs/{id}               run status + usage
    GET  /runs/{id}/events        resumable SSE (?from= cursor, Last-Event-ID)
    POST /runs/{id}/signals/{s}   deliver a human-in-the-loop signal
    POST /runs/{id}/cancel        cancel
    GET  /ops                     op catalog
    GET  /llm/tools               LLM tool definitions (?style=anthropic|openai)
    POST /mcp                     MCP server (JSON-RPC / Streamable HTTP POST)
    GET  /skills                  mounted skills
    GET  /pools                   admission-control stats
    GET  /healthz
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, Optional, get_type_hints

from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .context import RunContext, parse_duration
from .determinism import MODES as DETERMINISM_MODES
from .events import parse_event
from .hooks import Hooks
from .journal import SQLiteBackend
from .llm import BaseLLM
from .mcp import MCPServer
from .ops import OpRegistry, _schema_from_signature
from .pools import Pool, PoolSaturated
from .run import RunManager, RunStatus
from .skills import Skill, SkillSet


class _RunRoute:
    def __init__(self, path: str, handler: Callable[..., Any], *,
                 on_disconnect: str, deadline: Optional[float],
                 budget_usd: Optional[float], budget_tokens: Optional[int],
                 pools: list[Pool], durability: str) -> None:
        self.path = path
        self.handler = handler
        self.on_disconnect = on_disconnect
        self.deadline = deadline
        self.budget_usd = budget_usd
        self.budget_tokens = budget_tokens
        self.pools = pools
        self.durability = durability
        self.args_model, self.args_schema = _schema_from_signature(handler)


class AgentAPI:
    def __init__(self, *, title: str = "agentapi", version: str = "0.1.0",
                 llm: Optional[BaseLLM] = None,
                 retention_s: float = 3600.0,
                 durable: Optional[str] = None,
                 determinism: str = "raise") -> None:
        if determinism not in DETERMINISM_MODES:
            raise ValueError(
                f"determinism must be one of {DETERMINISM_MODES}")
        self.determinism = determinism
        self.title = title
        self.version = version
        self.llm = llm
        self.backend = SQLiteBackend(durable) if durable else None
        self.runs = RunManager(retention_s=retention_s, backend=self.backend)
        self.ops = OpRegistry()
        self.hooks = Hooks()
        self.skills = SkillSet()
        self.pools: dict[str, Pool] = {}
        self._run_routes: dict[str, _RunRoute] = {}
        self._asgi: Optional[Starlette] = None
        self.mcp = MCPServer(title, self._McpOps(self), version)
        if llm is not None:
            llm.hooks = self.hooks

    class _McpOps:
        """Ops view for MCP that includes skill ops."""
        def __init__(self, app: "AgentAPI") -> None:
            self.app = app

        def mcp_tools(self) -> list[dict[str, Any]]:
            tools = self.app.ops.mcp_tools()
            for skill in self.app.skills.all():
                tools.extend(skill.ops.mcp_tools())
            return tools

        def get(self, name: str):
            op = self.app.ops.get(name)
            if op is not None:
                return op
            for skill in self.app.skills.all():
                op = skill.ops.get(name)
                if op is not None:
                    return op
            return None

    # -- decorators ----------------------------------------------------------
    def run(self, path: str, *, on_disconnect: str = "detach",
            deadline: Optional[str | float] = None,
            budget_usd: Optional[float] = None,
            budget_tokens: Optional[int] = None,
            pools: Optional[list[Pool]] = None,
            durability: str = "resumable") -> Callable[[Callable], Callable]:
        """Register a run handler. The handler is an async generator yielding
        Events (streaming) or a coroutine returning a result (request/response
        over the same run machinery)."""
        if on_disconnect not in ("detach", "cancel", "drain"):
            raise ValueError("on_disconnect must be detach|cancel|drain")
        if durability not in ("ephemeral", "resumable", "durable"):
            raise ValueError("durability must be ephemeral|resumable|durable")
        if durability == "durable" and self.backend is None:
            raise ValueError(
                'durability="durable" requires AgentAPI(durable="path.db")')

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._run_routes[path] = _RunRoute(
                path, fn,
                on_disconnect=on_disconnect,
                deadline=None if deadline is None else parse_duration(deadline),
                budget_usd=budget_usd, budget_tokens=budget_tokens,
                pools=pools or [], durability=durability)
            return fn
        return decorate

    def op(self, fn: Optional[Callable[..., Any]] = None, *,
           name: Optional[str] = None, description: Optional[str] = None,
           http: Optional[str] = None, llm_tool: bool = True,
           mcp: bool = True) -> Any:
        """Register a multi-surface op: HTTP + LLM tool + MCP from one
        signature."""
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            self.ops.register(func, name=name, description=description,
                              http=http, llm_tool=llm_tool, mcp=mcp)
            return func
        return decorate if fn is None else decorate(fn)

    def hook(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            self.hooks.register(event, func)
            return func
        return decorate

    def pool(self, name: str, **kwargs: Any) -> Pool:
        pool = Pool(name, **kwargs)
        self.pools[name] = pool
        return pool

    def include_skill(self, skill: Skill) -> Skill:
        self.skills.add(skill)
        self.hooks.merge(skill.hooks)
        return skill

    # -- llm tool exports ----------------------------------------------------
    def llm_tools(self, style: str = "anthropic") -> list[dict[str, Any]]:
        tools = self.ops.llm_tools(style)
        for skill in self.skills.all():
            tools.extend(skill.ops.llm_tools(style))
        return tools

    async def call_op(self, name: str, arguments: dict[str, Any],
                      call_id: Optional[str] = None) -> Any:
        """Dispatch a model-issued tool call by name (agent loop helper)."""
        op = self.mcp.ops.get(name)
        if op is None:
            raise ValueError(f"unknown op: {name}")
        return await op.call(arguments, hooks=self.hooks, call_id=call_id)

    async def agent(self, *, model: str, messages: list[dict[str, Any]],
                    system: Optional[str] = None,
                    tools: Optional[list[str]] = None,
                    max_turns: int = 10, **params: Any) -> dict[str, Any]:
        """Run an LLM tool loop against this app's ops inside the current
        run. See agentapi.agent.agent_loop."""
        from .agent import agent_loop
        return await agent_loop(self, model=model, messages=messages,
                                system=system, tools=tools,
                                max_turns=max_turns, **params)

    # -- run creation (shared by all transports) -----------------------------
    def _start_run(self, route: _RunRoute, kwargs: dict[str, Any], *,
                   tenant: Optional[str], idempotency_key: Optional[str]):
        if idempotency_key:
            existing = self.runs.by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing, False
        context = RunContext(
            "pending", tenant=tenant,
            deadline_s=route.deadline,
            max_usd=route.budget_usd, max_tokens=route.budget_tokens)
        context._llm = self.llm
        run = self.runs.start(route.path, route.handler, kwargs,
                              ctx=context, idempotency_key=idempotency_key,
                              hooks=self.hooks,
                              durable=route.durability == "durable",
                              determinism=self.determinism)
        return run, True

    # -- recovery (tier-2 durability) ---------------------------------------
    def recover(self) -> list[str]:
        """Restart every unfinished durable run from the journal. Call once
        at process start (e.g. in a lifespan handler). Completed steps
        return journaled results, past signals re-deliver the same payloads,
        and re-emitted history is deduplicated — so side effects run once
        and the event log continues where it left off."""
        if self.backend is None:
            return []
        recovered: list[str] = []
        for row in self.backend.unfinished_runs():
            route = self._run_routes.get(row["route"])
            if route is None or row["id"] in self.runs.runs:
                continue
            context = RunContext(
                row["id"], tenant=row["tenant"],
                deadline_s=route.deadline,
                max_usd=route.budget_usd, max_tokens=route.budget_tokens,
                metadata=row["metadata"])
            context._llm = self.llm
            context._step_journal = self.backend.steps(row["id"])
            for name, payload in self.backend.signals(row["id"]):
                context.deliver_signal(name, payload)
            replay_events = [parse_event(e)
                             for e in self.backend.events(row["id"])]
            self.runs.start(route.path, route.handler, row["kwargs"],
                            ctx=context, hooks=self.hooks, durable=True,
                            run_id=row["id"], replay_events=replay_events,
                            determinism=self.determinism)
            recovered.append(row["id"])
        return recovered

    # -- ASGI ---------------------------------------------------------------
    @property
    def asgi(self) -> Starlette:
        if self._asgi is None:
            self._asgi = self._build_asgi()
        return self._asgi

    async def __call__(self, scope: dict[str, Any], receive: Any,
                       send: Any) -> None:
        await self.asgi(scope, receive, send)

    def _build_asgi(self) -> Starlette:
        routes: list[Route] = []

        for route in self._run_routes.values():
            routes.append(Route(route.path, self._make_run_endpoint(route),
                                methods=["POST"]))

        for op in self.ops.all():
            if op.http:
                method, _, path = op.http.partition(" ")
                routes.append(Route(path or f"/ops/{op.name}",
                                    self._make_op_endpoint(op),
                                    methods=[method or "POST"]))

        routes += [
            Route("/runs/{run_id}", self._get_run, methods=["GET"]),
            Route("/runs/{run_id}/events", self._get_events, methods=["GET"]),
            Route("/runs/{run_id}/signals/{signal}", self._post_signal,
                  methods=["POST"]),
            Route("/runs/{run_id}/cancel", self._post_cancel, methods=["POST"]),
            Route("/ops", self._get_ops, methods=["GET"]),
            Route("/llm/tools", self._get_llm_tools, methods=["GET"]),
            Route("/mcp", self._post_mcp, methods=["POST"]),
            Route("/skills", self._get_skills, methods=["GET"]),
            Route("/pools", self._get_pools, methods=["GET"]),
            Route("/healthz", lambda r: JSONResponse({"ok": True}),
                  methods=["GET"]),
        ]
        return Starlette(routes=routes)

    def _make_run_endpoint(self, route: _RunRoute) -> Callable[..., Any]:
        async def endpoint(request: Request) -> Response:
            try:
                body = await request.json() if await request.body() else {}
            except json.JSONDecodeError:
                return JSONResponse({"error": "invalid JSON body"}, status_code=400)
            try:
                parsed = route.args_model.model_validate(body)
            except Exception as exc:  # noqa: BLE001 - validation error
                return JSONResponse({"error": str(exc)}, status_code=422)
            kwargs = {k: getattr(parsed, k) for k in type(parsed).model_fields}

            # Deadline-aware admission before any work happens.
            for pool in route.pools:
                estimated = pool._estimated_wait()  # noqa: SLF001
                if route.deadline is not None and estimated > route.deadline:
                    return JSONResponse(
                        {"error": f"pool {pool.name} saturated"},
                        status_code=429,
                        headers={"Retry-After": str(int(estimated) + 1)})

            run, created = self._start_run(
                route, kwargs,
                tenant=request.headers.get("x-tenant-id"),
                idempotency_key=request.headers.get("idempotency-key"))

            if request.query_params.get("stream") == "false":
                # Block until terminal; request/response over run machinery.
                if run.task is not None:
                    await asyncio.shield(run.task)
                return JSONResponse(run.describe(),
                                    status_code=200 if created else 200)
            return self._sse(run, from_seq=0,
                             on_disconnect=route.on_disconnect,
                             status_code=201 if created else 200)
        return endpoint

    def _make_op_endpoint(self, op: Any) -> Callable[..., Any]:
        async def endpoint(request: Request) -> Response:
            try:
                body = await request.json() if await request.body() else {}
            except json.JSONDecodeError:
                return JSONResponse({"error": "invalid JSON body"}, status_code=400)
            try:
                result = await op.call(body, hooks=self.hooks)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                                    status_code=422)
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            return JSONResponse({"result": result})
        return endpoint

    # -- transport: resumable SSE -------------------------------------------
    def _sse(self, run: Any, *, from_seq: int, on_disconnect: str,
             status_code: int = 200) -> StreamingResponse:
        async def body():
            try:
                async for event in run.events(from_seq):
                    payload = event.model_dump(by_alias=True)
                    yield (f"id: {event.seq}\n"
                           f"event: {event.type}\n"
                           f"data: {json.dumps(payload, default=str)}\n\n")
            finally:
                # Client went away (or stream ended). The run's fate is the
                # route's policy, not the socket's.
                if not run.log.closed and run.attached == 0:
                    if on_disconnect == "cancel":
                        self.runs.cancel(run.id)
                    # "detach": run continues; client resumes via
                    # GET /runs/{id}/events?from=<Last-Event-ID + 1>.
                    # "drain" is honoured by handlers via ctx.draining (future).

        return StreamingResponse(
            body(), status_code=status_code, media_type="text/event-stream",
            headers={"x-run-id": run.id, "cache-control": "no-cache",
                     "x-accel-buffering": "no"})

    # -- endpoints -----------------------------------------------------------
    async def _get_run(self, request: Request) -> Response:
        run_id = request.path_params["run_id"]
        run = self.runs.get(run_id)
        if run is None:
            if self.backend is not None:
                row = self.backend.get_run(run_id)
                if row is not None:
                    row["archived"] = True
                    return JSONResponse(row)
            return JSONResponse({"error": "run not found"}, status_code=404)
        return JSONResponse(run.describe())

    async def _get_events(self, request: Request) -> Response:
        run = self.runs.get(request.path_params["run_id"])
        if run is None:
            if self.backend is not None:
                stored = self.backend.events(request.path_params["run_id"])
                if stored:
                    from_seq = int(request.query_params.get("from", 0))
                    return JSONResponse({"events": stored[from_seq:],
                                         "next": len(stored), "closed": True,
                                         "archived": True})
            return JSONResponse({"error": "run not found"}, status_code=404)
        last_event_id = request.headers.get("last-event-id")
        from_seq = int(request.query_params.get(
            "from", int(last_event_id) + 1 if last_event_id else 0))
        if request.query_params.get("stream") == "false":
            events = [e.model_dump(by_alias=True) for e in run.log.read(from_seq)]
            return JSONResponse({"events": events, "next": run.log.next_seq,
                                 "closed": run.log.closed})
        route = self._run_routes.get(run.route)
        policy = route.on_disconnect if route else "detach"
        return self._sse(run, from_seq=from_seq, on_disconnect=policy)

    async def _post_signal(self, request: Request) -> Response:
        run_id = request.path_params["run_id"]
        signal = request.path_params["signal"]
        try:
            payload = await request.json() if await request.body() else None
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not self.runs.signal(run_id, signal, payload):
            return JSONResponse({"error": "run not found"}, status_code=404)
        return JSONResponse({"ok": True, "run_id": run_id, "signal": signal})

    async def _post_cancel(self, request: Request) -> Response:
        ok = self.runs.cancel(request.path_params["run_id"])
        if not ok:
            return JSONResponse({"error": "run not found or not cancellable"},
                                status_code=409)
        return JSONResponse({"ok": True})

    async def _get_ops(self, request: Request) -> Response:
        catalog = [{"name": op.name, "description": op.description,
                    "http": op.http, "llm_tool": op.llm_tool, "mcp": op.mcp,
                    "input_schema": op.args_schema}
                   for op in self.ops.all()]
        for skill in self.skills.all():
            catalog += [{"name": op.name, "description": op.description,
                         "skill": skill.name, "input_schema": op.args_schema}
                        for op in skill.ops.all()]
        return JSONResponse({"ops": catalog})

    async def _get_llm_tools(self, request: Request) -> Response:
        style = request.query_params.get("style", "anthropic")
        return JSONResponse({"tools": self.llm_tools(style)})

    async def _post_mcp(self, request: Request) -> Response:
        try:
            message = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"jsonrpc": "2.0", "id": None,
                                 "error": {"code": -32700,
                                           "message": "parse error"}},
                                status_code=400)
        if isinstance(message, list):  # batch
            responses = [r for r in [await self.mcp.handle(m) for m in message]
                         if r is not None]
            return JSONResponse(responses)
        response = await self.mcp.handle(message)
        if response is None:
            return Response(status_code=202)
        return JSONResponse(response)

    async def _get_skills(self, request: Request) -> Response:
        return JSONResponse({"skills": [s.describe()
                                        for s in self.skills.all()]})

    async def _get_pools(self, request: Request) -> Response:
        return JSONResponse({"pools": [p.stats() for p in self.pools.values()]})
