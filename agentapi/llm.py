"""LLM client facade wired into ctx: usage metering, budget enforcement,
deadline propagation, pool admission and hook instrumentation on every call.

Providers plug in behind one tiny interface (``complete``/``stream``).
``AnthropicLLM`` talks to the Claude API over httpx; ``MockLLM`` powers
tests and examples with zero network.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional

from .context import get_ctx
from .pools import Pool


class BaseLLM:
    """Provider interface. Implement ``_complete`` and ``_stream``."""

    def __init__(self, *, pool: Optional[Pool] = None,
                 usd_per_input_mtok: float = 3.0,
                 usd_per_output_mtok: float = 15.0,
                 hooks: Any = None) -> None:
        self.pool = pool
        self.usd_per_input_mtok = usd_per_input_mtok
        self.usd_per_output_mtok = usd_per_output_mtok
        self.hooks = hooks

    # -- public api ---------------------------------------------------------
    async def complete(self, *, model: str, messages: list[dict[str, Any]],
                       **params: Any) -> dict[str, Any]:
        ctx = get_ctx()
        ctx.check()
        if self.hooks:
            await self.hooks.fire("on_llm_call", model, params)
        async with self._admission(ctx):
            response = await self._complete(model=model, messages=messages,
                                            **params)
        self._charge(ctx, response.get("usage", {}))
        if self.hooks:
            await self.hooks.fire("on_llm_result", model, response.get("usage"))
        return response

    async def stream(self, *, model: str, messages: list[dict[str, Any]],
                     **params: Any) -> AsyncIterator[str]:
        ctx = get_ctx()
        ctx.check()
        if self.hooks:
            await self.hooks.fire("on_llm_call", model, params)
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        async with self._admission(ctx):
            async for chunk in self._stream(model=model, messages=messages,
                                            usage=usage, **params):
                ctx.check()
                # Meter as we go so budgets trip mid-stream, not after.
                ctx.charge(output_tokens=1,
                           usd=self.usd_per_output_mtok / 1_000_000)
                yield chunk
        ctx.charge(input_tokens=usage.get("input_tokens", 0),
                   usd=usage.get("input_tokens", 0)
                   * self.usd_per_input_mtok / 1_000_000,
                   llm_calls=1)
        if self.hooks:
            await self.hooks.fire("on_llm_result", model, usage)

    # -- internals ----------------------------------------------------------
    def _admission(self, ctx: Any):
        if self.pool is not None:
            remaining = ctx.deadline_remaining
            return self.pool.acquire(
                deadline_remaining=None if remaining == float("inf") else remaining)
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _null():
            yield None
        return _null()

    def _charge(self, ctx: Any, usage: dict[str, Any]) -> None:
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        ctx.charge(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=(input_tokens * self.usd_per_input_mtok
                 + output_tokens * self.usd_per_output_mtok) / 1_000_000,
            llm_calls=1,
        )

    async def _complete(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _stream(self, **kwargs: Any) -> AsyncIterator[str]:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    """Claude API provider over httpx (streaming via SSE)."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.anthropic.com", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def _complete(self, *, model: str, messages: list[dict[str, Any]],
                        max_tokens: int = 4096, **params: Any) -> dict[str, Any]:
        import httpx
        payload = {"model": model, "messages": messages,
                   "max_tokens": max_tokens, **params}
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(f"{self.base_url}/v1/messages",
                                         headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def _stream(self, *, model: str, messages: list[dict[str, Any]],
                      usage: dict[str, int], max_tokens: int = 4096,
                      **params: Any) -> AsyncIterator[str]:
        import httpx
        payload = {"model": model, "messages": messages,
                   "max_tokens": max_tokens, "stream": True, **params}
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream("POST", f"{self.base_url}/v1/messages",
                                     headers=self._headers(),
                                     json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    kind = data.get("type")
                    if kind == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                    elif kind == "message_start":
                        usage["input_tokens"] = (data.get("message", {})
                                                 .get("usage", {})
                                                 .get("input_tokens", 0))


class MockLLM(BaseLLM):
    """Deterministic provider for tests and examples: streams a canned
    (or echo-derived) response token by token."""

    def __init__(self, script: Optional[list[str]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.script = script or []
        self._cursor = 0

    def _next_text(self, messages: list[dict[str, Any]]) -> str:
        if self._cursor < len(self.script):
            text = self.script[self._cursor]
            self._cursor += 1
            return text
        last = messages[-1]["content"] if messages else ""
        return f"echo: {last}"

    async def _complete(self, *, model: str, messages: list[dict[str, Any]],
                        **params: Any) -> dict[str, Any]:
        text = self._next_text(messages)
        return {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": sum(len(str(m.get('content', '')).split())
                                          for m in messages),
                      "output_tokens": len(text.split())},
        }

    async def _stream(self, *, model: str, messages: list[dict[str, Any]],
                      usage: dict[str, int], **params: Any) -> AsyncIterator[str]:
        text = self._next_text(messages)
        usage["input_tokens"] = sum(len(str(m.get("content", "")).split())
                                    for m in messages)
        for word in text.split(" "):
            usage["output_tokens"] += 1
            yield word + " "
