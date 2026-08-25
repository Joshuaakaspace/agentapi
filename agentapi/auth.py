"""Authentication and tenant isolation.

Until now ``x-tenant-id`` was trusted exactly as sent, which means any
caller could bill another tenant's budget, read another tenant's run, or
cancel it. Runs are long-lived, addressable objects holding prompts and
results — the most sensitive thing in an agent system — so ownership has to
be enforced, not assumed.

An ``Authenticator`` turns a request into a ``Principal``. The principal's
tenant is authoritative: it overrides any header the client sent. Every run
records its owner at creation, and every run-addressed endpoint checks it.

    @app.authenticator
    async def authenticate(request):
        token = (request.headers.get("authorization") or "").removeprefix("Bearer ")
        record = await lookup(token)
        if record is None:
            return None                       # -> 401
        return Principal(id=record.user, tenant=record.org,
                         scopes={"runs:write"})

With no authenticator registered the app is open — same as FastAPI with no
dependency — but ``require_auth=True`` makes that a startup error instead of
a silent hole.
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass(frozen=True)
class Principal:
    """Who is making this request."""

    id: str
    tenant: Optional[str] = None
    scopes: frozenset[str] = field(default_factory=frozenset)
    claims: dict[str, Any] = field(default_factory=dict)

    def has_scope(self, scope: str) -> bool:
        return not self.scopes or scope in self.scopes

    @property
    def is_anonymous(self) -> bool:
        return self.id == ANONYMOUS.id


ANONYMOUS = Principal(id="anonymous")

Authenticator = Callable[[Any], Awaitable[Optional[Principal]]]


class AuthError(Exception):
    """Raised when a request cannot be authenticated or authorised."""

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


def bearer_tokens(tokens: dict[str, Principal]) -> Authenticator:
    """A ready-made authenticator for a fixed token->principal map.

    Useful for internal services and tests. Comparison is constant-time so
    the map cannot be probed by timing.
    """
    async def authenticate(request: Any) -> Optional[Principal]:
        header = request.headers.get("authorization") or ""
        if not header.lower().startswith("bearer "):
            return None
        presented = header[7:].strip()
        for token, principal in tokens.items():
            if hmac.compare_digest(token, presented):
                return principal
        return None
    return authenticate


def owns(principal: Principal, run: Any) -> bool:
    """May this principal see or steer this run?

    Ownership is by tenant when the run has one, else by principal id. A
    run created anonymously (no authenticator configured) is public.
    """
    owner = getattr(run, "owner", None)
    if owner is None:
        return True
    owner_id, owner_tenant = owner
    if owner_tenant is not None:
        return principal.tenant == owner_tenant
    return principal.id == owner_id
