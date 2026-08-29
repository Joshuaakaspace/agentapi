"""Configuration from the environment (12-factor).

Every knob that differs between a laptop and production is readable from
the environment, so an image can be promoted between environments without
a rebuild.

    app = AgentAPI.from_env(llm=AnthropicLLM())

Precedence: explicit keyword arguments beat environment variables, which
beat defaults — so a value set in code is never silently overridden by a
stray variable in a shell.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Config:
    """Resolved runtime configuration."""

    durable: str | None = None
    fanout: str | None = None
    determinism: str = "raise"
    require_auth: bool = False
    retention_s: float = 3600.0
    rate_limit_per_minute: float | None = None
    rate_limit_burst: float | None = None
    redact: bool = False
    log_level: str = "INFO"
    log_json: bool = True
    metrics: bool = True
    shutdown_grace_s: float = 30.0
    max_body_bytes: int = 4 << 20
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str = "AGENTAPI_") -> Config:
        def key(name: str) -> str:
            return f"{prefix}{name}"
        return cls(
            durable=os.environ.get(key("DURABLE")) or None,
            fanout=os.environ.get(key("FANOUT")) or None,
            determinism=os.environ.get(key("DETERMINISM"), "raise"),
            require_auth=_bool(key("REQUIRE_AUTH"), False),
            retention_s=_float(key("RETENTION_S"), 3600.0) or 3600.0,
            rate_limit_per_minute=_float(key("RATE_LIMIT_PER_MINUTE"), None),
            rate_limit_burst=_float(key("RATE_LIMIT_BURST"), None),
            redact=_bool(key("REDACT"), False),
            log_level=os.environ.get(key("LOG_LEVEL"), "INFO"),
            log_json=_bool(key("LOG_JSON"), True),
            metrics=_bool(key("METRICS"), True),
            shutdown_grace_s=_float(key("SHUTDOWN_GRACE_S"), 30.0) or 30.0,
            max_body_bytes=_int(key("MAX_BODY_BYTES"), 4 << 20),
        )

    def describe(self) -> dict[str, Any]:
        """Safe to log: contains no credentials, and DSNs are host-only."""
        return {
            "durable": _scrub_dsn(self.durable),
            "fanout": _scrub_dsn(self.fanout),
            "determinism": self.determinism,
            "require_auth": self.require_auth,
            "redact": self.redact,
            "metrics": self.metrics,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "shutdown_grace_s": self.shutdown_grace_s,
        }


def _scrub_dsn(value: str | None) -> str | None:
    """Never log a DSN verbatim: they routinely carry passwords."""
    if not value or "://" not in value:
        return value
    scheme, _, rest = value.partition("://")
    host = rest.split("@")[-1]
    return f"{scheme}://{host}"
