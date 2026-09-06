# Security

## Reporting a vulnerability

Please email the maintainer rather than opening a public issue, and include
steps to reproduce. You'll get an acknowledgement within a few days.

## Scope worth knowing before you report

**The sandbox is blast-radius reduction, not a security boundary.** It
resolves symlinks before the containment check, applies rlimits, kills
process groups on timeout, scrubs the environment and caps output. It does
**not** use seccomp or a user namespace beyond an optional `unshare -n`.
Escaping it from genuinely hostile code sharing the kernel is expected to be
possible and is documented as such in `PRODUCTION.md`. Reports that
demonstrate a *stronger* claim than that being false — e.g. a path-jail
bypass or a credential leak into a sandboxed command — are exactly what we
want to hear about.

**Without an authenticator the app is open by design**, the same way a
FastAPI app with no dependencies is. `require_auth=True` turns that into a
startup error.

**Third-party MCP servers are trusted with what you pass them** via `env=`
and whatever they can reach on the network. Their tool descriptions are read
by the model.
