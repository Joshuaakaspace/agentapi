# Multi-stage: build wheels once, ship a slim runtime image.
FROM python:3.12-slim AS build
WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip build
COPY pyproject.toml README.md ./
COPY agentapi ./agentapi
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[postgres,redis,otel,anthropic,server]"

FROM python:3.12-slim
# Run as a non-root user: the agent sandbox drops privileges it does not
# have, so the process must not start with root's.
RUN useradd --create-home --uid 10001 agentapi
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
        agentapi uvicorn && rm -rf /wheels
COPY --chown=agentapi:agentapi agentapi ./agentapi
COPY --chown=agentapi:agentapi examples ./examples
USER agentapi

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENTAPI_LOG_JSON=true

EXPOSE 8000

# Liveness only — readiness is /readyz and belongs to the orchestrator, not
# to Docker, because a draining pod is unhealthy for traffic but must not
# be restarted.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2).status==200 else 1)"

# --timeout-graceful-shutdown must exceed AGENTAPI_SHUTDOWN_GRACE_S so the
# app finishes draining before uvicorn pulls the rug.
CMD ["uvicorn", "examples.agent:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "45"]
