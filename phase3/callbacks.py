"""
Phase 3 — Observability callbacks.

Two tracing backends, both optional, both toggled via .env:

  LangSmith  — LangChain's hosted platform. Zero setup, great for development.
               Uses the EU endpoint to keep query data in Europe.
               Downside: data leaves your infrastructure (goes to LangChain's servers).

  LangFuse   — Open-source alternative. Can be self-hosted on GCP Cloud Run,
               keeping all trace data inside your EU perimeter — directly relevant
               to GDPR compliance. In Phase 6 we will deploy it on Cloud Run.
               For now, uses LangFuse Cloud (also has an EU region).

Why two backends?
  The job description explicitly names both. Being able to compare them —
  and explain when you'd choose self-hosted LangFuse vs. managed LangSmith —
  is exactly the kind of depth a senior architect is expected to have.

  Short answer for interviews:
    - LangSmith: fastest to get running, best LangGraph integration
    - LangFuse: self-hostable → GDPR-safe for production systems with real user data
"""

from langchain_core.tracers.langchain import LangChainTracer

from config import settings


def get_callbacks() -> list:
    """
    Returns active callback handlers based on .env configuration.

    Both tracers implement the LangChain callback interface, so they can be
    passed identically to graph.invoke() or graph.stream(). You can enable
    both simultaneously — traces will be sent to both platforms.
    """
    callbacks = []

    # --- LangSmith ---
    # LangChainTracer reads LANGCHAIN_ENDPOINT from the environment — not from our
    # settings object. We set it explicitly here so the EU endpoint is always used.
    if settings.langsmith_tracing and settings.langsmith_api_key:
        import os
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        callbacks.append(
            LangChainTracer(project_name=settings.langsmith_project)
        )

    # --- LangFuse ---
    # LangFuse provides a LangChain callback handler via the langfuse package.
    # It requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in the environment.
    # The host defaults to LangFuse Cloud; set LANGFUSE_HOST to your self-hosted
    # Cloud Run URL (Phase 6) to keep all data in the EU.
    if settings.langfuse_tracing and settings.langfuse_secret_key:
        try:
            from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
            callbacks.append(
                LangfuseCallbackHandler(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            )
        except ImportError:
            # langfuse not installed — skip silently, don't crash the agent
            pass

    return callbacks


def active_backends() -> list[str]:
    """Returns the names of currently enabled tracing backends."""
    backends = []
    if settings.langsmith_tracing and settings.langsmith_api_key:
        backends.append("LangSmith")
    if settings.langfuse_tracing and settings.langfuse_secret_key:
        backends.append("LangFuse")
    return backends
