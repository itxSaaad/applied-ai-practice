import config
from utils import telemetry
from utils.logging import get_logger

logger = get_logger(__name__)


def configure_observability():
    """Configure LangSmith tracing, Langfuse callbacks, and Google ADK OpenTelemetry."""
    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        raise RuntimeError(
            "Langfuse OpenTelemetry requires LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY; configure .env before instrumenting ADK."
        )
    langfuse_handler = telemetry.init_telemetry(service_name="course-1-adk")
    tracer_provider = telemetry.get_tracer_provider()
    logger.info(
        "Configured ADK OpenTelemetry export",
        extra={
            "day_module": "day5_observability.pattern1_dual_observability",
            "source_file": "day5_observability/pattern1_dual_observability.py",
        },
    )
    return langfuse_handler, tracer_provider


def configure_if_enabled():
    """Instrument ADK when Langfuse credentials exist; otherwise log local-only mode."""
    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        logger.warning("Langfuse ADK export is disabled because credentials are not configured.")
        return None
    _, provider = configure_observability()
    return provider


def flush_observability(tracer_provider) -> None:
    """Flush provider buffers at process shutdown and surface export failures."""
    if tracer_provider is not telemetry.get_tracer_provider():
        raise ValueError("Cannot flush a tracer provider that was not initialized here.")
    telemetry.flush_telemetry()


if __name__ == "__main__":
    handler, provider = configure_observability()
    try:
        print("Dual Observability Configured:")
        print(f" - LangSmith Project : {config.LANGSMITH_PROJECT} " f"(Tracing={config.LANGSMITH_TRACING})")
        handler_name = type(handler).__name__ if handler else "not configured"
        print(f" - Langfuse Handler  : {handler_name}")
        print(f" - ADK OTel Endpoint : {config.LANGFUSE_OTEL_ENDPOINT}")
    finally:
        flush_observability(provider)
