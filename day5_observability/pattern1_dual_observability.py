import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_observability():
    """Configure LangSmith tracing, Langfuse callbacks, and Google ADK OpenTelemetry."""
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=config.LANGFUSE_OTEL_ENDPOINT,
                headers=config.LANGFUSE_OTEL_HEADERS,
            )
        )
    )
    trace.set_tracer_provider(tracer_provider)

    langfuse_handler = config.get_langfuse_handler()
    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
    return langfuse_handler, tracer_provider


if __name__ == "__main__":
    handler, _provider = configure_observability()
    print("Dual Observability Configured:")
    print(
        f" - LangSmith Project : {config.LANGSMITH_PROJECT} "
        f"(Tracing={config.LANGSMITH_TRACING})"
    )
    handler_name = type(handler).__name__ if handler else "not configured"
    print(f" - Langfuse Handler  : {handler_name}")
    print(f" - ADK OTel Endpoint : {config.LANGFUSE_OTEL_ENDPOINT}")
