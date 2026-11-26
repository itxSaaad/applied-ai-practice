"""Tracing, metric, and provider telemetry helpers."""

import inspect
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import FrameType
from typing import ParamSpec, TypedDict, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_langfuse_client = None
_langfuse_handler = None
_langsmith_client = None
_tracer_provider = None
_shutdown = False


class TraceProviderStatus(TypedDict, total=False):
    enabled: bool
    missing_credentials: list[str]
    configured: bool


def _config():
    import config

    return config


def _logger():
    from utils.logging import get_logger

    return get_logger(__name__)


def get_langfuse_client():
    """Return the singleton Langfuse client when both credentials are configured."""
    global _langfuse_client
    config = _config()
    if _langfuse_client is None and config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
        import langfuse

        _langfuse_client = langfuse.Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_BASE_URL,
        )
    return _langfuse_client


def get_langfuse_handler():
    """Return the LangChain callback handler, or None without Langfuse credentials."""
    global _langfuse_handler
    config = _config()
    if _langfuse_handler is None:
        client = get_langfuse_client()
        if client:
            try:
                from langfuse.langchain import CallbackHandler
            except ImportError:
                from langfuse.callback import CallbackHandler

            _langfuse_handler = CallbackHandler(public_key=config.LANGFUSE_PUBLIC_KEY)
    return _langfuse_handler


def get_langsmith_client():
    """Return the singleton LangSmith client when tracing is configured."""
    global _langsmith_client
    config = _config()
    if _langsmith_client is None and config.LANGSMITH_TRACING and config.LANGSMITH_API_KEY:
        from langsmith import Client

        _langsmith_client = Client()
    return _langsmith_client


def get_callbacks():
    """Return active LangChain callbacks; LangSmith uses the config env bridge."""
    handler = get_langfuse_handler()
    return [handler] if handler is not None else []


def trace_metadata(name: str, source_file: str = "") -> dict[str, str]:
    """Return consistent, non-sensitive metadata for both tracing providers."""
    config = _config()
    relative_file = source_file
    if relative_file:
        try:
            relative_file = Path(relative_file).resolve().relative_to(config.ROOT_DIR).as_posix()
        except ValueError:
            relative_file = Path(relative_file).name
    day = config.COURSE_DAY
    if not day and relative_file:
        day_folder = Path(relative_file).parts[0]
        day = day_folder.split("_", maxsplit=1)[0] if day_folder.startswith("day") else ""
    metadata = {
        "course_day": day,
        "source_file": relative_file or config.COURSE_SOURCE_FILE,
        "run_id": config.COURSE_RUN_ID,
        "operation": name,
    }
    if config.PROMPT_VERSION:
        metadata["prompt_version"] = config.PROMPT_VERSION
    if config.TRACE_USER_ID:
        metadata["user_id"] = config.TRACE_USER_ID
    return metadata


def _course_source_file(frame: FrameType | None, fallback: str = "") -> str:
    config = _config()
    caller = frame.f_back if frame else None
    while caller is not None:
        candidate = Path(caller.f_code.co_filename).resolve()
        try:
            relative = candidate.relative_to(config.ROOT_DIR)
        except ValueError:
            relative = None
        if relative and relative.parts[0].startswith(("day1_", "day2_", "day3_", "day4_", "day5_")):
            return str(candidate)
        caller = caller.f_back
    return fallback


def get_run_config(
    *,
    session_id: str = "",
    prompt_version: str = "",
    user_id: str = "",
    model: str = "",
    provider: str = "",
) -> dict:
    """Build consistent LangChain run metadata and configured tracing callbacks."""
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    source_file = _course_source_file(
        frame,
        caller.f_code.co_filename if caller else "",
    )
    metadata = trace_metadata("langchain_run", source_file)
    if session_id:
        metadata["session_id"] = session_id
        metadata["thread_id"] = session_id
    if prompt_version:
        metadata["prompt_version"] = prompt_version
    if user_id:
        metadata["user_id"] = user_id
    if model:
        metadata["model"] = model
    if provider:
        metadata["provider"] = provider
    day = metadata["course_day"]
    tags = ["course-1"]
    if day:
        tags.append(day)
    if provider:
        tags.append(provider)
    return {
        "callbacks": get_callbacks(),
        "metadata": metadata,
        "tags": tags,
    }


def log_metric(name: str, value: float, **attributes: object) -> None:
    """Log a numeric measurement as structured fields, not an encoded message."""
    _config()
    source_file = str(attributes.pop("_source_file", ""))
    if not source_file:
        source_file = _course_source_file(inspect.currentframe())
    metadata = trace_metadata(name, source_file)
    _logger().info(
        "metric",
        extra={
            "metric_name": name,
            "metric_value": value,
            "metric_attributes": attributes,
            "course_day": metadata["course_day"],
            "source_file": metadata["source_file"],
            "run_id": metadata["run_id"],
        },
    )


def init_telemetry(service_name: str = "applied-ai-practice"):
    """Initialize optional Langfuse/ADK tracing and return the LangChain handler."""
    global _tracer_provider, _shutdown
    config = _config()
    if not isinstance(service_name, str) or not service_name.strip():
        raise ValueError("service_name must not be empty.")

    langfuse_enabled = bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)
    if langfuse_enabled and _shutdown:
        raise RuntimeError("Telemetry was already shut down in this process.")
    if langfuse_enabled and _tracer_provider is None:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        current_provider = trace.get_tracer_provider()
        if isinstance(current_provider, TracerProvider):
            # Langfuse 5.x or an earlier bootstrap may already own the global
            # TracerProvider; attach the ADK OTLP processor and instrumentor to it.
            provider = current_provider
        else:
            resource_attributes = {
                "service.name": service_name,
                "course.day": config.COURSE_DAY,
                "course.run_id": config.COURSE_RUN_ID,
            }
            provider = TracerProvider(resource=Resource.create(resource_attributes))
            trace.set_tracer_provider(provider)

        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=config.LANGFUSE_OTEL_ENDPOINT,
                    headers=config.LANGFUSE_OTEL_HEADERS,
                )
            )
        )
        instrumentor = GoogleADKInstrumentor()
        if not getattr(instrumentor, "is_instrumented_by_opentelemetry", False):
            instrumentor.instrument(tracer_provider=provider)
        _tracer_provider = provider
        _shutdown = False

        _logger().info(
            "ADK OpenTelemetry initialized",
            extra={
                "day_module": "utils.telemetry",
                "source_file": "utils/telemetry.py",
            },
        )
    elif not langfuse_enabled:
        _logger().info(
            "Provider tracing is disabled; using local structured logs.",
            extra={
                "day_module": "utils.telemetry",
                "source_file": "utils/telemetry.py",
            },
        )

    return config.get_langfuse_handler()


def get_tracer_provider():
    """Return the initialized OpenTelemetry provider, if export is configured."""
    return _tracer_provider


def trace_provider_status() -> dict[str, TraceProviderStatus]:
    """Report which trace providers are enabled and what credentials are missing."""
    config = _config()
    langsmith_missing = ["LANGSMITH_API_KEY"] if config.LANGSMITH_TRACING and not config.LANGSMITH_API_KEY else []
    langfuse_missing = [
        name
        for name, value in (
            ("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY),
            ("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY),
        )
        if not value
    ]
    return {
        "langsmith": {
            "enabled": bool(config.LANGSMITH_TRACING and config.LANGSMITH_API_KEY),
            "missing_credentials": langsmith_missing,
        },
        "langfuse": {
            "enabled": not langfuse_missing,
            "missing_credentials": langfuse_missing,
        },
        "langfuse_adk_otel": {
            "enabled": _tracer_provider is not None and not _shutdown,
            "configured": not langfuse_missing,
        },
    }


def _record_flush_status(
    provider: str,
    *,
    enabled: bool,
    status: str,
    missing_credentials: list[str] | None = None,
) -> None:
    _logger().info(
        "Telemetry provider flush status.",
        extra={
            "trace_provider": provider,
            "provider_enabled": enabled,
            "flush_status": status,
            "missing_credentials": missing_credentials or [],
        },
    )


def flush_traces() -> dict[str, str]:
    """Flush each enabled Langfuse/LangSmith client and report disabled providers."""
    provider_status = trace_provider_status()
    flush_statuses: dict[str, str] = {}
    failures: list[Exception] = []
    providers = (
        ("langfuse", get_langfuse_client),
        ("langsmith", get_langsmith_client),
    )
    for provider, client_factory in providers:
        settings = provider_status[provider]
        enabled = bool(settings["enabled"])
        missing_credentials = settings["missing_credentials"]
        if not enabled:
            flush_statuses[provider] = "disabled"
            _record_flush_status(
                provider,
                enabled=False,
                status="disabled",
                missing_credentials=missing_credentials,
            )
            continue
        try:
            client = client_factory()
            if client is None:
                raise RuntimeError(f"{provider} is enabled but its client is unavailable.")
            client.flush()
        except Exception as error:
            flush_statuses[provider] = "failed"
            _logger().exception(
                "Failed to flush traces for provider %s.",
                provider,
                extra={"trace_provider": provider},
            )
            failure = RuntimeError(f"{provider} trace flush failed.")
            failure.__cause__ = error
            failures.append(failure)
        else:
            flush_statuses[provider] = "flushed"
        _record_flush_status(
            provider,
            enabled=True,
            status=flush_statuses[provider],
        )
    if failures:
        raise ExceptionGroup("One or more trace providers failed to flush.", failures)
    return flush_statuses


def flush_telemetry() -> dict[str, str]:
    """Flush ADK OTLP and both trace clients, surfacing every provider failure."""
    global _shutdown
    provider_status = trace_provider_status()
    flush_statuses: dict[str, str] = {}
    failures: list[Exception] = []
    if _tracer_provider is None:
        status = "not_initialized" if provider_status["langfuse_adk_otel"]["configured"] else "disabled"
        flush_statuses["langfuse_adk_otel"] = status
        _record_flush_status(
            "langfuse_adk_otel",
            enabled=False,
            status=status,
            missing_credentials=provider_status["langfuse"]["missing_credentials"],
        )
    elif _shutdown:
        flush_statuses["langfuse_adk_otel"] = "already_shutdown"
        _record_flush_status(
            "langfuse_adk_otel",
            enabled=False,
            status="already_shutdown",
        )
    else:
        otel_errors: list[Exception] = []
        try:
            if not _tracer_provider.force_flush(timeout_millis=10_000):
                raise TimeoutError("Timed out while flushing Langfuse OTLP spans.")
        except Exception as error:
            otel_errors.append(error)
            _logger().exception("Failed to flush Langfuse ADK OpenTelemetry spans.")
        try:
            _tracer_provider.shutdown()
        except Exception as error:
            otel_errors.append(error)
            _logger().exception("Failed to shut down the ADK OpenTelemetry provider.")
        else:
            _shutdown = True
        status = "failed" if otel_errors else "flushed"
        flush_statuses["langfuse_adk_otel"] = status
        _record_flush_status("langfuse_adk_otel", enabled=True, status=status)
        if otel_errors:
            failures.extend(otel_errors)

    try:
        flush_statuses.update(flush_traces())
    except ExceptionGroup as error:
        failures.extend(error.exceptions)
    if failures:
        raise ExceptionGroup("Telemetry flush encountered provider failures.", failures)
    return flush_statuses


def trace_operation(
    name: str,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
    flush: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Add provider spans and duration/status metrics to an operation."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        config = _config()
        traced_function = function
        metadata = trace_metadata(name, function.__code__.co_filename)
        if config.LANGSMITH_TRACING:
            from langsmith import traceable

            trace_options = {
                "name": name,
                "project_name": config.LANGSMITH_PROJECT,
                "metadata": metadata,
            }
            langsmith_client = get_langsmith_client()
            if langsmith_client is not None:
                trace_options["client"] = langsmith_client
            if not capture_input:
                trace_options["process_inputs"] = lambda _: {"omitted": True}
            if not capture_output:
                trace_options["process_outputs"] = lambda _: {"omitted": True}
            traced_function = traceable(**trace_options)(traced_function)

        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
                settings = _config()
                started = perf_counter()
                status = "success"
                try:
                    if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
                        client = get_langfuse_client()
                        observation_input = {"args": args, "kwargs": kwargs} if capture_input else {"omitted": True}
                        with client.start_as_current_observation(
                            name=name,
                            as_type="span",
                            input=observation_input,
                            metadata=metadata,
                        ) as observation:
                            result = await traced_function(*args, **kwargs)
                            observation.update(output=result if capture_output else {"omitted": True})
                            return result
                    return await traced_function(*args, **kwargs)
                except Exception:
                    status = "error"
                    _logger().exception("Operation %s failed", name)
                    raise
                finally:
                    log_metric(
                        "operation_duration_ms",
                        round((perf_counter() - started) * 1000, 2),
                        _source_file=function.__code__.co_filename,
                        operation=name,
                        status=status,
                    )
                    if flush:
                        flush_traces()

            return async_wrapper

        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            settings = _config()
            started = perf_counter()
            status = "success"
            try:
                if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
                    client = get_langfuse_client()
                    observation_input = {"args": args, "kwargs": kwargs} if capture_input else {"omitted": True}
                    with client.start_as_current_observation(
                        name=name,
                        as_type="span",
                        input=observation_input,
                        metadata=metadata,
                    ) as observation:
                        result = traced_function(*args, **kwargs)
                        observation.update(output=result if capture_output else {"omitted": True})
                        return result
                return traced_function(*args, **kwargs)
            except Exception:
                status = "error"
                _logger().exception("Operation %s failed", name)
                raise
            finally:
                log_metric(
                    "operation_duration_ms",
                    round((perf_counter() - started) * 1000, 2),
                    _source_file=function.__code__.co_filename,
                    operation=name,
                    status=status,
                )
                if flush:
                    flush_traces()

        return wrapper

    return decorate
