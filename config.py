import base64
import inspect
import logging
import os
import sys
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar
from dotenv import load_dotenv

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Always load the uncommitted .env from the repo root
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("applied_ai")

# LLM Providers Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LangSmith Configuration
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agentic-course")
LANGSMITH_OTEL_ENABLED = os.getenv("LANGSMITH_OTEL_ENABLED", "true").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "")

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
LANGFUSE_OTEL_ENDPOINT = f"{LANGFUSE_BASE_URL.rstrip('/')}/api/public/otel"
LANGFUSE_OTEL_HEADERS = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    _langfuse_auth = base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
    ).decode()
    LANGFUSE_OTEL_HEADERS = {"Authorization": f"Basic {_langfuse_auth}"}

# Automatic LangSmith bridge:
# LangChain internally checks LANGCHAIN_* variables. This bridge is the only
# environment mapping config needs to perform.
if LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
if LANGSMITH_PROJECT:
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
if LANGSMITH_ENDPOINT:
    os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGSMITH_ENDPOINT)

_langfuse_client = None
_langfuse_handler = None
_langsmith_client = None

def get_langfuse_client():
    """Returns the singleton Langfuse client instance."""
    global _langfuse_client
    if _langfuse_client is None and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        import langfuse

        _langfuse_client = langfuse.Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL,
        )
    return _langfuse_client

def get_langfuse_handler():
    """Returns a singleton Langfuse CallbackHandler for LangChain, or None if keys are missing."""
    global _langfuse_handler
    if _langfuse_handler is None:
        client = get_langfuse_client()
        if client:
            try:
                from langfuse.langchain import CallbackHandler
            except ImportError:
                from langfuse.callback import CallbackHandler

            _langfuse_handler = CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY)
    return _langfuse_handler

def get_langsmith_client():
    """Returns the singleton LangSmith client when tracing is configured."""
    global _langsmith_client
    if _langsmith_client is None and LANGSMITH_TRACING and os.getenv("LANGSMITH_API_KEY"):
        from langsmith import Client

        _langsmith_client = Client()
    return _langsmith_client

def get_callbacks():
    """Returns active LangChain callbacks; LangSmith tracing uses the env bridge."""
    handler = get_langfuse_handler()
    return [handler] if handler is not None else []

P = ParamSpec("P")
R = TypeVar("R")


def trace_operation(
    name: str,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
    flush: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Adds LangSmith/Langfuse spans and duration/status logs to an operation."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        traced_function = function
        if LANGSMITH_TRACING:
            from langsmith import traceable

            trace_options = {
                "name": name,
                "project_name": LANGSMITH_PROJECT,
            }
            langsmith_client = get_langsmith_client()
            if langsmith_client is not None:
                trace_options["client"] = langsmith_client
            if not capture_input:
                trace_options["process_inputs"] = lambda _: {"omitted": True}
            if not capture_output:
                trace_options["process_outputs"] = lambda _: {"omitted": True}
            traced_function = traceable(**trace_options)(traced_function)

        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            from langfuse import observe

            traced_function = observe(
                name=name,
                capture_input=capture_input,
                capture_output=capture_output,
            )(traced_function)

        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
                started = perf_counter()
                status = "success"
                try:
                    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
                        get_langfuse_client()
                    return await traced_function(*args, **kwargs)
                except Exception:
                    status = "error"
                    logger.exception("Operation %s failed", name)
                    raise
                finally:
                    logger.info(
                        "metric operation=%s duration_ms=%.2f status=%s",
                        name,
                        (perf_counter() - started) * 1000,
                        status,
                    )
                    if flush:
                        flush_traces()

            return async_wrapper

        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            started = perf_counter()
            status = "success"
            try:
                if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
                    get_langfuse_client()
                return traced_function(*args, **kwargs)
            except Exception:
                status = "error"
                logger.exception("Operation %s failed", name)
                raise
            finally:
                logger.info(
                    "metric operation=%s duration_ms=%.2f status=%s",
                    name,
                    (perf_counter() - started) * 1000,
                    status,
                )
                if flush:
                    flush_traces()

        return wrapper

    return decorate

def flush_traces():
    """Flushes queued Langfuse and LangSmith traces before script termination."""
    if _langfuse_client is not None:
        _langfuse_client.flush()

    if _langsmith_client is not None:
        _langsmith_client.flush()
