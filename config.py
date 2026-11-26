"""Central project paths and environment-backed settings."""

import base64
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RUNBOOKS_DIR = DATA_DIR / "runbooks"
SEED_DIR = DATA_DIR / "seed"
POISONED_DIR = DATA_DIR / "poisoned"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DB_DIR = ARTIFACTS_DIR / "db"
CHROMA_DIR = ARTIFACTS_DIR / "chroma_db"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"
LOGS_DIR = ARTIFACTS_DIR / "logs"

for directory in (
    RUNBOOKS_DIR,
    SEED_DIR,
    POISONED_DIR,
    DB_DIR,
    CHROMA_DIR,
    BENCHMARKS_DIR,
    LOGS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

INCIDENTS_DB_PATH = str(DB_DIR / "incidents.db")
CHECKPOINTS_DB_PATH = str(DB_DIR / "checkpoints.sqlite")
CRAG_CHECKPOINTS_DB_PATH = str(DB_DIR / "crag_checkpoints.sqlite")
SENT_EMAILS_PATH = str(DB_DIR / "sent_emails.txt")
APP_LOG_PATH = str(LOGS_DIR / "app.jsonl")

ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
os.environ.setdefault("COURSE_RUN_ID", uuid4().hex)

COURSE_DAY = os.getenv("COURSE_DAY", "")
COURSE_SOURCE_FILE = os.getenv("COURSE_SOURCE_FILE", "")
COURSE_RUN_ID = os.getenv("COURSE_RUN_ID", "")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "local-v1")
TRACE_USER_ID = os.getenv("TRACE_USER_ID", "")

# LLM providers
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_REQUEST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "300"))
ADK_AGENT_RUN_TIMEOUT_SECONDS = int(os.getenv("ADK_AGENT_RUN_TIMEOUT_SECONDS", "900"))
if OLLAMA_REQUEST_TIMEOUT_SECONDS <= 0 or ADK_AGENT_RUN_TIMEOUT_SECONDS <= 0:
    raise ValueError("Model and agent timeouts must be positive integers.")

# LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true" if LANGSMITH_API_KEY else "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agentic-course")
LANGSMITH_OTEL_ENABLED = os.getenv("LANGSMITH_OTEL_ENABLED", "true").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
LANGFUSE_OTEL_ENDPOINT = f"{LANGFUSE_BASE_URL.rstrip('/')}/api/public/otel"
LANGFUSE_OTEL_HEADERS = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    _langfuse_auth = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    LANGFUSE_OTEL_HEADERS = {"Authorization": f"Basic {_langfuse_auth}"}

# SDK aliases are configured here; application modules consume the values above.
os.environ["LANGSMITH_TRACING"] = str(LANGSMITH_TRACING).lower()
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
os.environ["LANGSMITH_OTEL_ENABLED"] = str(LANGSMITH_OTEL_ENABLED).lower()
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGSMITH_TRACING).lower()
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
if LANGSMITH_PROJECT:
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
if LANGSMITH_ENDPOINT:
    os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGSMITH_ENDPOINT)


def get_offline_test_environment(*, course_day: str, source_file: str, run_id: str) -> dict[str, str]:
    """Create a provider-free child-process environment for acceptance tests."""
    environment = os.environ.copy()
    environment.update(
        {
            "COURSE_DAY": course_day,
            "COURSE_SOURCE_FILE": source_file.replace("\\", "/"),
            "COURSE_RUN_ID": run_id,
            "GOOGLE_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
            "LANGSMITH_API_KEY": "",
            "LANGSMITH_TRACING": "false",
            "LANGSMITH_OTEL_ENABLED": "false",
        }
    )
    return environment


from utils.logging import (  # noqa: E402
    StructuredFormatter,
    configure_logging,
    get_logger,
    get_structured_logger,
    redact_sensitive_data,
    redact_sensitive_text,
)

logger = get_logger("applied_ai_practice")

from utils.telemetry import (  # noqa: E402
    flush_telemetry,
    flush_traces,
    get_callbacks,
    get_langfuse_client,
    get_langfuse_handler,
    get_langsmith_client,
    get_run_config,
    get_tracer_provider,
    init_telemetry,
    log_metric,
    trace_metadata,
    trace_operation,
    trace_provider_status,
)
