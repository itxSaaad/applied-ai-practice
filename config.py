import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Always load the uncommitted .env from the repo root
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Automatic LangSmith bridge:
# LangChain internally checks LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, LANGCHAIN_ENDPOINT.
# Map LANGSMITH_* to LANGCHAIN_* in os.environ without mutating the user's .env file.
if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
if api_key := os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
if project := os.getenv("LANGSMITH_PROJECT"):
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
if endpoint := os.getenv("LANGSMITH_ENDPOINT"):
    os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)

# LLM Providers Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LangSmith Configuration
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agentic-course")

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

_langfuse_handler = None

def get_langfuse_handler():
    """Returns a singleton Langfuse CallbackHandler for LangChain, or None if keys are missing."""
    global _langfuse_handler
    if _langfuse_handler is None and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            from langfuse.callback import CallbackHandler
            _langfuse_handler = CallbackHandler(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_BASE_URL
            )
        except Exception as e:
            print(f"[config] Warning: Failed to initialize Langfuse callback: {e}")
    return _langfuse_handler

def get_callbacks():
    """Returns the list of active callbacks (e.g. Langfuse) to pass to LangChain invocations."""
    handler = get_langfuse_handler()
    return [handler] if handler is not None else []

def flush_traces():
    """Flushes queued background traces (Langfuse & LangSmith) before script termination."""
    handler = get_langfuse_handler()
    if handler:
        try:
            handler.flush()
        except Exception:
            pass
