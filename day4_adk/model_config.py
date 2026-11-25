import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from google.adk.models.lite_llm import LiteLlm


def get_adk_model():
    """Returns a local Ollama model via LiteLLM when USE_LOCAL_LLM=true, else Gemini."""
    if os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
        return LiteLlm(
            model=f"ollama_chat/{os.getenv('LOCAL_MODEL', 'qwen3:8b')}",
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            temperature=0,
            max_tokens=128,
            timeout=90,
            num_retries=0,
            think=False,
        )
    return os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
