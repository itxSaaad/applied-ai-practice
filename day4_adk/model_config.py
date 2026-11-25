import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

from google.adk.models.lite_llm import LiteLlm


def get_adk_model():
    """Returns a local Ollama model via LiteLLM when USE_LOCAL_LLM=true, else Gemini."""
    if config.USE_LOCAL_LLM:
        return LiteLlm(
            model=f"ollama_chat/{config.LOCAL_MODEL}",
            api_base=config.OLLAMA_BASE_URL,
            temperature=0,
            max_tokens=128,
            timeout=90,
            num_retries=0,
            think=False,
        )
    return config.LLM_MODEL
