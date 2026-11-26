import inspect
from typing import Any

from google.adk.models.lite_llm import LiteLlm

import config
from utils.rate_limit import async_acquire_rate_limit


def get_adk_model():
    """Returns a local Ollama model via LiteLLM when USE_LOCAL_LLM=true, else Gemini."""
    if config.USE_LOCAL_LLM:
        return LiteLlm(
            model=f"ollama_chat/{config.LOCAL_MODEL}",
            api_base=config.OLLAMA_BASE_URL,
            temperature=0,
            max_tokens=512,
            timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
            num_retries=0,
            think=False,
        )
    return config.LLM_MODEL


async def rate_limit_before_model_callback(callback_context: Any, llm_request: Any) -> None:
    """Throttle cloud Gemini ADK model calls to stay within RPM/TPM/RPD limits."""
    if config.USE_LOCAL_LLM:
        return
    await async_acquire_rate_limit(config.LLM_MODEL, llm_request)
    return


def compose_before_model_callbacks(*callbacks):
    """Run ADK before_model_callbacks sequentially, returning the first non-None result."""

    async def _combined(callback_context: Any, llm_request: Any):
        for callback in callbacks:
            result = callback(callback_context, llm_request)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    return _combined
