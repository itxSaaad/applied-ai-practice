import os
import sys
from pathlib import Path
from typing import Optional

# Ensure repository root is in sys.path so config can be imported from any location
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


def _extract_clean_text(message) -> str:
    """Extracts clean text content from AIMessage, handling multimodal/content-block formats."""
    if hasattr(message, "text") and message.text:
        return message.text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        if parts:
            return "".join(parts)
    return str(content)


def get_llm(use_local: bool = False, temperature: float = 0.0) -> BaseChatModel:
    """
    Instantiates a provider-agnostic LangChain chat model based on environment variables.

    Args:
        use_local: If True, routes execution to local Ollama runtime; otherwise uses Gemini Cloud.
        temperature: Sampling temperature controlling output variance.
    """
    if use_local:
        model_name = config.LOCAL_MODEL
        try:
            return ChatOllama(
                model=model_name,
                base_url=config.OLLAMA_BASE_URL,
                temperature=temperature,
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize local Ollama model '{model_name}'. "
                f"Verify that the Ollama daemon is active on port 11434. Details: {e}"
            )
    else:
        model_name = config.LLM_MODEL
        api_key = config.GOOGLE_API_KEY
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Ensure it is configured in your uncommitted .env file."
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )


def chat(prompt: str, use_local: Optional[bool] = None) -> str:
    """
    Dispatches a text prompt to the configured LLM wrapper.

    Passes tracing callbacks (Langfuse + LangSmith) and flushes pending traces.
    Returns clean text without raw thinking-signature lists.
    """
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    if use_local is not None:
        llm = get_llm(use_local=use_local)
        response = llm.invoke(prompt, config=invoke_config)
        config.flush_traces()
        return _extract_clean_text(response)

    # Verification contract handling for setup validation
    if prompt.strip().lower() == "ping":
        try:
            cloud_llm = get_llm(use_local=False)
            cloud_res = cloud_llm.invoke("ping", config=invoke_config)
            cloud_text = _extract_clean_text(cloud_res)
        except Exception as e:
            cloud_text = f"[Ping check: {e}]"

        try:
            local_llm = get_llm(use_local=True)
            local_res = local_llm.invoke("ping", config=invoke_config)
            local_text = _extract_clean_text(local_res)
        except Exception as e:
            local_text = f"[Ping check: {e}]"

        config.flush_traces()
        return (
            f"[Cloud: {config.LLM_MODEL}] -> {cloud_text}\n"
            f"[Local: {config.LOCAL_MODEL}] -> {local_text}"
        )

    llm = get_llm(use_local=False)
    response = llm.invoke(prompt, config=invoke_config)
    config.flush_traces()
    return _extract_clean_text(response)


if __name__ == "__main__":
    print("Testing Provider Routing & Dual Tracing (LangSmith + Langfuse)...")
    print(chat("ping"))
