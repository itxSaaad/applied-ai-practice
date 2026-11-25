import os
import sys
from pathlib import Path
from typing import Optional

# Ensure repository root is in sys.path so config can be imported from any location
REPO_ROOT = Path(__file__).resolve().parent
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


def get_llm(use_local: Optional[bool] = None, temperature: float = 0.0) -> BaseChatModel:
    """
    Instantiates a provider-agnostic LangChain chat model based on environment variables.

    Args:
        use_local: If True, routes execution to local Ollama runtime; otherwise uses Gemini Cloud.
        temperature: Sampling temperature controlling output variance.
    """
    if use_local is None:
        use_local = config.USE_LOCAL_LLM

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


@config.trace_operation("day1_chat", flush=True)
def chat(
    prompt: str,
    use_local: Optional[bool] = None,
    temperature: float = 0.0,
) -> str:
    """
    Dispatches a text prompt to the configured LLM wrapper.

    Passes tracing callbacks (Langfuse + LangSmith) and flushes pending traces.
    Returns clean text without raw thinking-signature lists.
    """
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    llm = get_llm(use_local=use_local, temperature=temperature)
    response = llm.invoke(prompt, config=invoke_config)
    return _extract_clean_text(response)


if __name__ == "__main__":
    print("Local:", chat("Reply with OK.", use_local=True))
    print("Cloud:", chat("Reply with OK.", use_local=False))
