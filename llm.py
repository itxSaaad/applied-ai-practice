import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

def get_llm(use_local: bool = False, temperature: float = 0.0) -> BaseChatModel:
    """
    Instantiates a provider-agnostic LangChain chat model based on environment variables.
    
    Args:
        use_local: If True, routes execution to local Ollama runtime; otherwise uses Gemini Cloud.
        temperature: Sampling temperature controlling output variance.
    """
    if use_local:
        model_name = os.getenv("LOCAL_MODEL", "qwen3:8b")
        try:
            return ChatOllama(
                model=model_name,
                temperature=temperature
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize local Ollama model '{model_name}'. "
                f"Verify that the Ollama daemon is active on port 11434. Details: {e}"
            )
    else:
        model_name = os.getenv("LLM_MODEL", "gemini-3.8-flash")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Ensure it is configured in your uncommitted .env file."
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )

def chat(prompt: str, use_local: Optional[bool] = None) -> str:
    """
    Dispatches a text prompt to the configured LLM wrapper.
    
    If use_local is None or prompt is 'ping', executes dual provider validation (Cloud & Local)
    to confirm routing and emit operational traces to both LangSmith and Langfuse.
    """
    if use_local is not None:
        llm = get_llm(use_local=use_local)
        response = llm.invoke(prompt)
        return str(response.content)

    # Verification contract handling for setup validation
    if prompt.strip().lower() == "ping":
        cloud_llm = get_llm(use_local=False)
        local_llm = get_llm(use_local=True)
        
        cloud_res = cloud_llm.invoke("ping")
        local_res = local_llm.invoke("ping")
        
        return (
            f"[Cloud: {os.getenv('LLM_MODEL', 'gemini-3.8-flash')}] -> {cloud_res.content}\n"
            f"[Local: {os.getenv('LOCAL_MODEL', 'qwen3:8b')}] -> {local_res.content}"
        )

    llm = get_llm(use_local=False)
    response = llm.invoke(prompt)
    return str(response.content)

if __name__ == "__main__":
    print("Testing Provider Routing...")
    print(chat("ping"))