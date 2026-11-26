from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

import config
from llm import _extract_clean_text, get_chat_model
from utils.logging import get_logger

logger = get_logger(__name__)


def build_resilient_model(primary_model=None, fallback_model=None):
    # Primary model using Google AI Studio via langchain-google-genai
    primary_llm = primary_model
    if primary_llm is None:
        primary_model_name = config.LLM_MODEL
        primary_llm = get_chat_model(
            f"google_genai:{primary_model_name}",
            temperature=0.0,
        )

    # Secondary fallback model using local Ollama instance via langchain-ollama
    if fallback_model is None:
        fallback_model_name = config.LOCAL_MODEL
        fallback_model = init_chat_model(
            f"ollama:{fallback_model_name}",
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
        )

    # Configure LCEL fallback execution policy
    return primary_llm.with_fallbacks([fallback_model])


if __name__ == "__main__":
    llm = build_resilient_model()
    logger.info("Executing invocation through resilient fallback model pipeline...")
    invoke_config = config.get_run_config()

    try:
        response = llm.invoke(
            [HumanMessage(content="Explain system idempotency in 2 sentences.")],
            config=invoke_config,
        )
        config.flush_traces()
        response_text = _extract_clean_text(response)
        logger.info(f"Response Received: {response_text}")
    except Exception as e:
        logger.warning(f"Invocation note: {e}")
