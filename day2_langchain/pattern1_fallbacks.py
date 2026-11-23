import logging
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelInit")


def build_resilient_model():
    # Primary model using Google AI Studio via langchain-google-genai
    primary_model_name = os.getenv("LLM_MODEL", config.LLM_MODEL)
    primary_llm = init_chat_model(
        f"google_genai:{primary_model_name}",
        temperature=0.0,
    )

    # Secondary fallback model using local Ollama instance via langchain-ollama
    fallback_model_name = os.getenv("LOCAL_MODEL", config.LOCAL_MODEL)
    fallback_llm = init_chat_model(
        f"ollama:{fallback_model_name}",
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.0,
    )

    # Configure LCEL fallback execution policy
    resilient_llm = primary_llm.with_fallbacks([fallback_llm])
    return resilient_llm


if __name__ == "__main__":
    llm = build_resilient_model()
    logger.info("Executing invocation through resilient fallback model pipeline...")
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    try:
        response = llm.invoke(
            [HumanMessage(content="Explain system idempotency in 2 sentences.")],
            config=invoke_config,
        )
        config.flush_traces()
        response_text = getattr(response, "text", response.content)
        logger.info(f"Response Received: {response_text}")
    except Exception as e:
        logger.warning(f"Invocation note: {e}")
