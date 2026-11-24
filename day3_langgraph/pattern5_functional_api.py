import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.func import entrypoint, task

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FunctionalAPI")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", config.LLM_MODEL),
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0,
)


@task
def fetch_user_data_task(user_id: str) -> str:
    """Isolated task step 1."""
    return f"User Record [{user_id}]: Preferred Role=DevOps, Region=us-east-1."


@task
def generate_summary_task(user_data: str, query: str) -> str:
    """Isolated task step 2."""
    prompt = f"Using context '{user_data}', answer: {query}"
    res = llm.invoke([HumanMessage(content=prompt)])
    res_text = getattr(res, "text", res.content)
    return str(res_text)


@entrypoint()
def workflow_entrypoint(inputs: dict) -> dict:
    """Imperative orchestrator control function using standard Python logic flow."""
    user_id = inputs.get("user_id", "default_user")
    query = inputs.get("query", "Summarize user context.")

    data_future = fetch_user_data_task(user_id)
    user_data = data_future.result()

    if "DevOps" in user_data:
        summary_future = generate_summary_task(user_data, query)
        result = summary_future.result()
    else:
        result = "Non-DevOps execution path bypassed analysis."

    return {"status": "SUCCESS", "output": result}


if __name__ == "__main__":
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    try:
        result = workflow_entrypoint.invoke(
            {"user_id": "usr_99", "query": "Check infrastructure profile."},
            config=invoke_config,
        )
        config.flush_traces()
        print("Functional API Result:", result)
    except Exception as e:
        logger.error(f"Functional API execution error: {e}")
