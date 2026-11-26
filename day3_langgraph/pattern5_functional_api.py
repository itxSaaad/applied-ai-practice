from langchain_core.messages import HumanMessage
from langgraph.func import entrypoint, task

import config
from llm import _extract_clean_text, get_llm
from utils.logging import get_logger

logger = get_logger(__name__)

llm = None


def _get_model():
    return llm if llm is not None else get_llm()


@task
def fetch_user_data_task(user_id: str) -> str:
    """Isolated task step 1."""
    return f"User Record [{user_id}]: Preferred Role=DevOps, Region=us-east-1."


@task
def generate_summary_task(user_data: str, query: str) -> str:
    """Isolated task step 2."""
    prompt = f"Using context '{user_data}', answer: {query}"
    res = _get_model().invoke(
        [HumanMessage(content=prompt)],
        config=config.get_run_config(),
    )
    return _extract_clean_text(res)


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
    invoke_config = config.get_run_config()

    try:
        result = workflow_entrypoint.invoke(
            {"user_id": "usr_99", "query": "Check infrastructure profile."},
            config=invoke_config,
        )
        config.flush_traces()
        print("Functional API Result:", result)
    except Exception as e:
        logger.error(f"Functional API execution error: {e}")
