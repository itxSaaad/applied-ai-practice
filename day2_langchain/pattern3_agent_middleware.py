import csv
import json
from collections.abc import MutableMapping

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.tools import tool

import config
from day2_langchain.pattern2_safe_tool import query_read_only_sqlite
from llm import get_llm
from utils.logging import get_logger

logger = get_logger(__name__)
MAX_MODEL_CALLS = 5


@tool
def calculate_metrics(x: int, y: int) -> str:
    """Calculates the sum of two integer metrics."""
    return str(x + y)


@tool
def query_employee_directory(department: str) -> str:
    """Look up active headcount for a specified department."""
    directory_path = config.DATA_DIR / "employee_directory.json"
    directory = json.loads(directory_path.read_text(encoding="utf-8"))
    record = directory.get(department.lower())
    return f"{record['active_headcount']} active staff" if record else "Department not found."


# Custom Middleware Hook enforcing a maximum model-call budget
@wrap_model_call
def call_budget_middleware(request, handler):
    state = request.state if hasattr(request, "state") else None
    if not isinstance(state, MutableMapping):
        raise RuntimeError("The model-call budget requires a mutable dictionary state.")
    call_count = state.get("model_call_count", 0)

    if not isinstance(call_count, int) or isinstance(call_count, bool) or call_count < 0:
        raise ValueError("model_call_count must be a non-negative integer.")
    if call_count >= MAX_MODEL_CALLS:
        logger.warning(
            "Model-call budget exhausted.",
            extra={
                "tool_name": "model",
                "guardrail_triggered": True,
                "model_call_count": call_count,
                "model_call_limit": MAX_MODEL_CALLS,
                "day_module": "day2_langchain.pattern3_agent_middleware",
            },
        )
        raise RuntimeError(f"Execution terminated: model-call budget of {MAX_MODEL_CALLS} exhausted.")

    state["model_call_count"] = call_count + 1
    config.log_metric(
        "model_call_count",
        call_count + 1,
        maximum=MAX_MODEL_CALLS,
    )
    return handler(request)


tools = [calculate_metrics, query_employee_directory]
tools.append(query_read_only_sqlite)
system_prompt = (
    "You are an enterprise operations assistant. Use tools when necessary. "
    "The incidents table columns are incident_id, service, severity, status, "
    "title, and created_at. Query only columns from this schema. Seeded "
    "severity/status values are lowercase; use lower(column) comparisons "
    "when the user's casing may differ."
)


def build_agent(model_override=None):
    """Build the agent lazily so tool tests do not require a provider credential."""
    return create_agent(
        model=model_override or get_llm(),
        tools=tools,
        system_prompt=system_prompt,
        middleware=[call_budget_middleware],
    )


if __name__ == "__main__":
    agent = build_agent()
    with (config.DATA_DIR / "day2_agent_questions.csv").open(encoding="utf-8", newline="") as question_file:
        scripted_questions = list(csv.DictReader(question_file))
    for case in scripted_questions:
        invoke_config = config.get_run_config(prompt_version=config.PROMPT_VERSION)
        result = agent.invoke(
            {"messages": [("user", case["question"])]},
            config=invoke_config,
        )
        last_msg = result["messages"][-1]
        logger.info(
            "Completed scripted agent case %s; expected_tool=%s",
            case["question_id"],
            case["expected_tool"],
        )
        print(f"{case['question_id']}: " f"{getattr(last_msg, 'text', last_msg.content)}")
    config.flush_traces()
