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
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentMiddleware")


@tool
def calculate_metrics(x: int, y: int) -> str:
    """Calculates the sum of two integer metrics."""
    return str(x + y)


@tool
def query_employee_directory(department: str) -> str:
    """Look up active headcount for a specified department."""
    directory = {"engineering": "42 active staff", "support": "18 active staff"}
    return directory.get(department.lower(), "Department not found.")


# Custom Middleware Hook enforcing a maximum model-call budget
@wrap_model_call
def call_budget_middleware(request, handler):
    max_calls = 5
    # Inspect internal state dictionary key for current invocation count
    state = request.state if hasattr(request, "state") and request.state is not None else {}
    call_count = state.get("model_call_count", 0)

    if call_count >= max_calls:
        raise RuntimeError(f"Execution Terminated: Hard limit of {max_calls} model calls exceeded.")

    state["model_call_count"] = call_count + 1
    return handler(request)


model = init_chat_model(
    f"google_genai:{os.getenv('LLM_MODEL', 'gemini-3.5-flash-lite')}",
    temperature=0.0,
)
tools = [calculate_metrics, query_employee_directory]
system_prompt = "You are an enterprise operations assistant. Use tools when necessary."

# Modern v1 create_agent instantiating a LangGraph-backed execution engine
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[call_budget_middleware],
)


if __name__ == "__main__":
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    result = agent.invoke(
        {"messages": [("user", "What is the combined headcount of engineering and support?")]},
        config=invoke_config,
    )
    config.flush_traces()
    last_msg = result["messages"][-1]
    print("Agent Output:", getattr(last_msg, "text", last_msg.content))
