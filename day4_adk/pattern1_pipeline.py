import asyncio
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from day4_adk.model_config import get_adk_model


# Idiomatic ADK 2.x plain Python function tool definition
@config.trace_operation("day4_read_log")
def read_log(service: str, minutes: int) -> str:
    """Reads logs for a specified service over the given past minutes."""
    return f"Logs for {service} over last {minutes}m: No critical errors found."


# Sub-agents for parallel log checking, writing to isolated output_key entries
check_auth = LlmAgent(
    name="check_auth_logs",
    model=get_adk_model(),
    instruction=(
        "Call read_log exactly once for auth-service with minutes=15. "
        "Return its result in one concise sentence without additional analysis."
    ),
    tools=[read_log],
    output_key="auth_logs",
)

check_payment = LlmAgent(
    name="check_payment_logs",
    model=get_adk_model(),
    instruction=(
        "Call read_log exactly once for payment-service with minutes=15. "
        "Return its result in one concise sentence without additional analysis."
    ),
    tools=[read_log],
    output_key="payment_logs",
)

parallel_log_checker = ParallelAgent(
    name="parallel_log_checker",
    sub_agents=[check_auth, check_payment],
)


def normalize_no_incident_summary(callback_context: CallbackContext):
    auth_logs = str(callback_context.state.get("auth_logs", "")).lower()
    payment_logs = str(callback_context.state.get("payment_logs", "")).lower()
    if "no critical errors" in auth_logs and "no critical errors" in payment_logs:
        summary = (
            "Both auth-service and payment-service logs for the last 15 minutes "
            "show no critical errors."
        )
        callback_context.state["final_root_cause"] = summary
        return types.Content(role="model", parts=[types.Part(text=summary)])


# Root cause summarizer reading aggregated outputs from state
root_cause_summarizer = LlmAgent(
    name="root_cause_summarizer",
    model=get_adk_model(),
    instruction=(
        "Review auth logs: {auth_logs}\nPayment logs: {payment_logs}\n"
        "Summarize the findings and likely root cause in no more than two "
        "sentences. Do not speculate or include reasoning."
    ),
    output_key="final_root_cause",
    after_agent_callback=normalize_no_incident_summary,
)

# Complete pipeline
incident_investigation_pipeline = SequentialAgent(
    name="incident_investigation_pipeline",
    sub_agents=[parallel_log_checker, root_cause_summarizer],
)


@config.trace_operation("day4_pipeline_run", flush=True)
async def main():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="day4_pipeline", user_id="dev_user", session_id="sess_1"
    )
    runner = Runner(
        agent=incident_investigation_pipeline,
        app_name="day4_pipeline",
        session_service=session_service,
    )
    msg = types.Content(role="user", parts=[types.Part(text="Run incident log sweep.")])
    print("[Pipeline]: Starting incident log sweep...")

    async def consume_events():
        async for event in runner.run_async(
            user_id="dev_user",
            session_id=session.id,
            new_message=msg,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        print(f"[{event.author}]: {part.text}")

    await asyncio.wait_for(consume_events(), timeout=240)


if __name__ == "__main__":
    asyncio.run(main())
