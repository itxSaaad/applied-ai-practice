import asyncio
import re
from datetime import datetime, timedelta

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import config
from day4_adk.model_config import (
    compose_before_model_callbacks,
    get_adk_model,
    rate_limit_before_model_callback,
)
from day4_adk.pattern3_guardrail_callbacks import (
    before_model_guardrail,
    enforce_read_only_sql,
    query_incidents,
)
from day5_observability.pattern1_dual_observability import (
    configure_if_enabled,
    flush_observability,
)


# Idiomatic ADK 2.x plain Python function tool definition
@config.trace_operation("day4_read_log")
def read_log(service: str, minutes: int) -> str:
    """Reads the tracked local log fixture for a service and trailing time window."""
    if not re.fullmatch(r"[a-z0-9-]+", service):
        raise ValueError("Service name must contain only lowercase letters, digits, or hyphens.")
    if minutes <= 0:
        raise ValueError("minutes must be a positive integer.")
    logs_dir = config.DATA_DIR / "sample_logs"
    path = logs_dir / f"{service}.log"
    if not path.exists():
        available = sorted(p.stem for p in logs_dir.glob("*.log"))
        return f"No log file found for '{service}'. " f"Available log services: {', '.join(available)}."
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed = [(datetime.fromisoformat(line[:20].replace("Z", "+00:00")), line) for line in lines]
    latest = max(timestamp for timestamp, _ in parsed)
    lower_bound = latest - timedelta(minutes=minutes)
    selected = [line for timestamp, line in parsed if timestamp >= lower_bound]
    config.log_metric(
        "log_lines_returned",
        len(selected),
        service=service,
        window_minutes=minutes,
    )
    return "\n".join(selected)


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
    before_model_callback=rate_limit_before_model_callback,
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
    before_model_callback=rate_limit_before_model_callback,
)

check_checkout = LlmAgent(
    name="check_checkout_logs",
    model=get_adk_model(),
    instruction=(
        "Call read_log exactly once for checkout-service with minutes=15. "
        "Return its result in one concise sentence without additional analysis."
    ),
    tools=[read_log],
    output_key="checkout_logs",
    before_model_callback=rate_limit_before_model_callback,
)

check_worker = LlmAgent(
    name="check_worker_logs",
    model=get_adk_model(),
    instruction=(
        "Call read_log exactly once for worker-service with minutes=15. "
        "Return its result in one concise sentence without additional analysis."
    ),
    tools=[read_log],
    output_key="worker_logs",
    before_model_callback=rate_limit_before_model_callback,
)

incident_triage = LlmAgent(
    name="incident_triage",
    model=get_adk_model(),
    instruction=(
        "Use query_incidents with a read-only SELECT to identify current incidents "
        "before the parallel log sweep. Summarize only returned evidence."
    ),
    tools=[query_incidents, read_log],
    output_key="triage_findings",
    before_tool_callback=enforce_read_only_sql,
    before_model_callback=compose_before_model_callbacks(before_model_guardrail, rate_limit_before_model_callback),
)

parallel_log_checker = ParallelAgent(
    name="parallel_log_checker",
    sub_agents=[check_auth, check_payment, check_checkout, check_worker],
)


# Root cause summarizer reading triage and all parallel outputs from state
root_cause_summarizer = LlmAgent(
    name="root_cause_summarizer",
    model=get_adk_model(),
    instruction=(
        "Review database triage: {triage_findings}\n"
        "Auth logs: {auth_logs}\nPayment logs: {payment_logs}\n"
        "Checkout logs: {checkout_logs}\nWorker logs: {worker_logs}\n"
        "Summarize the findings and likely root cause in no more than two "
        "sentences. Do not speculate or include reasoning."
    ),
    output_key="final_root_cause",
    before_model_callback=rate_limit_before_model_callback,
)

# Complete pipeline
incident_investigation_pipeline = SequentialAgent(
    name="incident_investigation_pipeline",
    sub_agents=[incident_triage, parallel_log_checker, root_cause_summarizer],
)


@config.trace_operation("day4_pipeline_run", flush=True)
async def main():
    tracer_provider = configure_if_enabled()
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="day4_pipeline", user_id="dev_user", session_id="sess_1")
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
                for part in event.content.parts or []:
                    if part.text:
                        print(f"[{event.author}]: {part.text}")

    try:
        await asyncio.wait_for(
            consume_events(),
            timeout=config.ADK_AGENT_RUN_TIMEOUT_SECONDS,
        )
    finally:
        if tracer_provider is not None:
            flush_observability(tracer_provider)


if __name__ == "__main__":
    asyncio.run(main())
