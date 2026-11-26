"""ADK sub-agent port of the Day 3 researcher/analyst/writer supervisor."""

import asyncio
import csv
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import config
from day3_langgraph.pattern4_supervisor import search_runbooks as langchain_search_tool
from day4_adk.model_config import get_adk_model, rate_limit_before_model_callback
from day5_observability.pattern1_dual_observability import (
    configure_if_enabled,
    flush_observability,
)
from utils.logging import get_logger

logger = get_logger(__name__)


def search_runbooks(query: str) -> str:
    """Search the same runbook corpus used by the Day 3 researcher."""
    return langchain_search_tool.invoke({"query": query})


def query_incidents(sql: str) -> str:
    """Query the incident sample through the shared read-only SQL guard."""
    from day4_adk.pattern3_guardrail_callbacks import query_incidents as read_only_query

    return read_only_query(sql)


researcher = LlmAgent(
    name="researcher",
    model=get_adk_model(),
    description="Finds cited evidence in the local runbook corpus.",
    instruction="Use search_runbooks and return concise source-cited evidence.",
    tools=[search_runbooks],
    output_key="research_findings",
    before_model_callback=rate_limit_before_model_callback,
)

analyst = LlmAgent(
    name="analyst",
    model=get_adk_model(),
    description="Queries incident records using read-only SQL.",
    instruction=(
        "Use query_incidents with a single SELECT statement. Return the matching "
        "incident evidence and do not modify data."
    ),
    tools=[query_incidents],
    output_key="incident_findings",
    before_model_callback=rate_limit_before_model_callback,
)

writer = LlmAgent(
    name="writer",
    model=get_adk_model(),
    description="Produces the final concise response from worker evidence.",
    instruction=(
        "Write the final response using only the available research and incident "
        "findings. Cite runbook sources and distinguish facts from unknowns."
    ),
    output_key="final_response",
    before_model_callback=rate_limit_before_model_callback,
)

adk_supervisor = LlmAgent(
    name="incident_supervisor",
    model=get_adk_model(),
    description="Routes incident requests to researcher, analyst, and writer workers.",
    instruction=(
        "Delegate runbook questions to researcher and incident data questions to "
        "analyst. Use writer after the required evidence is available. Do not invent "
        "tool results."
    ),
    sub_agents=[researcher, analyst, writer],
    before_model_callback=rate_limit_before_model_callback,
)


@config.trace_operation("day4_adk_supervisor_tasks", flush=True)
async def run_scripted_tasks() -> list[str]:
    tracer_provider = configure_if_enabled()
    with (config.DATA_DIR / "day3_supervisor_tasks.csv").open(encoding="utf-8", newline="") as file:
        tasks = list(csv.DictReader(file))
    if len(tasks) != 5:
        raise ValueError(f"Expected five supervisor tasks; found {len(tasks)}.")

    sessions = InMemorySessionService()
    runner = Runner(
        agent=adk_supervisor,
        app_name="day4_adk_supervisor",
        session_service=sessions,
    )
    results = []
    try:
        for task in tasks:
            user_id = "course-demo"
            session = await sessions.create_session(
                app_name="day4_adk_supervisor",
                user_id=user_id,
                session_id=uuid4().hex,
            )
            message = types.Content(
                role="user",
                parts=[types.Part(text=task["task"])],
            )
            final_text = ""
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=message,
            ):
                if event.is_final_response() and event.content:
                    final_text = "\n".join(part.text for part in event.content.parts if part.text)
            if not final_text:
                raise RuntimeError(f"No final response for {task['task_id']}.")
            logger.info(
                "Completed ADK supervisor task %s.",
                task["task_id"],
                extra={"task_id": task["task_id"]},
            )
            results.append(final_text)
            print(f"{task['task_id']}: {final_text}")
    finally:
        if tracer_provider is not None:
            flush_observability(tracer_provider)
    return results


if __name__ == "__main__":
    asyncio.run(run_scripted_tasks())
