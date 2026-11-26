"""Google ADK incident-analyst exposed via A2A protocol with graceful fallback client."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

import config
from capstone.data_agent import run_read_only_sql
from day4_adk.model_config import get_adk_model, rate_limit_before_model_callback
from day4_adk.pattern3_guardrail_callbacks import enforce_read_only_sql
from utils.telemetry import init_telemetry

logger = config.get_logger(__name__)


def query_opspilot_incidents(sql: str) -> list[dict]:
    """Read-only SQL query tool over OpsPilot services, deploys, and incidents."""
    blocked = enforce_read_only_sql("query_incidents", {"sql": sql})
    if blocked is not None:
        raise ValueError(blocked["error"])
    return run_read_only_sql(sql)


incident_analyst_agent = LlmAgent(
    name="incident_analyst",
    model=get_adk_model(),
    description="Analyzes OpsPilot incidents, deploys, and service ownership over read-only SQL.",
    instruction=(
        "You are OpsPilot's Incident Analyst. Query `incidents`, `deploys`, and `services` "
        "using `query_opspilot_incidents` (read-only SELECT only) and provide a concise root-cause summary."
    ),
    tools=[query_opspilot_incidents],
    before_model_callback=rate_limit_before_model_callback,
)

app = FastAPI(title="OpsPilot ADK Incident Analyst (A2A)", version="0.1.0")
_sessions = InMemorySessionService()


class A2ATaskRequest(BaseModel):
    task_id: str = "task-1"
    prompt: str


@app.get("/.well-known/agent.json")
def agent_card() -> dict[str, Any]:
    """Publish A2A AgentCard metadata."""
    return {
        "name": "incident-analyst",
        "description": incident_analyst_agent.description,
        "url": os.getenv("OPSPILOT_ADK_URL", "http://127.0.0.1:8002"),
        "version": "0.1.0",
        "capabilities": {"streaming": False},
        "skills": [{"id": "incident-rca", "name": "Incident Root Cause Analysis"}],
    }


@app.post("/a2a/tasks")
async def execute_a2a_task(req: A2ATaskRequest) -> dict[str, Any]:
    """Execute an A2A task on the Google ADK incident_analyst agent."""
    init_telemetry(service_name="opspilot-adk-analyst")
    runner = Runner(agent=incident_analyst_agent, app_name="opspilot_adk", session_service=_sessions)
    session = await _sessions.create_session(app_name="opspilot_adk", user_id="supervisor")
    content = types.Content(role="user", parts=[types.Part.from_text(text=req.prompt)])
    parts: list[str] = []
    async for event in runner.run_async(user_id="supervisor", session_id=session.id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            parts.extend(p.text for p in event.content.parts if p.text)
    return {
        "task_id": req.task_id,
        "agent": "incident-analyst",
        "status": "completed",
        "output": "\n".join(parts).strip() or "No ADK response generated.",
    }
