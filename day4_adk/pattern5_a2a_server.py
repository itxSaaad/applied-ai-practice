import logging
import sys
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from day4_adk.model_config import get_adk_model

logger = logging.getLogger("A2AServer")


class A2AServer:
    """Expose an ADK agent through discovery and task-submission endpoints."""

    def __init__(self, agent: LlmAgent, agent_card: Dict[str, Any]):
        self.agent = agent
        self.agent_card = agent_card
        self.app_name = agent.name
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )

    def build_app(self):
        """Construct an ASGI app that runs submitted prompts through the ADK agent."""
        fastapi_app = FastAPI(
            title=self.agent_card.get("name", "A2A Agent Server"),
            description=self.agent_card.get("description", ""),
            version=self.agent_card.get("version", "1.0.0"),
        )

        @fastapi_app.get("/a2a/v1/agent-card")
        async def get_agent_card():
            return self.agent_card

        @fastapi_app.post("/a2a/v1/tasks")
        async def execute_task(task_payload: "TaskRequest"):
            user_id = "a2a-client"
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=uuid4().hex,
            )
            message = types.Content(
                role="user",
                parts=[types.Part(text=task_payload.prompt)],
            )
            response_text = []
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=message,
            ):
                if event.is_final_response() and event.content:
                    response_text.extend(
                        part.text for part in event.content.parts if part.text
                    )
            if not response_text:
                raise RuntimeError("The ADK agent completed without a text response.")
            return {
                "status": "completed",
                "agent": self.agent.name,
                "task_output": "\n".join(response_text),
            }

        return fastapi_app


class TaskRequest(BaseModel):
    prompt: str


# Instantiate core ADK Agent
incident_analyst = LlmAgent(
    name="incident_analyst",
    model=get_adk_model(),
    instruction="Analyze incident logs and database records to diagnose root causes.",
)

# Wrap agent in A2A Server mechanics
a2a_server = A2AServer(
    agent=incident_analyst,
    agent_card={
        "name": "Incident Analyst Remote Agent",
        "description": "ADK-powered agent capable of deep incident triage.",
        "version": "1.0.0",
        "endpoints": {"task": "/a2a/v1/tasks"},
    },
)

class A2AObservabilityMiddleware:
    def __init__(self, wrapped_app):
        self.wrapped_app = wrapped_app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.wrapped_app(scope, receive, send)
            return
        await self._trace_http_request(scope, receive, send)

    @config.trace_operation(
        "day4_a2a_http_request",
        capture_input=False,
        capture_output=False,
    )
    async def _trace_http_request(self, scope, receive, send):
        status_code = None

        async def record_response_status(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.wrapped_app(scope, receive, record_response_status)
        finally:
            logger.info(
                "metric operation=day4_a2a_http_request method=%s status_code=%s",
                scope.get("method"),
                status_code,
            )


# Build an observable ASGI application around the native or fallback server.
app = A2AObservabilityMiddleware(a2a_server.build_app())

def run_server():
    logger.info("Starting A2A server on http://127.0.0.1:8000")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    finally:
        config.flush_traces()


if __name__ == "__main__":
    print(f"Launching A2A Server for {incident_analyst.name} on http://0.0.0.0:8000...")
    run_server()
