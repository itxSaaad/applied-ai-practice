"""FastAPI SSE streaming API for OpsPilot (/chat, /threads/{thread_id}/resume, /health)."""

from __future__ import annotations

import json
import uuid
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

import config
from capstone.observability import build_opspilot_run_config
from capstone.ops_pilot_supervisor import build_opspilot_graph
from capstone.seed import OPSPILOT_DB_PATH, seed_opspilot_database

logger = config.get_logger(__name__)
app = FastAPI(title="OpsPilot Incident Response Copilot", version="0.1.0")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    approved: bool = True


@lru_cache(maxsize=1)
def _get_graph():
    if not OPSPILOT_DB_PATH.exists():
        seed_opspilot_database(OPSPILOT_DB_PATH)
    return build_opspilot_graph()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "opspilot"}


@app.post("/chat")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream OpsPilot LangGraph execution updates via Server-Sent Events (SSE)."""
    thread_id = req.thread_id or str(uuid.uuid4())
    run_cfg: dict[str, Any] = {
        **build_opspilot_run_config(
            thread_id=thread_id,
            scenario_id="api-chat",
            agent_name="opspilot_api",
            service="opspilot-api",
            milestone="p2-orchestration",
            day="day7",
        ),
        "configurable": {"thread_id": thread_id},
    }

    def event_generator():
        graph = _get_graph()
        try:
            for update in graph.stream(
                {"question": req.message},
                config=run_cfg,
                stream_mode="updates",
            ):
                for node_name, node_payload in update.items():
                    payload = {
                        "event": "node_update",
                        "name": node_name,
                        "thread_id": thread_id,
                        "keys": list(node_payload.keys()) if isinstance(node_payload, dict) else [],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
            final_payload = {
                "event": "done",
                "thread_id": thread_id,
                "interrupted": bool(snapshot.next),
                "values": {
                    k: v
                    for k, v in (snapshot.values or {}).items()
                    if k in {"final_answer", "draft", "approval_status", "citations"}
                },
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'thread_id': thread_id, 'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/threads/{thread_id}/resume")
def resume_thread(thread_id: str, req: ResumeRequest) -> dict[str, Any]:
    """Resume an interrupted OpsPilot thread using Command(resume=...)."""
    run_cfg: dict[str, Any] = {
        **build_opspilot_run_config(
            thread_id=thread_id,
            scenario_id="api-resume",
            agent_name="opspilot_api",
            service="opspilot-api",
            milestone="p2-orchestration",
            day="day7",
        ),
        "configurable": {"thread_id": thread_id},
    }
    result = _get_graph().invoke(Command(resume={"approved": req.approved}), config=run_cfg)
    return {
        "thread_id": thread_id,
        "approval_status": result.get("approval_status"),
        "jira_created": result.get("jira_created", False),
        "slack_posted": result.get("slack_posted", False),
        "final_answer": result.get("final_answer", ""),
    }
