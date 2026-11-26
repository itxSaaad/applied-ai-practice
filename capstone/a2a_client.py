"""A2A client for calling the remote Google ADK incident-analyst with graceful fallback."""

from __future__ import annotations

import os
from typing import Any

import httpx

import config

logger = config.get_logger(__name__)


@config.trace_operation("opspilot_a2a_call")
def call_adk_analyst(prompt: str, base_url: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
    """Call remote ADK A2A service; return graceful degraded response if unreachable."""
    target = (base_url or os.getenv("OPSPILOT_ADK_URL", "http://127.0.0.1:8002")).rstrip("/")
    try:
        resp = httpx.post(
            f"{target}/a2a/tasks",
            json={"task_id": "opspilot-a2a", "prompt": prompt},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        config.log_metric("opspilot_a2a_degraded", 0.0)
        return {"degraded": False, "source": "adk-a2a", "analysis": data.get("output", "")}
    except Exception as exc:
        logger.warning(
            "ADK A2A service unreachable; returning degraded fallback.",
            extra={"agent_name": "a2a_client", "tool_name": "call_adk_analyst", "error": str(exc)},
        )
        config.log_metric("opspilot_a2a_degraded", 1.0)
        return {
            "degraded": True,
            "source": "adk-a2a-fallback",
            "error": str(exc),
            "analysis": (
                "[Degraded Mode] ADK Incident Analyst service is currently unreachable; "
                "continuing with local LangGraph Knowledge and Data agents."
            ),
        }
