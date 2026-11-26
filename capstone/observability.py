"""Three-Pillars observability helpers: P50/P95 latency, token cost, and milestone/service tags."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import config


def build_opspilot_run_config(
    *,
    thread_id: str,
    scenario_id: str = "",
    agent_name: str = "opspilot_supervisor",
    service: str = "opspilot-api",
    milestone: str = "capstone",
    day: str = "days-6-10",
    prompt_version: str = "",
) -> dict[str, Any]:
    """Attach Langfuse/LangSmith callbacks, metadata, and milestone/day/service tags."""
    cfg = config.get_run_config(
        session_id=thread_id,
        prompt_version=prompt_version or config.PROMPT_VERSION or "v1",
        user_id=config.TRACE_USER_ID or "oncall-engineer",
        model=config.LOCAL_MODEL if config.USE_LOCAL_LLM else config.LLM_MODEL,
        provider="ollama" if config.USE_LOCAL_LLM else "gemini",
    )
    extra_tags = [
        f"milestone:{milestone}",
        f"day:{day}",
        f"service:{service}",
        f"agent:{agent_name}",
    ]
    cfg["tags"] = list(dict.fromkeys([*cfg.get("tags", []), *extra_tags]))
    cfg["metadata"].update(
        {
            "environment": os.getenv("OPSPILOT_ENV", "local"),
            "scenario_id": scenario_id,
            "agent_name": agent_name,
            "service": service,
            "milestone": milestone,
            "course_day": day,
        }
    )
    return cfg


def summarize_system_metrics(
    latencies_ms: list[float],
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    errors: int = 0,
) -> dict[str, float | int]:
    """Compute P50/P95 latency, token totals, estimated API cost, and error rate."""
    arr = np.asarray(latencies_ms or [0.0], dtype=float)
    total = max(len(latencies_ms), 1)
    est_cost = (input_tokens * 0.0000001) + (output_tokens * 0.0000004)
    return {
        "turns": len(latencies_ms),
        "p50_latency_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(est_cost, 6),
        "error_rate": round(errors / total, 4),
    }
