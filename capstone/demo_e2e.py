"""End-to-end OpsPilot demo script simulating a real on-call user prompt and saving artifacts."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from langgraph.types import Command

import config
from capstone.a2a_client import call_adk_analyst
from capstone.ingest import ingest_knowledge_base, seed_opspilot_database
from capstone.observability import build_opspilot_run_config, summarize_system_metrics
from capstone.ops_pilot_supervisor import build_opspilot_graph
from capstone.writer_agent import JIRA_FILE, SLACK_FILE

E2E_REPORT_JSON = config.ARTIFACTS_DIR / "opspilot" / "e2e_demo_report.json"
E2E_REPORT_MD = config.ARTIFACTS_DIR / "opspilot" / "e2e_demo_report.md"


def run_local_e2e(prompt: str, *, adk_url: str = "http://127.0.0.1:8002") -> dict[str, Any]:
    """Run full local E2E flow: seed -> supervisor -> HITL interrupt -> resume -> artifacts."""
    t0 = perf_counter()
    db_counts = seed_opspilot_database()
    chunk_count = ingest_knowledge_base()

    thread_id = f"e2e-{uuid.uuid4().hex[:8]}"
    run_cfg = build_opspilot_run_config(
        thread_id=thread_id, scenario_id="payments-api-spike", agent_name="opspilot_e2e"
    )
    run_cfg["configurable"] = {"thread_id": thread_id}

    graph = build_opspilot_graph(adk_url=adk_url)

    t_turn1 = perf_counter()
    graph.invoke({"question": prompt}, config=run_cfg)
    turn1_ms = (perf_counter() - t_turn1) * 1000
    paused_state = graph.get_state({"configurable": {"thread_id": thread_id}})

    t_turn2 = perf_counter()
    resumed = graph.invoke(Command(resume={"approved": True}), config=run_cfg)
    turn2_ms = (perf_counter() - t_turn2) * 1000

    adk_result = call_adk_analyst(
        "Summarize payments-api recent deploys and open incidents", base_url=adk_url, timeout=2.0
    )

    metrics = summarize_system_metrics([turn1_ms, turn2_ms])
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": "local",
        "thread_id": thread_id,
        "prompt": prompt,
        "seed_counts": db_counts,
        "runbook_chunks": chunk_count,
        "interrupted_for_approval": bool(paused_state.next),
        "approval_status": resumed.get("approval_status"),
        "jira_created": resumed.get("jira_created", False),
        "slack_posted": resumed.get("slack_posted", False),
        "citations": resumed.get("citations", []),
        "adk_degraded": adk_result.get("degraded", True),
        "final_answer": resumed.get("final_answer", ""),
        "jira_file": str(JIRA_FILE),
        "slack_file": str(SLACK_FILE),
        "system_metrics": metrics,
        "total_duration_ms": round((perf_counter() - t0) * 1000, 2),
    }
    _write_reports(report)
    return report


def run_live_api_e2e(prompt: str, *, api_url: str = "http://127.0.0.1:8000") -> dict[str, Any]:
    """Run full E2E flow against a live OpsPilot FastAPI + Docker Compose stack."""
    thread_id = f"live-{uuid.uuid4().hex[:8]}"
    t0 = perf_counter()
    health = httpx.get(f"{api_url.rstrip('/')}/health", timeout=5.0)
    health.raise_for_status()

    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        f"{api_url.rstrip('/')}/chat",
        json={"message": prompt, "thread_id": thread_id},
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))

    resumed = httpx.post(
        f"{api_url.rstrip('/')}/threads/{thread_id}/resume",
        json={"approved": True},
        timeout=30.0,
    )
    resumed.raise_for_status()
    resume_body = resumed.json()

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": "live_api",
        "api_url": api_url,
        "thread_id": thread_id,
        "prompt": prompt,
        "sse_event_count": len(events),
        "interrupted_for_approval": any(e.get("interrupted") for e in events),
        "resume_response": resume_body,
        "total_duration_ms": round((perf_counter() - t0) * 1000, 2),
    }
    _write_reports(report)
    return report


def _write_reports(report: dict[str, Any]) -> None:
    E2E_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    E2E_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_lines = [
        "# OpsPilot End-to-End Demo Report",
        "",
        f"- **Timestamp:** {report['timestamp']}",
        f"- **Mode:** {report['mode']}",
        f"- **Thread ID:** `{report['thread_id']}`",
        f"- **Prompt:** {report['prompt']}",
        f"- **Interrupted for approval:** {report.get('interrupted_for_approval')}",
        f"- **Total duration:** {report.get('total_duration_ms')} ms",
        "",
        "## Artifacts Generated",
        f"- JSON report: `{E2E_REPORT_JSON}`",
        f"- Jira tickets: `{JIRA_FILE}`",
        f"- Slack updates: `{SLACK_FILE}`",
    ]
    E2E_REPORT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpsPilot end-to-end user scenario.")
    parser.add_argument(
        "--prompt",
        default="Investigate payments-api latency spike, check open incidents and runbooks, and create a Jira ticket.",
    )
    parser.add_argument("--api-url", default="", help="Optional live FastAPI URL (e.g. http://127.0.0.1:8000).")
    parser.add_argument("--adk-url", default="http://127.0.0.1:8002", help="ADK A2A service URL.")
    args = parser.parse_args()

    if args.api_url:
        report = run_live_api_e2e(args.prompt, api_url=args.api_url)
    else:
        report = run_local_e2e(args.prompt, adk_url=args.adk_url)

    print(json.dumps({k: v for k, v in report.items() if k != "final_answer"}, indent=2))
    print(f"\nArtifacts saved under: {Path(E2E_REPORT_JSON).parent}")


if __name__ == "__main__":
    main()
