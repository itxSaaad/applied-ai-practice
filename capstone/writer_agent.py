"""Writer Agent with LangGraph human approval interrupt() and idempotent Jira/Slack sinks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.types import interrupt

import config as app_config
from capstone.guardrails import redact_pii

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = app_config.get_logger(__name__)

OPSPILOT_OUT_DIR = app_config.ARTIFACTS_DIR / "opspilot"
JIRA_FILE = OPSPILOT_OUT_DIR / "jira_tickets.json"
SLACK_FILE = OPSPILOT_OUT_DIR / "slack_updates.json"


def _append_idempotent(file_path: Path, thread_id: str, record: dict[str, Any]) -> bool:
    """Write record once per thread_id so process restarts/resumes never duplicate side effects."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if file_path.exists():
        existing = json.loads(file_path.read_text(encoding="utf-8"))
    if any(item.get("thread_id") == thread_id for item in existing):
        return False
    existing.append({"thread_id": thread_id, **record})
    file_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return True


def draft_postmortem(question: str, context: str) -> str:
    """Create a concise redacted postmortem & remediation draft."""
    clean_context = redact_pii(context)
    return (
        f"## OpsPilot Incident Draft\n"
        f"**Request:** {redact_pii(question)}\n\n"
        f"**Findings & Remediation:**\n{clean_context}\n\n"
        f"**Proposed Actions:** Create Jira follow-up ticket and post #incidents Slack update."
    )


def writer_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Draft remediation plan and pause via interrupt() before writing Jira/Slack side effects."""
    thread_id = str(config.get("configurable", {}).get("thread_id", "")).strip()
    if not thread_id:
        raise ValueError("thread_id is required in configurable for idempotent approval.")

    context_parts = [
        state.get("knowledge_answer", ""),
        state.get("data_answer", ""),
        state.get("adk_answer", ""),
    ]
    combined_context = "\n".join(part for part in context_parts if part).strip() or "No prior findings attached."
    draft = draft_postmortem(state.get("question", ""), combined_context)

    decision = interrupt(
        {
            "action": "create_jira_and_slack_update",
            "thread_id": thread_id,
            "draft": draft,
        }
    )

    approved = False
    if isinstance(decision, dict):
        approved = bool(decision.get("approved", False))
    elif isinstance(decision, bool):
        approved = decision
    elif isinstance(decision, str):
        approved = decision.strip().lower() in {"yes", "approve", "approved", "true"}

    if not approved:
        logger.info("OpsPilot approval rejected.", extra={"thread_id": thread_id})
        app_config.log_metric("opspilot_hitl_approved", 0.0, thread_id=thread_id)
        return {
            "draft": draft,
            "approval_status": "rejected",
            "final_answer": f"{draft}\n\n**Status:** Approval rejected; no Jira ticket or Slack message was sent.",
        }

    jira_path = Path(config.get("configurable", {}).get("jira_file", JIRA_FILE))
    slack_path = Path(config.get("configurable", {}).get("slack_file", SLACK_FILE))

    jira_written = _append_idempotent(
        jira_path,
        thread_id,
        {"summary": f"OpsPilot remediation: {state.get('question', '')[:80]}", "status": "OPEN"},
    )
    slack_written = _append_idempotent(
        slack_path,
        thread_id,
        {"channel": "#incidents", "message": redact_pii(draft[:240])},
    )
    logger.info(
        "OpsPilot approval executed.",
        extra={"thread_id": thread_id, "jira_created": jira_written, "slack_posted": slack_written},
    )
    app_config.log_metric("opspilot_hitl_approved", 1.0, thread_id=thread_id)

    return {
        "draft": draft,
        "approval_status": "approved",
        "jira_created": jira_written,
        "slack_posted": slack_written,
        "final_answer": f"{draft}\n\n**Status:** Approved (jira_created={jira_written}, slack_posted={slack_written}).",
    }
