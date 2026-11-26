import json
import sqlite3
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent

import config
from data.seed.seed_incidents_db import seed_database
from day4_adk.model_config import (
    compose_before_model_callbacks,
    get_adk_model,
    rate_limit_before_model_callback,
)
from utils.logging import get_logger
from utils.sqlite import fetch_read_only_rows, is_single_select

DATABASE = Path(config.INCIDENTS_DB_PATH)
logger = get_logger(__name__)


def query_incidents(sql: str) -> str:
    """Executes one read-only SELECT query against the seeded incidents database.

    The incidents table has columns: incident_id, service, severity, status,
    title, and created_at. Seeded severity and status values are lowercase.
    """
    if not is_single_select(sql):
        raise ValueError("Only one SELECT statement may be executed.")
    if not DATABASE.exists():
        seed_database(database=DATABASE)
    try:
        rows = fetch_read_only_rows(DATABASE, sql)
    except sqlite3.Error as error:
        return json.dumps(
            {
                "error": str(error),
                "table": "incidents",
                "columns": [
                    "incident_id",
                    "service",
                    "severity",
                    "status",
                    "title",
                    "created_at",
                ],
            },
            ensure_ascii=False,
        )
    config.log_metric("incident_query_rows", len(rows), read_only=True)
    return json.dumps(rows, ensure_ascii=False)


@config.trace_operation(
    "day4_read_only_sql_guardrail",
    capture_input=False,
    capture_output=False,
)
def enforce_read_only_sql(tool_name: Any, tool_args: dict[str, Any], tool_context: Any = None) -> dict[str, Any] | None:
    """Callback to block non-SELECT SQL queries strictly before execution."""
    resolved_name = getattr(tool_name, "name", tool_name)
    if resolved_name == "query_incidents":
        query = tool_args.get("sql", "")
        if not is_single_select(query):
            logger.warning(
                "Blocked non-read-only database tool call.",
                extra={
                    "tool_name": str(resolved_name),
                    "guardrail_triggered": True,
                    "day_module": "day4_adk.pattern3_guardrail_callbacks",
                },
            )
            # Short-circuit execution by returning an error dictionary
            return {"error": "Security Violation: Only read-only SELECT queries are allowed."}
    return None


def redact_pii_from_prompt(prompt_text: str) -> str:
    """Utility function to strip email addresses and phone numbers."""
    return config.redact_sensitive_text(prompt_text)


@config.trace_operation(
    "day4_pii_redaction_guardrail",
    capture_input=False,
    capture_output=False,
)
def before_model_guardrail(agent_name: Any, prompt: Any) -> str | None:
    """Redact direct prompt strings and ADK LlmRequest message contents."""
    if isinstance(prompt, str):
        redacted = redact_pii_from_prompt(prompt)
        if redacted != prompt:
            logger.warning(
                "Redacted PII before model invocation.",
                extra={
                    "guardrail_triggered": True,
                    "day_module": "day4_adk.pattern3_guardrail_callbacks",
                },
            )
        return redacted

    # ADK calls this with (CallbackContext, LlmRequest); redact user/model text
    # in place and return None so the guarded request proceeds normally.
    contents = getattr(prompt, "contents", None)
    if contents is not None:
        for content in contents:
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    redacted = redact_pii_from_prompt(text)
                    if redacted != text:
                        logger.warning(
                            "Redacted PII before model invocation.",
                            extra={
                                "guardrail_triggered": True,
                                "day_module": "day4_adk.pattern3_guardrail_callbacks",
                            },
                        )
                    part.text = redacted
    return None


# Explicit agent binding with callbacks passed into constructor
guarded_agent = LlmAgent(
    name="guarded_db_agent",
    model=get_adk_model(),
    instruction="Execute database lookups based on user requests.",
    tools=[query_incidents],
    before_tool_callback=enforce_read_only_sql,
    before_model_callback=compose_before_model_callbacks(before_model_guardrail, rate_limit_before_model_callback),
)


if __name__ == "__main__":
    print("PII Redaction Test:", before_model_guardrail("guarded_db_agent", "Contact dev@company.com or 555-123-4567"))
    print("Blocked SQL Test:", enforce_read_only_sql("query_incidents", {"sql": "DROP TABLE incidents"}))
    print("Allowed SQL Test:", enforce_read_only_sql("query_incidents", {"sql": "SELECT * FROM incidents"}))
    config.flush_traces()
