import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from google.adk.agents import LlmAgent
from day4_adk.model_config import get_adk_model


def query_incidents(sql: str) -> str:
    """Executes a SQL query against the incidents database."""
    return f"Executed query: {sql}"


@config.trace_operation(
    "day4_read_only_sql_guardrail",
    capture_input=False,
    capture_output=False,
)
def enforce_read_only_sql(tool_name: Any, tool_args: Dict[str, Any], tool_context: Any = None) -> Optional[Dict[str, Any]]:
    """Callback to block non-SELECT SQL queries strictly before execution."""
    resolved_name = getattr(tool_name, "name", tool_name)
    if resolved_name == "query_incidents":
        query = tool_args.get("sql", "").strip().upper()
        if not query.startswith("SELECT"):
            # Short-circuit execution by returning an error dictionary
            return {"error": "Security Violation: Only read-only SELECT queries are allowed."}
    return None


def redact_pii_from_prompt(prompt_text: str) -> str:
    """Utility function to strip email addresses and phone numbers."""
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"

    redacted = re.sub(email_pattern, "[REDACTED_EMAIL]", prompt_text)
    redacted = re.sub(phone_pattern, "[REDACTED_PHONE]", redacted)
    return redacted


@config.trace_operation(
    "day4_pii_redaction_guardrail",
    capture_input=False,
    capture_output=False,
)
def before_model_guardrail(agent_name: Any, prompt: Any) -> Optional[str]:
    """Redact direct prompt strings and ADK LlmRequest message contents."""
    if isinstance(prompt, str):
        return redact_pii_from_prompt(prompt)

    # ADK calls this with (CallbackContext, LlmRequest); redact user/model text
    # in place and return None so the guarded request proceeds normally.
    contents = getattr(prompt, "contents", None)
    if contents is not None:
        for content in contents:
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    part.text = redact_pii_from_prompt(text)
    return None


# Explicit agent binding with callbacks passed into constructor
guarded_agent = LlmAgent(
    name="guarded_db_agent",
    model=get_adk_model(),
    instruction="Execute database lookups based on user requests.",
    tools=[query_incidents],
    before_tool_callback=enforce_read_only_sql,
    before_model_callback=before_model_guardrail,
)


if __name__ == "__main__":
    print("PII Redaction Test:", before_model_guardrail("guarded_db_agent", "Contact dev@company.com or 555-123-4567"))
    print("Blocked SQL Test:", enforce_read_only_sql("query_incidents", {"sql": "DROP TABLE incidents"}))
    print("Allowed SQL Test:", enforce_read_only_sql("query_incidents", {"sql": "SELECT * FROM incidents"}))
    config.flush_traces()
