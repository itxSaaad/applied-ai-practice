"""Security guardrails for OpsPilot reusing Day 4 & Day 5 defensive helpers."""

from __future__ import annotations

import re

from day4_adk.pattern3_guardrail_callbacks import enforce_read_only_sql
from day5_observability.pattern4_security_guardrails import wrap_untrusted_context
from utils.logging import redact_sensitive_text
from utils.sqlite import is_single_select

INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system\s+prompt|api\s+key|secret)", re.IGNORECASE),
    re.compile(r"</retrieved_context>\s*<system>", re.IGNORECASE),
    re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\s+TABLE\b", re.IGNORECASE),
)


def detect_prompt_injection(text: str) -> list[str]:
    """Return matched prompt-injection pattern strings if any are present."""
    if not isinstance(text, str):
        return []
    return [pattern.pattern for pattern in INJECTION_PATTERNS if pattern.search(text)]


def wrap_untrusted_text(source: str, content: str) -> str:
    """Tag untrusted retrieved content using Day 5's XML wrapper and flag injection signals."""
    flags = detect_prompt_injection(content)
    wrapped = wrap_untrusted_context(f"Inspect source {source}.", content)
    if flags:
        return f"[INJECTION_SIGNALS_DETECTED=true source={source}]\n{wrapped}"
    return wrapped


def redact_pii(text: str) -> str:
    """Mask email addresses, phone numbers, and secret tokens."""
    return redact_sensitive_text(text)


def validate_sql_query(sql: str) -> str:
    """Enforce single read-only SELECT statement allow-list via Day 4 & shared SQLite helper."""
    blocked = enforce_read_only_sql("query_incidents", {"sql": sql})
    if blocked is not None or not is_single_select(sql):
        raise ValueError("SQL guardrail blocked non-SELECT or multi-statement query.")
    return sql.strip()
