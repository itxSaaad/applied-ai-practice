"""Shared SQLite helpers for the course's read-only data tools."""

import re
import sqlite3
from pathlib import Path
from typing import Any

_SINGLE_SELECT = re.compile(
    r"\s*SELECT\b[^;]*(?:;)?\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


def is_single_select(statement: object) -> bool:
    """Return whether input is exactly one SELECT statement."""
    return isinstance(statement, str) and _SINGLE_SELECT.fullmatch(statement) is not None


def fetch_read_only_rows(database: Path, statement: str) -> list[dict[str, Any]]:
    """Execute one SELECT through a SQLite read-only URI and return row mappings."""
    if not is_single_select(statement):
        raise ValueError("Only one read-only SELECT statement is allowed.")

    database_uri = f"{Path(database).resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        cursor = connection.execute(statement)
        columns = [column[0] for column in cursor.description or ()]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()
