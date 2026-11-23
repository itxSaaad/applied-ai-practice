import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafeTool")

DB_PATH = Path(__file__).resolve().parent.parent / "incidents.db"


def seed_database():
    """Initializes and seeds the sample incidents table in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.execute("DROP TABLE IF EXISTS incidents")
        conn.execute(
            """
            CREATE TABLE incidents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO incidents (title, severity, status) VALUES (?, ?, ?)",
            [
                ("Production checkout API 504 gateway timeout", "Critical", "Investigating"),
                ("High latency on user authentication service", "High", "Identified"),
                ("Periodic worker memory pressure alert", "Medium", "Resolved"),
            ],
        )
    conn.close()


@tool
def query_read_only_sqlite(query: str) -> str:
    """Executes a strictly READ-ONLY SQL query against incidents.db.
    Enforces that queries must start with SELECT; write or DDL operations raise ValueError.
    """
    normalized = query.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Write and non-SELECT statements are forbidden.")

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        result = [dict(zip(columns, row)) for row in rows]
        return json.dumps(result, indent=2)
    finally:
        conn.close()


if __name__ == "__main__":
    print("1. Seeding sample incidents database...")
    seed_database()

    print("\n2. Executing valid read-only SELECT query:")
    valid_query = "SELECT id, title, severity, status FROM incidents WHERE severity = 'Critical'"
    select_result = query_read_only_sqlite.invoke({"query": valid_query})
    print(select_result)

    print("\n3. Testing programmatic guardrail on DROP TABLE statement:")
    malicious_query = "DROP TABLE incidents"
    try:
        query_read_only_sqlite.invoke({"query": malicious_query})
        print("ERROR: Guardrail failed to block write query!")
    except ValueError as e:
        print(f"SUCCESS: Blocked dangerous query as expected: {e}")

    config.flush_traces()
