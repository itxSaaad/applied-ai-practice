import json
from pathlib import Path

from langchain_core.tools import tool

import config
from data.seed.seed_incidents_db import seed_database
from utils.logging import get_logger
from utils.sqlite import fetch_read_only_rows, is_single_select

DB_PATH = Path(config.INCIDENTS_DB_PATH)
logger = get_logger(__name__)


@tool
@config.trace_operation(
    "day2_read_only_sqlite_query",
    capture_input=False,
    capture_output=False,
)
def query_read_only_sqlite(query: str) -> str:
    """Executes a strictly READ-ONLY SQL query against incidents.db.

    The incidents table has columns incident_id, service, severity, status,
    title, and created_at. Seeded severity and status values are lowercase;
    use case-insensitive comparisons when the user's casing may differ.
    Queries must start with SELECT; writes and DDL are rejected.
    """
    if not is_single_select(query):
        logger.warning(
            "Blocked non-read-only SQLite query.",
            extra={
                "tool_name": "query_read_only_sqlite",
                "guardrail_triggered": True,
                "day_module": "day2_langchain.pattern2_safe_tool",
            },
        )
        raise ValueError("Write and non-SELECT statements are forbidden.")

    if not DB_PATH.exists():
        seed_database(database=DB_PATH)
    return json.dumps(fetch_read_only_rows(DB_PATH, query), indent=2)


if __name__ == "__main__":
    print("1. Seeding sample incidents database...")
    seed_database()

    print("\n2. Executing valid read-only SELECT query:")
    valid_query = "SELECT incident_id, title, severity, status FROM incidents " "WHERE severity = 'critical'"
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
