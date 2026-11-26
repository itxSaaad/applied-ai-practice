"""Build the reproducible local incidents database from versioned JSON fixtures."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import config

SEED_FILE = config.SEED_DIR / "sample_incidents.json"
DATABASE = Path(config.INCIDENTS_DB_PATH)
REQUIRED_FIELDS = (
    "incident_id",
    "service",
    "severity",
    "status",
    "title",
    "created_at",
)


def load_seed_rows(seed_file: Path = SEED_FILE) -> list[dict[str, str]]:
    """Load and validate seed rows before opening or modifying the database."""
    data: Any = json.loads(seed_file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Incident seed data must be a non-empty JSON list: {seed_file}")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Incident seed row {index} must be a JSON object.")
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Incident seed row {index} is missing: {', '.join(missing)}")
        normalized = {
            field: row[field].strip() if isinstance(row[field], str) else row[field] for field in REQUIRED_FIELDS
        }
        if any(not isinstance(value, str) or not value.strip() for value in normalized.values()):
            raise ValueError(f"Incident seed row {index} contains an empty or non-string value.")
        incident_id = normalized["incident_id"]
        if incident_id in seen_ids:
            raise ValueError(f"Duplicate incident_id in seed data: {incident_id}")
        seen_ids.add(incident_id)
        try:
            created_at = datetime.fromisoformat(normalized["created_at"].replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                raise ValueError("Timestamp must include a UTC offset.")
        except ValueError as error:
            raise ValueError(f"Incident seed row {index} has an invalid created_at timestamp.") from error
        rows.append(normalized)
    return rows


def seed_database(
    database: Path = DATABASE,
    seed_file: Path = SEED_FILE,
) -> int:
    """Create or deterministically refresh the managed incidents fixture table."""
    rows = load_seed_rows(seed_file)
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database, timeout=10) as connection:
        connection.execute("PRAGMA busy_timeout = 10000")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(incidents)").fetchall()}
        expected_columns = set(REQUIRED_FIELDS)
        if columns and columns != expected_columns:
            connection.execute("DROP TABLE incidents")

        connection.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                service TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
        connection.execute("DELETE FROM incidents")
        connection.executemany(
            """
            INSERT INTO incidents
                (incident_id, service, severity, status, title, created_at)
            VALUES
                (:incident_id, :service, :severity, :status, :title, :created_at)
            """,
            rows,
        )
    config.log_metric(
        "incident_seed_rows",
        len(rows),
        _source_file=str(Path(__file__)),
        table="incidents",
    )
    return len(rows)


if __name__ == "__main__":
    print(f"Seeded {seed_database()} incidents into {DATABASE}")
