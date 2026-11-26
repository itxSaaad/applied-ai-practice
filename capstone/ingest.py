"""Ingestion entrypoint for OpsPilot SQLite database and Chroma runbook index."""

from __future__ import annotations

from capstone.seed import (
    OPSPILOT_CHROMA_DIR,
    OPSPILOT_DB_PATH,
    ingest_knowledge_base,
    load_runbook_documents,
    seed_opspilot_database,
)


def main() -> None:
    counts = seed_opspilot_database()
    chunks = ingest_knowledge_base()
    print(f"Ingested OpsPilot SQLite ({counts}) and {chunks} runbook chunks into {OPSPILOT_CHROMA_DIR}.")


__all__ = [
    "OPSPILOT_CHROMA_DIR",
    "OPSPILOT_DB_PATH",
    "ingest_knowledge_base",
    "load_runbook_documents",
    "seed_opspilot_database",
]

if __name__ == "__main__":
    main()
