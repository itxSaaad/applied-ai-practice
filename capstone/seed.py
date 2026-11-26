"""Deterministic OpsPilot SQLite seeder (services, deploys, incidents) and knowledge indexer."""

from __future__ import annotations

import random
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

if TYPE_CHECKING:
    from pathlib import Path

OPSPILOT_DB_PATH = config.ARTIFACTS_DIR / "db" / "opspilot.sqlite"
OPSPILOT_CHROMA_DIR = config.ARTIFACTS_DIR / "chroma_opspilot"

SERVICES = [
    ("payments-api", "checkout-squad", "tier-1", "Handles card authorizations and payment intents"),
    ("checkout-service", "checkout-squad", "tier-1", "Orchestrates cart checkout and order placement"),
    ("auth-service", "identity-squad", "tier-1", "Issues and validates JWT sessions and OAuth tokens"),
    ("worker-service", "platform-squad", "tier-2", "Executes async webhooks and invoice jobs"),
    ("search-service", "discovery-squad", "tier-2", "Serves catalog full-text and vector search"),
    ("billing-service", "revenue-squad", "tier-1", "Generates subscriptions and ledger entries"),
    ("notification-api", "comms-squad", "tier-2", "Dispatches email, SMS, and push alerts"),
    ("inventory-api", "fulfillment-squad", "tier-1", "Tracks warehouse SKU reservations"),
    ("shipping-service", "fulfillment-squad", "tier-2", "Computes carrier rates and tracking labels"),
    ("analytics-ingest", "data-squad", "tier-3", "Collects clickstream telemetry events"),
    ("gateway-proxy", "edge-squad", "tier-1", "Routes external API traffic and enforces rate limits"),
    ("fraud-scorer", "risk-squad", "tier-1", "Scores payment transactions for fraud risk"),
]


def seed_opspilot_database(db_path: Path | None = None, *, seed: int = 42) -> dict[str, int]:
    """Create and deterministically populate services (12), deploys (300), and incidents (120)."""
    target = db_path or OPSPILOT_DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    random.Random(seed)
    base_time = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)

    with sqlite3.connect(target) as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS incidents;
            DROP TABLE IF EXISTS deploys;
            DROP TABLE IF EXISTS services;

            CREATE TABLE services (
                service_id TEXT PRIMARY KEY,
                owner_team TEXT NOT NULL,
                tier TEXT NOT NULL,
                description TEXT NOT NULL
            );

            CREATE TABLE deploys (
                deploy_id TEXT PRIMARY KEY,
                service_id TEXT NOT NULL,
                version TEXT NOT NULL,
                deployed_by TEXT NOT NULL,
                status TEXT NOT NULL,
                deployed_at TEXT NOT NULL,
                FOREIGN KEY (service_id) REFERENCES services(service_id)
            );

            CREATE TABLE incidents (
                incident_id TEXT PRIMARY KEY,
                service_id TEXT NOT NULL,
                title TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (service_id) REFERENCES services(service_id)
            );
            """)

        conn.executemany(
            "INSERT INTO services (service_id, owner_team, tier, description) VALUES (?, ?, ?, ?)",
            SERVICES,
        )

        deploys: list[tuple[str, str, str, str, str, str]] = []
        for idx in range(1, 301):
            svc = SERVICES[(idx - 1) % len(SERVICES)][0]
            ts = (base_time + timedelta(hours=idx * 2)).isoformat()
            status = "rolled_back" if idx == 295 else ("failed" if idx % 29 == 0 else "succeeded")
            version = f"v2.{idx // 10}.{idx % 10}"
            if idx == 299:
                svc = "payments-api"
                version = "v2.30.1"
                status = "succeeded"
            deploys.append((f"DEP{idx:04d}", svc, version, f"engineer-{(idx % 8) + 1}", status, ts))

        conn.executemany(
            "INSERT INTO deploys (deploy_id, service_id, version, deployed_by, status, deployed_at) VALUES (?, ?, ?, ?, ?, ?)",
            deploys,
        )

        severities = ["low", "medium", "high", "critical"]
        statuses = ["resolved", "resolved", "mitigated", "open"]
        incidents: list[tuple[str, str, str, str, str, str, str]] = []
        for idx in range(1, 121):
            svc = SERVICES[(idx - 1) % len(SERVICES)][0]
            ts = (base_time + timedelta(hours=idx * 5)).isoformat()
            sev = severities[idx % len(severities)]
            st = statuses[idx % len(statuses)]
            title = f"{svc} elevated error rate #{idx}"
            cause = "Connection pool exhaustion under burst load"
            if idx == 120:
                svc = "payments-api"
                sev = "critical"
                st = "open"
                title = "payments-api p99 latency spike above 2500ms after v2.30.1"
                cause = "Unindexed lookup and DB pool saturation introduced in deploy DEP0299 (v2.30.1)"
            incidents.append((f"INC{idx:04d}", svc, title, sev, st, cause, ts))

        conn.executemany(
            "INSERT INTO incidents (incident_id, service_id, title, severity, status, root_cause, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            incidents,
        )
        conn.commit()

    return {"services": len(SERVICES), "deploys": len(deploys), "incidents": len(incidents)}


def load_runbook_documents() -> list[Document]:
    """Load all tracked runbooks from data/runbooks/ as LangChain Documents."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs: list[Document] = []
    for path in sorted(config.RUNBOOKS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": path.name}))
    return docs


def ingest_knowledge_base() -> int:
    """Index runbook chunks into Chroma when embeddings are available, returning chunk count."""
    docs = load_runbook_documents()
    if not docs:
        return 0
    try:
        from langchain_chroma import Chroma

        from llm import get_embeddings

        embeddings = get_embeddings()
        OPSPILOT_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(OPSPILOT_CHROMA_DIR),
            collection_name="opspilot_runbooks",
        )
    except Exception:
        # Local lexical fallback remains available without requiring Ollama/Gemini embeddings
        pass
    return len(docs)


if __name__ == "__main__":
    counts = seed_opspilot_database()
    chunks = ingest_knowledge_base()
    print(f"Seeded OpsPilot SQLite: {counts}; prepared {chunks} runbook chunks.")
