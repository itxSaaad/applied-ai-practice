# OpsPilot — Multi-Agent Incident Response Copilot (Capstone)

Production-grade multi-agent incident response system built with LangGraph v1, Google ADK v2 (A2A), FastAPI SSE, Postgres/SQLite checkpointing, ChromaDB, and Langfuse/LangSmith observability.

## Architecture

- **LangGraph Supervisor (`capstone/ops_pilot_supervisor.py`)**: Routes queries across Knowledge, Data, ADK A2A Analyst, and Writer agents.
- **Knowledge Agent (`capstone/knowledge_agent.py`)**: Corrective RAG subgraph (`retrieve -> grade -> rewrite -> generate`) with runbook citations.
- **Data Agent (`capstone/data_agent.py`)**: Read-only SQLite query subgraph over `services` (12), `deploys` (300), and `incidents` (120).
- **ADK Incident Analyst (`capstone/incident_analyst_a2a.py`)**: Google ADK `LlmAgent` served via A2A (`/.well-known/agent.json`, `/a2a/tasks`) with graceful fallback when unreachable.
- **Writer Agent & Approval Gate (`capstone/writer_agent.py`)**: Drafts postmortems and pauses via `interrupt()` before idempotently writing Jira/Slack records.
- **FastAPI SSE API (`capstone/main.py`)**: `POST /chat` (SSE streaming), `POST /threads/{thread_id}/resume`, `GET /health`.
- **Guardrails (`capstone/guardrails.py`)**: Prompt-injection detection/escaping, PII masking, and single-SELECT SQL allow-list.

## Quickstart (local)

```powershell
# 1. Seed SQLite (12 services, 300 deploys, 120 incidents) & index runbooks
uv run python -m capstone.seed

# 2. Run OpsPilot test suite & 30-case evaluation gate
uv run python -m pytest -q evals/test_capstone.py
uv run python -m evals.eval_runner --min-pass-rate 0.8

# 3. Start ADK A2A service + FastAPI SSE server (two terminals)
uv run uvicorn capstone.incident_analyst_a2a:app --port 8002
uv run uvicorn capstone.main:app --port 8000
```

## Docker Compose (Postgres + Chroma + ADK + API)

```powershell
docker compose -f capstone/docker-compose.yml up --build
```
