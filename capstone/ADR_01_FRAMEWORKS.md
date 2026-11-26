# ADR 01: Multi-Agent Frameworks — LangGraph Supervisor + Google ADK over A2A

## Context
OpsPilot requires deterministic stateful orchestration with pause/resume human approval gates alongside cross-framework collaboration with a Google ADK incident analyst agent.

## Decision
- Use **LangGraph (`StateGraph`)** for the central supervisor (`capstone/ops_pilot_supervisor.py`), Corrective RAG (`capstone/knowledge_agent.py`), read-only SQL (`capstone/data_agent.py`), and `interrupt()` approval (`capstone/writer_agent.py`).
- Expose the **Google ADK `incident-analyst`** (`capstone/incident_analyst_a2a.py`) as an independent A2A HTTP service called through `capstone/a2a_client.py`.
- Enforce Day 2 fallbacks (`.with_fallbacks`), Day 4 SQL/PII callbacks, and Day 5 prompt management + dual tracing.

## Consequences
- Fine-grained routing, step budgets (`MAX_SUPERVISOR_STEPS`), and `interrupt()` durability are controlled directly in LangGraph.
- If the ADK A2A service is down, `a2a_client.py` returns a structured degraded response instead of crashing the supervisor.
