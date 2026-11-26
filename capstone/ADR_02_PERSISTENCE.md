# ADR 02: Durable Checkpointing — PostgreSQL Primary & SQLite Local Fallback

## Context
OpsPilot's Writer Agent pauses via `interrupt()` for human approval before creating Jira tickets or posting Slack updates. State must survive process/container restarts, and resuming must never duplicate side effects.

## Decision
- Use **`PostgresSaver`** when `POSTGRES_URI` is configured (Docker Compose production/local-stack mode).
- Fall back automatically to **`SqliteSaver`** (`artifacts/db/opspilot_checkpoints.sqlite`) when running offline or without Docker.
- Key all Jira/Slack JSON writes by `thread_id` in `capstone/writer_agent.py` so `Command(resume=...)` executes side effects at most once.

## Consequences
- Zero external infrastructure required to run offline tests (`pytest`).
- Identical resume semantics across Postgres and SQLite checkpointers.
