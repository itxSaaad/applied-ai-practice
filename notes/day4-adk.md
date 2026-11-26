# Day 4 — Google ADK, MCP, and A2A

Day 4 demonstrates deterministic agent workflows, parallel evidence gathering,
callbacks, scoped filesystem tools, and agent-to-agent communication.

## Main entry points

| Module | Purpose |
| --- | --- |
| `day4_adk/model_config.py` | Selects the configured Ollama or Gemini model. |
| `day4_adk/agent.py` | ADK root agent and telemetry shutdown registration. |
| `day4_adk/pattern1_pipeline.py` | Sequential incident pipeline with parallel evidence gathering. |
| `day4_adk/pattern2_critique_loop.py` | Bounded critique/refinement loop. |
| `day4_adk/pattern3_guardrail_callbacks.py` | Read-only incident query and before-model/tool guardrails. |
| `day4_adk/pattern4_mcp_runbooks.py` | Read-only MCP access limited to `data/runbooks/`. |
| `day4_adk/pattern5_a2a_server.py` | A2A task/discovery server. |
| `day4_adk/pattern6_supervisor.py` | Researcher/analyst/writer supervisor. |

The database is reproducibly seeded from `data/seed/sample_incidents.json`.
Runtime SQLite files live under `artifacts/db/`; trajectory cases are stored
in `evals/datasets/day4_incident_pipeline.evalset.json`.

## Run

```shell
uv run python -m data.seed.seed_incidents_db
uv run python -m day4_adk.pattern1_pipeline
uv run python -m day4_adk.pattern2_critique_loop
uv run python -m day4_adk.pattern3_guardrail_callbacks
uv run python -m day4_adk.pattern4_mcp_runbooks
uv run python -m day4_adk.pattern6_supervisor
uv run adk eval day4_adk evals/datasets/day4_incident_pipeline.evalset.json
```

The ADK pipeline and trajectory evaluation need a reachable configured model.
MCP also needs Node.js/npm. The filesystem server is read-only and scoped to
the runbook directory. `OLLAMA_REQUEST_TIMEOUT_SECONDS` configures each local
ADK model request (default 300 seconds); `ADK_AGENT_RUN_TIMEOUT_SECONDS`
configures whole workflows (default 900 seconds). These defaults were raised
after the local profile measured a p95 response latency near 180 seconds.

Run the A2A server and client in separate terminals:

```shell
uv run python -m day4_adk.pattern5_a2a_server
uv run python -m day4_adk.test_a2a_client
```

## Checks

`evals/test_day4_guardrails_offline.py` and
`evals/test_day4_adk_evals.py` cover deterministic callbacks and shutdown
contracts. `evals/test_trajectories.py` is an opt-in model-backed evaluation.
For framework trade-offs, see [Day 4 ADK vs. LangGraph](day4-adk-vs-langgraph.md).
