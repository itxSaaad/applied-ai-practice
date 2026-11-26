# Applied AI & Agentic Engineering Practice

Course 1 implementation for milestones Days 1–5. Dependencies are declared in
`pyproject.toml` and resolved in `uv.lock`. `uv` creates and manages the
project-local `.venv`; do not install the course dependencies into system
Python.

## Setup (all operating systems)

Requirements: `uv`. The checked-in `.python-version` selects Python 3.12 and
`uv` installs it if needed. Ollama is needed only for local-model and embedding
exercises. Node.js/npm is needed only for the Day 4 MCP filesystem exercise.

From the repository root, synchronize dependencies into `.venv`:

```shell
uv sync --extra dev
```

Activate the local environment when using bare `python` commands:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```shell
# macOS/Linux
source .venv/bin/activate
```

Alternatively, prefix commands with `uv run`; it uses the project `.venv`.
`uv lock --check` verifies dependency declarations against the committed lock.
The lock uses compatible Langfuse 5.2 and ADK 2.14 SDKs: the course's 5.1/2.13
combination has conflicting OpenTelemetry requirements (Langfuse requires
OpenTelemetry >=1.45 while ADK 2.13 caps it at 1.44).

Create `.env` only if it does not already exist:

```shell
uv run python -c "from pathlib import Path; p=Path('.env'); p.exists() or p.write_text(Path('.env.example').read_text(encoding='utf-8'), encoding='utf-8')"
```

`config.py` loads the repository `.env` automatically when imported (existing
shell environment values take precedence). Application modules read settings
through `config`; it is the only project Python module that reads or updates
process environment variables, including SDK-compatible aliases.

Set only the credentials/providers needed for the exercise. For local Ollama:

```ini
USE_LOCAL_LLM=true
LOCAL_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
```

Then fetch the chat and embedding models:

```shell
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Cloud exercises require a valid `GOOGLE_API_KEY`. Traces are sent to LangSmith
only when `LANGSMITH_TRACING=true` and its API key are configured; Langfuse
requires both keys. Blank/missing observability credentials keep local runs
available, but cannot create provider dashboard traces. Do not put secrets or
real customer data in course runs.

## Run and inspect the course

Run all offline acceptance checks with the project-local environment, without
API keys or model downloads:

```shell
uv run python run_all.py
```

Select one day with `--day day3`, or several by repeating `--day`. Each suite
records a JSONL event with date, day, source file, run ID, status, exit code,
duration, and output-log path under `artifacts/logs/YYYY-MM-DD/`. Structured
application events append to `artifacts/logs/app.jsonl`. The runner continues
through selected suites and exits non-zero if any fail. Runtime logs,
databases, checkpoints, vector stores, and benchmark outputs are ignored by Git.

Run the normal pytest suite (excluding live provider integrations):

```shell
uv run python -m pytest -m "not integration"
```

The `Course 1 | Quality and Evaluation` GitHub Actions pipeline runs lock,
compile, offline test, and evaluation gates on Windows, macOS, and Linux.
Provider-backed tests are marked `integration` and are kept out of pull-request
checks; run them only with explicit credentials and services.
The optional live provider workflow runs
`evals/test_trajectories.py::test_adk_incident_pipeline_trajectory` against the
Day 4 ADK agent. It uses Gemini by default and needs `GOOGLE_API_KEY`; when
`USE_LOCAL_LLM=true`, it uses the configured Ollama model and requires a
reachable Ollama server/model instead. LangSmith and Langfuse keys enable
dashboard tracing but are not required for the trajectory assertion itself.

## Structured logs, metrics, and traces

`config.py` owns paths and environment-backed settings. `utils/logging.py`
handles structured JSON logs, PII redaction, and logger setup. `utils/telemetry.py`
owns tracing clients, run metadata, operation spans, metrics, and ADK OpenTelemetry
bootstrap. The corresponding helpers remain re-exported from `config` for
compatibility; new code can import them from their focused utility modules.
`utils/sqlite.py` provides the shared read-only query validator and execution
helper used by the Day 2, Day 4, and Day 5 tools. ADK OTLP export is initialized
once and flushed explicitly before process exit.

Each run gets a unique run ID (the all-days runner sets one automatically) and a
consistent `LANGSMITH_PROJECT`. Record provider, model, prompt version, token
usage, latency, and evaluation scores where available; do not capture
credentials or real customer data. Remote dashboard delivery requires valid
keys, network access, and provider-side ingestion; local logs alone do not prove
remote delivery.

## Study notes

The [notes index](notes/README.md) links to the [repository map](notes/repository_map.md),
concise [Day 1](notes/day1-foundations.md) through [Day 5](notes/day5-observability.md)
guides, architecture decisions, benchmark findings, and security/evaluation references.

## Milestones and files

| Day | Main implementation | Tracked sample data and acceptance checks |
| --- | --- | --- |
| 1 — Foundations | `day1_foundations/` | Project corpus (currently 609 chunks); profile and sampling outputs are written to `artifacts/benchmarks/`. |
| 2 — LangChain | `day2_langchain/` | 20 support tickets, 25 retrieval QA pairs, five labeled extraction examples; RAG persists runbook embeddings under `artifacts/chroma_db/`. |
| 3 — LangGraph | `day3_langgraph/` | 20 incident-log records; CRAG and HITL checkpoint databases live in `artifacts/db/`. |
| 4 — Google ADK, MCP, A2A | `day4_adk/` | `data/seed/sample_incidents.json` deterministically seeds six records; the ADK trajectory dataset has ten cases. MCP needs Node.js/npm; trajectory evaluation needs a model. |
| 5 — Observability and evaluation | `day5_observability/` | Shared telemetry, six injection fixtures, ten-case offline gate, and a 20-row calibration example set explicitly awaiting human review. |

Static, versioned CSV/JSON/Markdown fixtures live in `data/` and
`evals/datasets/`. Generated CSVs, reports, SQLite files, Chroma data, and logs
live only in the ignored `artifacts/` tree.
