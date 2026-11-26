# Repository map

## Project boundaries

| Location | Responsibility |
| --- | --- |
| `config.py` | Loads root `.env`, owns paths and typed settings, and configures SDK environment aliases. |
| `utils/` | Shared structured logging, telemetry, and read-only SQLite helpers. |
| `day1_foundations/` … `day5_observability/` | Course examples organized by milestone; each day's guide maps its entry points. |
| `data/` and `evals/datasets/` | Tracked synthetic inputs, runbooks, adversarial fixtures, and expected evaluation data. |
| `evals/` | Offline guardrails and quality gates, plus explicitly opt-in provider integrations. |
| `artifacts/` | Ignored runtime logs, databases, vector indexes, benchmarks, and evaluation reports. |
| `run_all.py` | Runs the offline suites in isolated child processes and records a dated run ledger. |
| `.github/workflows/course1-quality-gates.yml` | Cross-platform lock, compile, and offline test pipeline. |

## Shared services

- Use `config` for paths and environment-backed values. It is the only project
  Python module that reads or updates process environment variables.
- Use `utils.logging.get_logger(__name__)` for application events. The shared
  logger writes redacted JSONL records under `artifacts/logs/`.
- Use `utils.telemetry` for LangSmith/Langfuse callbacks, trace metadata,
  operation metrics, provider status, and explicit flushes.
- Use `utils.sqlite` for validated, read-only incident queries rather than
  duplicating SQL checks in individual tools.
- Keep static fixtures in `data/` or `evals/datasets/`; never check in generated
  databases, logs, Chroma indexes, or measurements.

## Environment and validation

The root `pyproject.toml`, `uv.lock`, and `.python-version` define the local
Python environment. Run `uv sync --extra dev`, then use `uv run ...` so commands
resolve through the project-local `.venv`. Copy `.env.example` to `.env` only
when provider settings are needed; never print or commit credential values.

`uv run python run_all.py` executes the offline suite set. Provider-backed
trajectory and judge evaluations are opt-in and have separate model/service
requirements; see [security and evaluation](security_and_eval_playbook.md).

## Day guides

- [Day 1 — foundations and measurement](day1-foundations.md)
- [Day 2 — LangChain](day2-langchain.md)
- [Day 3 — LangGraph](day3-langgraph.md)
- [Day 4 — Google ADK, MCP, and A2A](day4-adk.md)
- [Day 5 — observability and evaluation](day5-observability.md)
