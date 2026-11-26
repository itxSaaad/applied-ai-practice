# Architecture decisions

## Context
The course progresses from single model calls to stateful, observable agent
systems. Each day adds a capability rather than replacing the earlier
measurement and validation layers.

## Decisions

1. **Day 1 — measure before tuning.** Provider calls are profiled for TTFT,
   total latency, and token usage. Provider identity and model name are trace
   metadata; measurements are generated under `artifacts/benchmarks/`, never
   checked in as if they were a stable baseline.
2. **Day 2 — use composable, typed application boundaries.** LangChain runnables
   and LCEL compose model and tool steps. Pydantic validates extracted tickets,
   while `utils.sqlite` centralizes read-only query validation/execution.
   A model-call budget constrains cost.
3. **Day 3 — make control flow explicit.** LangGraph `StateGraph` represents
   routing, retries, and grounding checks. SQLite checkpointing makes human
   approval resumable across process restarts; side effects use a stable thread
   identifier to remain idempotent.
4. **Day 4 — separate workflow orchestration from tool boundaries.** ADK
   sequential and parallel agents demonstrate deterministic orchestration,
   `output_key` passes explicit findings, and MCP/A2A provide bounded
   interoperability. Filesystem and SQL tools are restricted to local fixtures
   and read-only access.
5. **Day 5 — make observability and evaluation shared infrastructure.**
   `utils.logging` owns structured JSON formatting, module loggers, and PII /
   credential redaction. `utils.telemetry` owns callbacks, trace metadata,
   operation spans, metrics, and optional ADK OpenTelemetry. Versioned
   evaluation datasets separate deterministic checks from provider-backed
   trajectories and human-reviewed judge calibration.

## Cross-cutting boundaries

- `config.py` is the source of truth for repository paths and environment-backed
  settings. It loads the repository `.env` on import and configures SDK aliases;
  application modules do not read process environment variables directly.
- Shared implementation lives in `utils/`: `sqlite.py` guards database access,
  `logging.py` formats/redacts application events, and `telemetry.py` records
  trace context and measurements. `config` re-exports these helpers for
  compatibility; new helpers should be added to the relevant utility module.
- Runtime modules use `get_logger(__name__)` so logger names match their
  importable module paths. Avoid per-file `basicConfig`, bespoke JSON
  formatting, and duplicate provider initialization.
- Tests and command-line examples use the same path/configuration helpers as
  the application. Provider-dependent integration tests stay explicitly
  opt-in; offline gates must not need keys or network access.

## Runtime storage

`data/` contains versioned fixtures and runbooks. `artifacts/` contains
reproducible local databases, checkpoint state, Chroma indexes, generated
benchmarks, evaluation reports, and application logs. No runtime database,
vector-store, or generated measurement is a source-of-truth input to a test.

## Consequences

- Offline unit and prompt-contract checks run without model keys, network access,
  or Ollama.
- Local `.env` settings are loaded by `config`; shell-provided values take
  precedence. CI injects secrets through its platform and Python still reads
  them only through `config`.
- Provider-backed measurements are opt-in and must record the model and
  configuration before comparison.
- Local traces are not evidence of delivery to a remote dashboard; verify
  provider ingestion separately.
