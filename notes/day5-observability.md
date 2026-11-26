# Day 5 — observability and evaluation

Day 5 centralizes logs, provider traces, run metadata, and evaluation helpers;
it also adds deterministic injection checks and reviewed-label judge
calibration.

## Main entry points

| Module | Purpose |
| --- | --- |
| `utils/logging.py` | Structured JSONL logging and PII/credential redaction. |
| `utils/telemetry.py` | LangSmith/Langfuse clients, provider status, trace metadata, metrics, and flush. |
| `day5_observability/pattern1_dual_observability.py` | ADK OpenTelemetry-to-Langfuse setup and flush. |
| `day5_observability/pattern2_prompt_management.py` | Versioned prompt lookup with local fallback. |
| `day5_observability/pattern3_evaluators.py` | Quality evaluators, dataset runner, and judge calibration. |
| `day5_observability/pattern4_security_guardrails.py` | Untrusted-context delimiting, PII handling, and safe tools. |
| `day5_observability/pattern5_ci_gate.py` | Offline prompt-contract and source-retrieval gate. |
| `day5_observability/injection_drill.py` | Before/after injection canary runner. |
| `day5_observability/eval_experiment.py` | Named target evaluation over a LangSmith dataset. |

Use the shared `get_logger(__name__)` and telemetry helpers; do not configure
providers or format JSON logs separately in individual scripts. Call
`trace_provider_status()` to inspect enabled providers and missing credential
names. `flush_traces()` attempts each enabled Langfuse/LangSmith provider even
if another fails, logs individual outcomes, and raises aggregated failures.
`flush_telemetry()` also flushes the ADK OpenTelemetry provider.

## Run

```shell
uv run python -m day5_observability.tracing_setup
uv run python -m day5_observability.pattern2_prompt_management
uv run python -m day5_observability.pattern3_evaluators --calibrate
uv run python -m day5_observability.pattern4_security_guardrails
uv run python -m day5_observability.pattern5_ci_gate --limit 10
```

The offline gate needs no provider credentials. Langfuse requires both
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`; LangSmith tracing requires
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`. Credentials are read through
`config`, never printed. The 20 judge labels are provisional: a human must
review all examples and set `labels_reviewed` to `true` before calibration.
An offline fake-model injection result is not evidence of real-model resistance.

## Checks and measurements

`evals/test_day5_observability.py`, `evals/test_unit_guardrails.py`, and
`evals/test_ci_gate.py` cover local behavior. Provider delivery must be
confirmed in the provider dashboard; a local JSONL record only proves local
logging. See [security and evaluation](security_and_eval_playbook.md) and
[benchmark findings](benchmark_findings.md).
