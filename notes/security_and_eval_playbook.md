# Security and evaluation playbook

## OWASP LLM application checks

- **LLM01 — prompt injection:** Treat retrieved content as data, wrap it in
  escaped `<retrieved_context>` delimiters, and exercise the committed poisoned
  runbook. Do not execute instructions embedded in retrieved documents. The
  deterministic test checks the boundary; only a configured model drill
  measures model obedience.
- **LLM02 — sensitive information disclosure:** Redact email addresses, phone
  numbers, and credential-shaped values before JSON logs are written. Keep
  credentials in an untracked `.env`; never print or attach that file to traces.
- **LLM06 — excessive agency:** Use a single read-only `SELECT` for incident
  queries, bound model calls, cap graph recursion, and put human approval before
  simulated outbound actions. Keep filesystem MCP access read-only and scoped
  to `data/runbooks/`.
- Use `utils.logging.get_logger(__name__)` for application module loggers.
  Structured formatting and redaction are centralized; do not log raw
  credentials, retrieved secrets, or customer data.
- Keep environment reads in `config.py`. Local `.env` is loaded automatically;
  CI secrets are injected by the CI platform and then consumed through `config`.

## Judge calibration protocol

1. Collect at least 20 real agent outputs with their input and ground-truth
   references.
2. Have a human reviewer label each output with a binary score; record the
   reviewer and review date outside the prompt or trace payload.
3. Set `labels_reviewed` to `true` only after all labels are checked.
4. Run `python -m day5_observability.pattern3_evaluators --calibrate`; the judge must reach
   at least 80% agreement before its score is used as a quality gate.
5. If agreement is below threshold, revise the judge prompt and repeat with the
   same reviewed examples. Do not tune against an unreviewed fixture.

The committed 20-row calibration dataset is explicitly provisional and cannot
be used to claim human agreement until it is reviewed.

## Cost and tracing

- Run offline checks and local Ollama experiments without remote tracing when
  privacy or cost requires it.
- Enable LangSmith only for interactive debugging or a bounded evaluation run;
  batch runs should use a deliberate sample size to stay within account limits.
- Langfuse callback and ADK OTLP export activate only when both Langfuse keys
  are set. Flush queued spans before a short-lived process exits.
- Record provider, model, course day, source file, prompt version, run ID, and
  available token/latency metrics. Avoid real customer data and credentials in
  model prompts, logs, or dashboards.
- A local JSONL record proves local logging only. Confirm successful provider
  ingestion in the configured dashboard before reporting remote trace coverage.

## Provider-backed evaluations

| Evaluation | Model/service requirement | Credentials |
| --- | --- | --- |
| `evals/test_trajectories.py::test_adk_incident_pipeline_trajectory` | Runs `AgentEvaluator.evaluate` over the Day 4 incident dataset. Gemini is selected by default; `USE_LOCAL_LLM=true` selects the configured Ollama model. | Gemini needs `GOOGLE_API_KEY`; Ollama needs a reachable server and pulled model. |
| `day5_observability.pattern3_evaluators --calibrate` | Runs the LLM judge over 20 reviewed examples and enforces at least 80% agreement. It uses the configured local or cloud judge model. | Same model requirement as above. The dataset must have `labels_reviewed=true`; the committed labels are provisional. |
| `day5_observability.eval_experiment` | Runs named targets against an existing LangSmith dataset with factual, citation, trajectory, and latency evaluators. | Needs `LANGSMITH_API_KEY`, tracing enabled, and a dataset; the factual judge also needs the selected model service. |
| Langfuse callback / ADK OTLP delivery | Exports spans to the configured Langfuse project. | Needs both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`; this is optional for the evaluation assertions. |

The optional GitHub Actions provider job runs the ADK trajectory test and
defaults to Gemini, so its `GOOGLE_API_KEY` GitHub secret must be configured
separately from a developer's local `.env`. The all-days `run_all.py` runner
clears provider credentials in its offline child processes.
