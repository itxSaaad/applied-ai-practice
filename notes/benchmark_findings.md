# Benchmark findings

Measurements below were collected on 2026-11-26 using this Windows workspace.
They describe this machine and local model setup, not general provider
performance. Raw outputs are under ignored `artifacts/benchmarks/`.

## Day 1 — local model profile

`day1_foundations.profile --local --n 5` completed with Ollama `qwen3:8b`.
Provider usage metadata supplied token counts for every streamed response.

| Metric | p50 | p95 |
| --- | ---: | ---: |
| Time to first token | 96,325.86 ms | 106,746.68 ms |
| Total response latency | 171,781.04 ms | 179,934.22 ms |

Each run used 26 input and 729 output tokens. Run ID:
`537c19dfda1446a895c3e504f4f21a19`. The long TTFT makes multi-call local
agent workflows slow; their configured runtime limits must allow for this.
This profile found and fixed a streaming-accounting bug: LangChain can append
an empty final chunk after the provider's usage chunk, so usage must be
collected throughout the stream.

## Day 1 — vector search

`day1_foundations.benchmark_search` indexed 586 chunks from 44 source files
with Ollama `nomic-embed-text`.

| Measurement | Result |
| --- | ---: |
| Exact NumPy top-5 calculation | 31.83 ms |
| Chroma indexing plus query | 474.86 ms |
| Chroma top-5 overlap with exact NumPy top-5 | 100% |

This is one query and measures agreement with exact vector ranking, not
relevance judged by a human. Run ID:
`7e375a70e48b465d9bdc447a470dc1e2`.

## Day 1 — Gemini Cloud model profile & temperature

`day1_foundations.profile --cloud --n 5` completed with `gemini-3.5-flash-lite`
under the client-side 15 RPM rate limiter (`utils/rate_limit.py`), run ID
`94e037f757d44f6c88efb24acfd3d583`:

| Metric | p50 | p95 |
| --- | ---: | ---: |
| Time to first token | 2,259.90 ms | 2,964.73 ms |
| Total response latency | 3,156.04 ms | 4,016.10 ms |

Streaming chunks from `langchain_google_genai` reported `0.0` mean input
tokens and `6.0` mean output tokens (`profile.py` was updated to ignore `(0, 0)`
placeholder usage chunks).

`day1_foundations.benchmark_temp --iterations 5` completed with
`gemini-3.5-flash-lite` (run ID `67997c7d46154adfbbcf44a9e42f3834`, saved to
`artifacts/benchmarks/temperature_results.csv`):

| Temperature | Valid JSON Rate (%) | Unique Variations (of 5) |
| ---: | ---: | ---: |
| 0.0 | 100.0% | 3 |
| 0.7 | 100.0% | 4 |
| 1.2 | 100.0% | 2 |

Note: `langchain_google_genai` warns that `gemini-3.5-flash-lite` uses fixed
sampling defaults and ignores the `temperature` parameter.

## Day 2 — runbook chunking & cloud extraction

`day2_langchain.chunking_study` evaluated expected-source retrieval at `k=4`
for 25 QA cases, using Ollama `nomic-embed-text`.

| Chunk size | Overlap | Chunks | Expected-source hit@4 | Mean query latency |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 0 | 39 | 100% | 47.24 ms |
| 250 | 100 | 41 | 96% | 44.33 ms |
| 500 | 0 | 17 | 100% | 42.65 ms |
| 500 | 100 | 17 | 100% | 49.80 ms |
| 1000 | 0 | 12 | 96% | 42.38 ms |
| 1000 | 100 | 12 | 96% | 42.69 ms |

These are source-hit rates for this small fixture, not answer factuality
scores. Run ID: `2e7eaa37dbb8438e8804d8dfe72ab81e`.

`day2_langchain.batch_extract` completed all 20 tickets against
`gemini-3.5-flash-lite` with `20/20` (`100.0%`) schema-valid outputs against
its 19-ticket threshold (run ID `27e31f67d28b4e618a74878e419b6f15`,
`artifacts/benchmarks/ticket_extraction_results.csv`).

## Day 3 — parallel map/reduce & functional API

`day3_langgraph.benchmark_map_reduce` completed 20 parallel `Send` map calls
plus 1 reduce call against `gemini-3.5-flash-lite` in `70,424.40 ms`
(`70,767.47 ms` total operation duration) under the 15 RPM rate limiter with no
429 failures (run ID `18c4c1a5cf54475191e5b7bcfe729cf8`,
`artifacts/benchmarks/map_reduce_timings.csv`). Sequential baseline
(`--compare-sequential`) was not run to conserve daily RPD quota.
`day3_langgraph.pattern5_functional_api` was updated to lazy-load via
`get_llm()` and completed cleanly against Gemini.

## Day 4 — ADK model-backed workflows

- `day4_adk.pattern1_pipeline` completed in `23,869.92 ms` (run ID
  `2ecaf3c63fe747e096eef8c67e15acf3`) after fixing `TracerProvider` reuse in
  `utils/telemetry.py`, missing-log handling in `read_log`, and schema/error
  handling in `query_incidents`.
- `day4_adk.pattern2_critique_loop` completed in 1 iteration (`PASSED`).
- `day4_adk.pattern4_mcp_runbooks` answered all 3 runbook questions with
  citations (`checkout_504.md`, `recovery_checks.md`, `deploy_rollback.md`) in
  `60,785.17 ms` (run ID `c90754729b314b7eb56ebec3f9cf4550`) after resolving
  `npx.CMD` on Windows via `shutil.which("npx")`.
- `evals/test_trajectories.py::test_adk_incident_pipeline_trajectory` ran all
  10 cases to completion in `323.43 s` without rate-limit errors, but failed
  strict `AgentEvaluator` thresholds (`tool_trajectory_avg_score` `0.0` vs
  `1.0` expected; `response_match_score` `0.17`–`0.34` vs `0.8` expected) due
  to parallel tool ordering/argument variations and ROUGE-1 phrasing differences.

## Day 5 — offline gate & injection drill

The provider-free prompt-contract and source-retrieval gate passed 9/10 cases
(90%) against its 80% threshold. This does not evaluate generated answer
factuality. The latest report is `artifacts/benchmarks/eval_summary.json`;
run ID: `f9d5b2e451404276b8b127926644c61a`.

`day5_observability.injection_drill` completed all 6 cases against
`gemini-3.5-flash-lite` (run ID `73f9d6bdb41a43bf9cff6ce699d0e6b6`,
`artifacts/benchmarks/injection_drill_results.csv`): `0.0%` (`0/6`) obedience
before defense and `0.0%` (`0/6`) obedience after `<retrieved_context>`
wrapping.

## Offline acceptance suite

`uv run python run_all.py` passed all 10 configured offline suites (including
`evals/test_rate_limit.py`) on 2026-11-26. Provider-backed trajectory tests
remain excluded from this runner.

## Local workflow smoke checks

- The Day 2 local RAG example returned a checkout-timeout response citing
  `data/runbooks/checkout_504.md`.
- The Day 2 ten-question scripted agent completed, but its first run answered
  the capitalized `Critical` query incorrectly because seeded severity values
  are lowercase. After adding schema/case guidance, a focused rerun returned
  the expected critical incident. The full ten-question run has not yet been
  repeated with that final guidance.
- Day 3 HITL paused and resumed its SQLite checkpoint, and the CRAG example
  returned an answer citing the HTTP 500 runbook.
- The Day 4 callback example redacted email/phone data, rejected `DROP TABLE`,
  and permitted the read-only `SELECT` check.

## Not measured or not verified

- Local Ollama workflows were not re-run in this pass because the local Ollama
  daemon was unavailable (`WinError 10061` / server start timeout).
- Judge calibration remains blocked until all 20 provisional labels in
  `evals/datasets/day5_judge_calibration.json` are human-reviewed and
  `labels_reviewed` is set to `true`.
- Langfuse and LangSmith client flushes reported `flushed` across cloud runs,
  while ADK OpenTelemetry OTLP HTTP export returned `404 Not Found` against the
  configured `LANGFUSE_OTEL_ENDPOINT`; remote dashboard ingestion has not been
  independently verified.

## Historical captures

These earlier values have no recorded run date or complete provider/hardware
configuration. Keep them separate from the current local profile above.

| Historical model capture | Runs | TTFT p50 / p95 | Total latency p50 / p95 |
| --- | ---: | ---: | ---: |
| Ollama `qwen3:8b` | 5 | 760.72 / 3519.23 ms | 162163.23 / 184498.73 ms |
| Gemini `gemini-3.5-flash-lite` | 20 | 949.18 / 1786.10 ms | 1794.64 / 2539.81 ms |
