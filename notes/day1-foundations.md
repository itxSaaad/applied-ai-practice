# Day 1 — foundations and measurement

Day 1 establishes model-call, latency, token-usage, retrieval, and structured
output baselines before later days add orchestration.

## Main entry points

| Module | Purpose |
| --- | --- |
| `llm.py` | Shared traced chat-model entry point. |
| `day1_foundations/profile.py` | Profiles local/cloud model latency, TTFT, and usage. |
| `day1_foundations/benchmark_search.py` | Measures semantic search over the course corpus. |
| `day1_foundations/benchmark_temp.py` | Compares temperature and structured-output behavior. |
| `day1_foundations/tokenization_comparison.py` | Compares configured model tokenization. |
| `day1_foundations/decision_memo.md` | Captures model-selection and measurement rationale. |

## Run

```shell
uv run python -m day1_foundations.profile --both --n 20
uv run python -m day1_foundations.profile --local --n 20
uv run python -m day1_foundations.benchmark_search
uv run python -m day1_foundations.benchmark_temp --iterations 10
uv run python -m day1_foundations.tokenization_comparison
```

Profile and sampling experiments need a reachable configured model. Semantic
search needs Ollama and `nomic-embed-text`; token-count comparison needs both
the configured local model and Gemini. Generated measurements belong in
`artifacts/benchmarks/`, not source control.

The latest five-run local profile reads token counts from the provider's stream
metadata. The profile keeps usage metadata across all chunks because LangChain
can append a final empty chunk. Current measurements and caveats are in
[benchmark findings](benchmark_findings.md).

## Checks

`evals/test_day1_foundations.py` covers deterministic fixtures and offline
behavior. See [benchmark findings](benchmark_findings.md) for the distinction
between historical records and measurements that still need to be rerun.
