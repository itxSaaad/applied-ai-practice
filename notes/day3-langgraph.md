# Day 3 — LangGraph

Day 3 makes routing, retries, grounding checks, and human approval explicit in
graphs. SQLite checkpointing preserves resumable state between invocations.

## Main entry points

| Module | Purpose |
| --- | --- |
| `day3_langgraph/pattern1_crag.py` | Corrective RAG graph with grading, rewriting, and grounded answers. |
| `day3_langgraph/pattern2_hitl.py` | Durable approval flow with SQLite checkpoints. |
| `day3_langgraph/pattern3_map_reduce.py` | Map/reduce summarization graph. |
| `day3_langgraph/pattern4_supervisor.py` | Research/analysis/writer supervisor. |
| `day3_langgraph/pattern5_functional_api.py` | Functional API version of a compact graph workflow. |
| `day3_langgraph/benchmark_crag.py` | Retrieval, citation, and latency comparison. |
| `day3_langgraph/benchmark_map_reduce.py` | Sequential-versus-parallel map/reduce comparison. |

Incident and supervisor examples use tracked synthetic records in `data/`.
Checkpoints and simulated approved actions are written under `artifacts/db/`.

## Run

```shell
uv run python -m day3_langgraph.pattern1_crag
uv run python -m day3_langgraph.benchmark_crag
uv run python -m day3_langgraph.benchmark_map_reduce --compare-sequential
uv run python -m day3_langgraph.pattern2_hitl
uv run python -m day3_langgraph.pattern3_map_reduce
uv run python -m day3_langgraph.pattern4_supervisor
uv run python -m day3_langgraph.pattern5_functional_api
```

The CRAG benchmark compares expected-source citation rate and mean latency over
the shared QA cases. Citation presence is not a factual-accuracy judgment;
review generated answers independently.

## Checks

`day3_langgraph/test_graph.py` and `evals/test_unit_guardrails.py` cover graph
behavior and checkpoint persistence without requiring a live model.
