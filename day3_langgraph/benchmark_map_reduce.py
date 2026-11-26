"""Measure LangGraph Send fan-out and optionally compare a sequential baseline."""

import argparse
import csv
import json
import time

import config
from day3_langgraph.pattern3_map_reduce import (
    map_reduce_graph,
    reduce_summaries_node,
    summarize_doc_node,
)


@config.trace_operation("day3_map_reduce_timing", flush=True)
def run_benchmark(compare_sequential: bool) -> list[dict[str, str | int | float]]:
    documents = json.loads((config.DATA_DIR / "sample_incident_logs.json").read_text(encoding="utf-8"))
    if len(documents) != 20:
        raise ValueError(f"Expected 20 incident documents; found {len(documents)}.")
    measurements = []

    if compare_sequential:
        started = time.perf_counter()
        summaries = [summarize_doc_node({"document": document})["summaries"][0] for document in documents]
        reduce_summaries_node({"summaries": summaries})
        elapsed = (time.perf_counter() - started) * 1000
        measurements.append(
            {
                "strategy": "sequential",
                "documents": len(documents),
                "duration_ms": round(elapsed, 2),
            }
        )

    started = time.perf_counter()
    result = map_reduce_graph.invoke(
        {"documents": documents, "summaries": [], "final_brief": ""},
        config=config.get_run_config(
            model=config.LOCAL_MODEL if config.USE_LOCAL_LLM else config.LLM_MODEL,
            provider="local-ollama" if config.USE_LOCAL_LLM else "cloud-gemini",
        ),
    )
    elapsed = (time.perf_counter() - started) * 1000
    if len(result["summaries"]) != len(documents):
        raise RuntimeError(f"Map stage returned {len(result['summaries'])} summaries for " f"{len(documents)} inputs.")
    measurements.append(
        {
            "strategy": "parallel_send",
            "documents": len(documents),
            "duration_ms": round(elapsed, 2),
        }
    )

    output = config.BENCHMARKS_DIR / "map_reduce_timings.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    for row in measurements:
        config.log_metric(
            "map_reduce_duration_ms",
            float(row["duration_ms"]),
            strategy=row["strategy"],
            documents=row["documents"],
        )
        print(f"{row['strategy']}: {row['duration_ms']:.2f} ms " f"for {row['documents']} documents")
    print(f"Timings written to {output}")
    return measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-sequential",
        action="store_true",
        help="Run both sequential and parallel model calls; uses more model tokens.",
    )
    args = parser.parse_args()
    run_benchmark(compare_sequential=args.compare_sequential)
