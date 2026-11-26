"""Compare citation retrieval and latency for Day 2 RAG and Day 3 CRAG."""

import csv
import time

from langchain_core.messages import HumanMessage

import config
from day2_langchain.pattern5_rag_agent import rag_pipeline
from day3_langgraph.pattern1_crag import build_crag_graph


@config.trace_operation("day3_crag_vs_rag_benchmark", flush=True)
def run_benchmark() -> list[dict[str, str | int | float]]:
    with (config.DATA_DIR / "docs_qa.csv").open(encoding="utf-8", newline="") as file:
        questions = list(csv.DictReader(file))
    if len(questions) != 25:
        raise ValueError(f"Expected 25 comparison questions; found {len(questions)}.")

    graph = build_crag_graph()
    run_config = {"recursion_limit": 15, **config.get_run_config()}
    results = []
    for row in questions:
        for method in ("rag", "crag"):
            started = time.perf_counter()
            if method == "rag":
                response = rag_pipeline(row["question"])
                cited = any(citation.source_url == row["expected_source"] for citation in response.citations)
            else:
                response = graph.invoke(
                    {
                        "messages": [HumanMessage(content=f"Lookup runbook evidence: {row['question']}")],
                        "documents": [],
                        "question": "",
                        "rewrite_count": 0,
                        "grounded": False,
                    },
                    config=run_config,
                )
                final_message = response["messages"][-1]
                text = str(getattr(final_message, "text", final_message.content))
                cited = row["expected_source"] in text
            latency = (time.perf_counter() - started) * 1000
            results.append(
                {
                    "case_id": row["case_id"],
                    "method": method,
                    "expected_source_cited": int(cited),
                    "latency_ms": round(latency, 2),
                }
            )
            config.log_metric(
                "qa_case_latency_ms",
                round(latency, 2),
                case_id=row["case_id"],
                method=method,
                expected_source_cited=cited,
            )

    output = config.BENCHMARKS_DIR / "crag_vs_rag_results.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for method in ("rag", "crag"):
        cases = [item for item in results if item["method"] == method]
        citation_rate = sum(int(item["expected_source_cited"]) for item in cases) / len(cases)
        average_latency = sum(float(item["latency_ms"]) for item in cases) / len(cases)
        print(
            f"{method.upper()}: expected-source citation rate={citation_rate:.1%}, "
            f"mean latency={average_latency:.1f} ms"
        )
    print(f"Per-question measurements: {output}")
    return results


if __name__ == "__main__":
    run_benchmark()
