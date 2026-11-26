import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from langsmith.evaluation import evaluate
from openevals.llm import create_llm_as_judge

import config
from utils.logging import get_logger

JUDGE_MODEL = f"ollama:{config.LOCAL_MODEL}" if config.USE_LOCAL_LLM else f"google_genai:{config.LLM_MODEL}"
logger = get_logger(__name__)

correctness_evaluator = create_llm_as_judge(
    prompt="""You are an expert evaluator scoring an AI agent's response.
Compare the Agent Response against the Ground Truth Reference.
Input Query: {inputs}
Agent Response: {outputs}
Ground Truth: {reference_outputs}

Score 1 if the agent response is factually accurate and complete according to ground truth, else score 0.""",
    feedback_key="factual_correctness",
    model=JUDGE_MODEL,
)


def evaluate_citation_presence(run, example) -> dict:
    """Check whether the final response includes source citations."""
    output_text = str((run.outputs or {}).get("output", ""))
    has_citation = "http://" in output_text or "https://" in output_text or "Source:" in output_text
    return {"key": "citation_present", "score": int(has_citation)}


def evaluate_tool_trajectory(run, example) -> dict:
    """Compare observed tool spans with an expected tool or ordered trajectory."""
    spans = getattr(run, "child_spans", None) or []
    executed_tools = [span.name for span in spans if getattr(span, "run_type", "") == "tool"]
    outputs = example.outputs or {}
    expected_tools = outputs.get("expected_tools")
    expected_tool = outputs.get("expected_tool")
    if expected_tools is not None:
        if not isinstance(expected_tools, list) or any(not isinstance(tool, str) for tool in expected_tools):
            raise ValueError("expected_tools must be an ordered list of tool names.")
        trajectory_valid = executed_tools == expected_tools
        expected = " -> ".join(expected_tools)
    elif expected_tool:
        trajectory_valid = expected_tool in executed_tools
        expected = expected_tool
    else:
        trajectory_valid = bool(executed_tools)
        expected = "at least one tool call"
    return {
        "key": "trajectory_correct",
        "score": int(trajectory_valid),
        "comment": f"expected={expected}; actual={' -> '.join(executed_tools)}",
    }


def evaluate_latency(run, example) -> dict:
    """Score whether a run completed within the example's optional latency budget."""
    start = getattr(run, "start_time", None)
    end = getattr(run, "end_time", None)
    if start is None or end is None:
        return {"key": "latency_ms", "score": 0.0, "comment": "Run timestamps unavailable"}
    if end < start:
        return {
            "key": "latency_ms",
            "score": 0,
            "comment": "Run end_time precedes start_time.",
        }
    latency_ms = (end - start).total_seconds() * 1000
    budget_ms = (example.outputs or {}).get("latency_budget_ms", 30_000)
    if not isinstance(budget_ms, (int, float)) or budget_ms <= 0:
        raise ValueError("latency_budget_ms must be a positive number.")
    within_budget = latency_ms <= budget_ms
    return {
        "key": "latency_ms",
        "score": int(within_budget),
        "comment": f"latency_ms={latency_ms:.1f}; budget_ms={budget_ms}",
    }


def run_benchmark_experiment(dataset_name: str, target_agent_fn):
    """Run the benchmark dataset with factual, citation, and trajectory evaluators."""
    client = config.get_langsmith_client()
    if client is None:
        raise ValueError("LangSmith evaluation requires LANGSMITH_TRACING=true and LANGSMITH_API_KEY.")
    return evaluate(
        target_agent_fn,
        data=dataset_name,
        evaluators=[
            correctness_evaluator,
            evaluate_citation_presence,
            evaluate_tool_trajectory,
            evaluate_latency,
        ],
        experiment_prefix="ops-pilot-eval-run",
        client=client,
    )


def calibrate_judge(
    *,
    dataset_path: Path | None = None,
    judge_model: str = JUDGE_MODEL,
) -> dict[str, str | float | int]:
    """Measure judge agreement against reviewed labels and fail below 80%."""
    from llm import _extract_clean_text, get_chat_model

    source = dataset_path or (config.ROOT_DIR / "evals" / "datasets" / "day5_judge_calibration.json")
    dataset = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or dataset.get("labels_reviewed") is not True:
        raise ValueError(
            "Judge calibration is blocked until every provisional label has been "
            "reviewed and labels_reviewed is set to true."
        )
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Calibration dataset must contain an examples list.")
    if len(examples) != 20:
        raise ValueError(f"Expected 20 calibration labels; found {len(examples)}.")
    if any(
        not isinstance(example, dict)
        or example.get("human_score") not in (0, 1)
        or not all(example.get(field) for field in ("input", "output", "reference"))
        for example in examples
    ):
        raise ValueError("Each reviewed example needs input, output, reference, and binary human_score.")

    judge = get_chat_model(judge_model, temperature=0)
    matches = 0
    for example in examples:
        response = judge.invoke(
            "Compare the candidate answer with the reference. Return exactly PASS or FAIL.\n"
            f"Question: {example['input']}\n"
            f"Reference: {example['reference']}\n"
            f"Candidate: {example['output']}",
            config=config.get_run_config(
                prompt_version="judge-calibration-v1",
                model=judge_model,
                provider="local-ollama" if judge_model.startswith("ollama:") else "cloud",
            ),
        )
        decision = _extract_clean_text(response).strip().upper()
        prediction = int(decision.startswith("PASS"))
        matches += prediction == example["human_score"]

    agreement = matches / len(examples)
    report: dict[str, str | float | int] = {
        "dataset_id": dataset.get("dataset_id", source.stem),
        "timestamp": datetime.now(UTC).isoformat(),
        "model": judge_model,
        "matches": matches,
        "total": len(examples),
        "agreement_rate": agreement,
        "minimum_agreement": 0.8,
    }
    report_path = config.BENCHMARKS_DIR / "judge_calibration_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    config.log_metric(
        "judge_agreement_rate",
        agreement * 100,
        matches=matches,
        total=len(examples),
        minimum_agreement=80.0,
    )
    logger.info("Judge calibration report written to %s.", report_path)
    if agreement < 0.8:
        raise AssertionError(f"Judge agreement {agreement:.1%} is below the required 80.0%.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run the provider-backed judge calibration on reviewed examples.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional reviewed calibration dataset JSON path.",
    )
    args = parser.parse_args()
    if args.calibrate:
        print(calibrate_judge(dataset_path=args.dataset))
        return 0

    mock_run = SimpleNamespace(
        outputs={"output": "Restart worker pool per Source: https://runbooks.internal/error-500"},
        child_spans=[SimpleNamespace(name="query_incidents_sql", run_type="tool")],
    )
    mock_example = SimpleNamespace(outputs={"expected_tool": "query_incidents_sql"})

    print("Citation Evaluator  :", evaluate_citation_presence(mock_run, mock_example))
    print("Trajectory Evaluator:", evaluate_tool_trajectory(mock_run, mock_example))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
