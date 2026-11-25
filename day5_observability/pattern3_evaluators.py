import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langsmith.evaluation import evaluate
from openevals.llm import create_llm_as_judge

JUDGE_MODEL = (
    f"ollama:{config.LOCAL_MODEL}"
    if config.USE_LOCAL_LLM
    else f"google_genai:{config.LLM_MODEL}"
)

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
    """Check child spans for an expected tool execution."""
    spans = getattr(run, "child_spans", None) or []
    executed_tools = [span.name for span in spans if getattr(span, "run_type", "") == "tool"]
    expected_tool = (example.outputs or {}).get("expected_tool")
    trajectory_valid = expected_tool in executed_tools if expected_tool else bool(executed_tools)
    return {"key": "trajectory_correct", "score": int(trajectory_valid)}


def run_benchmark_experiment(dataset_name: str, target_agent_fn):
    """Run the benchmark dataset with factual, citation, and trajectory evaluators."""
    client = config.get_langsmith_client()
    if client is None:
        raise ValueError(
            "LangSmith evaluation requires LANGSMITH_TRACING=true and LANGSMITH_API_KEY."
        )
    return evaluate(
        target_agent_fn,
        data=dataset_name,
        evaluators=[
            correctness_evaluator,
            evaluate_citation_presence,
            evaluate_tool_trajectory,
        ],
        experiment_prefix="ops-pilot-eval-run",
        client=client,
    )


if __name__ == "__main__":
    mock_run = SimpleNamespace(
        outputs={"output": "Restart worker pool per Source: https://runbooks.internal/error-500"},
        child_spans=[SimpleNamespace(name="query_incidents_sql", run_type="tool")],
    )
    mock_example = SimpleNamespace(outputs={"expected_tool": "query_incidents_sql"})

    print("Citation Evaluator  :", evaluate_citation_presence(mock_run, mock_example))
    print("Trajectory Evaluator:", evaluate_tool_trajectory(mock_run, mock_example))
