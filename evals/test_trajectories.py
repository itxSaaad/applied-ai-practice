"""Provider-backed trajectory evaluation plus fast evaluator contract tests."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

import config
from day5_observability.pattern3_evaluators import (
    evaluate_latency,
    evaluate_tool_trajectory,
)


def test_tool_trajectory_requires_expected_order_when_provided():
    example = SimpleNamespace(outputs={"expected_tools": ["query_incidents", "read_log", "summarize"]})
    correct = SimpleNamespace(
        child_spans=[
            SimpleNamespace(name=name, run_type="tool") for name in ("query_incidents", "read_log", "summarize")
        ]
    )
    wrong_order = SimpleNamespace(
        child_spans=[
            SimpleNamespace(name=name, run_type="tool") for name in ("read_log", "query_incidents", "summarize")
        ]
    )
    assert evaluate_tool_trajectory(correct, example)["score"] == 1
    assert evaluate_tool_trajectory(wrong_order, example)["score"] == 0


def test_latency_evaluator_returns_a_bounded_pass_fail_score():
    started = datetime.now(UTC)
    example = SimpleNamespace(outputs={"latency_budget_ms": 1000})
    within_budget = SimpleNamespace(
        start_time=started,
        end_time=started + timedelta(milliseconds=10),
    )
    result = evaluate_latency(within_budget, example)
    assert result["score"] == 1
    over_budget = SimpleNamespace(
        start_time=started,
        end_time=started + timedelta(seconds=2),
    )
    assert evaluate_latency(over_budget, example)["score"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
@config.trace_operation("day4_pipeline_evaluation", flush=True)
async def test_adk_incident_pipeline_trajectory():
    await AgentEvaluator.evaluate(
        agent_module="day4_adk",
        eval_dataset_file_path_or_dir=("evals/datasets/day4_incident_pipeline.evalset.json"),
        num_runs=1,
    )
