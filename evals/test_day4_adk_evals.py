import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

import config
from day4_adk.pattern2_critique_loop import refinement_loop
from day4_adk.pattern3_guardrail_callbacks import (
    before_model_guardrail,
    enforce_read_only_sql,
)


def test_pii_redaction_guardrail():
    raw = "User john.doe@enterprise.com reported outage; call 555-867-5309."
    sanitized = before_model_guardrail("guarded_db_agent", raw)
    assert "john.doe@enterprise.com" not in sanitized
    assert "555-867-5309" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert "[REDACTED_PHONE]" in sanitized


def test_sql_read_only_short_circuit():
    for sql in ("DELETE FROM incidents", "DROP TABLE incidents"):
        blocked = enforce_read_only_sql("query_incidents", {"sql": sql})
        assert blocked is not None
        assert blocked["error"] == (
            "Security Violation: Only read-only SELECT queries are allowed."
        )

    allowed = enforce_read_only_sql(
        "query_incidents",
        {"sql": "SELECT * FROM incidents WHERE status = 'open'"},
    )
    assert allowed is None


def test_critique_loop_bound():
    assert refinement_loop.max_iterations == 3
    assert len(refinement_loop.sub_agents) == 2


def test_flush_does_not_initialize_tracing_clients(monkeypatch):
    monkeypatch.setattr(config, "_langfuse_client", None)
    monkeypatch.setattr(config, "_langsmith_client", None)
    monkeypatch.setattr(
        config,
        "get_langfuse_client",
        lambda: pytest.fail("flush must not initialize Langfuse at shutdown"),
    )
    monkeypatch.setattr(
        config,
        "get_langsmith_client",
        lambda: pytest.fail("flush must not initialize LangSmith at shutdown"),
    )

    config.flush_traces()


@pytest.mark.asyncio
@config.trace_operation("day4_pipeline_evaluation", flush=True)
async def test_adk_pipeline_trajectory():
    await AgentEvaluator.evaluate(
        agent_module="day4_adk",
        eval_dataset_file_path_or_dir=(
            "evals/day4_incident_pipeline.evalset.json"
        ),
        num_runs=1,
    )
