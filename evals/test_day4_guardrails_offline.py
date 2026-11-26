import json

import pytest

import config
from data.seed.seed_incidents_db import seed_database
from day4_adk import pattern1_pipeline
from day4_adk.pattern3_guardrail_callbacks import (
    before_model_guardrail,
    enforce_read_only_sql,
    query_incidents,
)
from day4_adk.pattern6_supervisor import adk_supervisor


def test_seeded_database_and_read_only_query(tmp_path, monkeypatch):
    database = tmp_path / "incidents.sqlite"
    monkeypatch.setattr("day4_adk.pattern3_guardrail_callbacks.DATABASE", database)
    assert seed_database(database) == 6
    records = json.loads(query_incidents("SELECT incident_id, service FROM incidents WHERE severity = 'critical'"))
    assert records[0]["incident_id"] == "INC001"
    for sql in ("DELETE FROM incidents", "SELECT 1; DROP TABLE incidents", "PRAGMA table_info(incidents)"):
        assert enforce_read_only_sql("query_incidents", {"sql": sql}) is not None
        with pytest.raises(ValueError, match="SELECT"):
            query_incidents(sql)


def test_sample_logs_are_read_and_paths_are_confined():
    auth_logs = pattern1_pipeline.read_log("auth-service", 15)
    assert "cache hit rate dropped" in auth_logs
    assert "token refresh latency" in auth_logs
    assert "queue depth" in pattern1_pipeline.read_log("worker-service", 15)
    with pytest.raises(ValueError, match="Service name"):
        pattern1_pipeline.read_log("../README", 15)
    with pytest.raises(ValueError, match="positive"):
        pattern1_pipeline.read_log("auth-service", 0)


def test_pii_is_redacted_in_plain_text_and_adk_request():
    sanitized = before_model_guardrail(
        "guarded-agent",
        "Contact operator@example.com at 555-867-5309.",
    )
    assert "[REDACTED_EMAIL]" in sanitized
    assert "[REDACTED_PHONE]" in sanitized


def test_day4_evalset_has_ten_cases_and_read_log_trajectory():
    path = config.ROOT_DIR / "evals" / "datasets" / "day4_incident_pipeline.evalset.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    cases = dataset["eval_cases"]
    assert len(cases) == 10
    assert all(
        "read_log" in {tool["name"] for tool in case["conversation"][0]["intermediate_data"]["tool_uses"]}
        for case in cases
    )


def test_adk_supervisor_has_three_specialized_sub_agents():
    assert {agent.name for agent in adk_supervisor.sub_agents} == {
        "researcher",
        "analyst",
        "writer",
    }
