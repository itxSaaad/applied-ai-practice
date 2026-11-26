import csv
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
from data.seed.seed_incidents_db import seed_database
from day2_langchain.pattern1_fallbacks import build_resilient_model
from day2_langchain.pattern2_safe_tool import query_read_only_sqlite
from day2_langchain.pattern3_agent_middleware import (
    MAX_MODEL_CALLS,
    call_budget_middleware,
    tools,
)
from day2_langchain.pattern4_extract_api import Ticket


def test_sample_tickets_and_retrieval_questions_meet_course_counts():
    with (config.DATA_DIR / "sample_tickets.csv").open(encoding="utf-8", newline="") as file:
        tickets = list(csv.DictReader(file))
    with (config.DATA_DIR / "docs_qa.csv").open(encoding="utf-8", newline="") as file:
        questions = list(csv.DictReader(file))
    assert len(tickets) == 20
    assert len(questions) == 25
    assert all(row["raw_text"].strip() for row in tickets)
    assert all(row["question"].strip() and row["reference_answer"].strip() for row in questions)
    with (config.DATA_DIR / "day2_agent_questions.csv").open(encoding="utf-8", newline="") as file:
        agent_cases = list(csv.DictReader(file))
    assert len(agent_cases) == 10
    assert {case["expected_tool"] for case in agent_cases} == {
        "calculate_metrics",
        "query_employee_directory",
        "query_read_only_sqlite",
    }
    assert {tool.name for tool in tools} == {
        "calculate_metrics",
        "query_employee_directory",
        "query_read_only_sqlite",
    }


def test_ticket_schema_validates_required_extraction_fields():
    ticket = Ticket.model_validate(
        {
            "category": "Infrastructure",
            "priority": "Critical",
            "product": "checkout-service",
            "sentiment": "Negative",
            "summary": "Checkout requests are timing out.",
        }
    )
    assert ticket.priority == "Critical"


def test_sqlite_tool_rejects_mutations_and_statement_chaining(tmp_path, monkeypatch):
    database = tmp_path / "incidents.db"
    seed_database(database)
    monkeypatch.setattr("day2_langchain.pattern2_safe_tool.DB_PATH", database)
    result = query_read_only_sqlite.invoke({"query": "SELECT COUNT(*) AS n FROM incidents"})
    assert '"n": 6' in result
    for query in (
        "DELETE FROM incidents",
        "DROP TABLE incidents;",
        "SELECT 1; DROP TABLE incidents",
        "PRAGMA table_info(incidents)",
    ):
        with pytest.raises(ValueError, match="forbidden"):
            query_read_only_sqlite.invoke({"query": query})


def test_model_call_budget_fails_closed_and_counts_each_invocation():
    state = {"model_call_count": MAX_MODEL_CALLS - 1}
    request = SimpleNamespace(state=state)
    invoke_middleware = call_budget_middleware.wrap_model_call
    assert invoke_middleware(request, lambda _: "ok") == "ok"
    assert state["model_call_count"] == MAX_MODEL_CALLS
    with pytest.raises(RuntimeError, match="budget"):
        invoke_middleware(request, lambda _: "must not run")
    with pytest.raises(RuntimeError, match="mutable dictionary"):
        invoke_middleware(SimpleNamespace(state=None), lambda _: "must not run")
    with pytest.raises(ValueError, match="non-negative integer"):
        invoke_middleware(
            SimpleNamespace(state={"model_call_count": True}),
            lambda _: "must not run",
        )


def test_lcel_fallback_runs_after_primary_raises():
    def fail(_):
        raise RuntimeError("simulated rate limit")

    primary = RunnableLambda(fail)
    fallback = RunnableLambda(lambda _: AIMessage(content="fallback response"))
    model = build_resilient_model(primary, fallback)
    response = model.invoke("request")
    assert response.content == "fallback response"
