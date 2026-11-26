"""Data Agent: Read-only SQL LangGraph subgraph over OpsPilot services/deploys/incidents."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph

import config
from capstone.seed import OPSPILOT_DB_PATH, seed_opspilot_database
from utils.sqlite import fetch_read_only_rows, is_single_select

if TYPE_CHECKING:
    from pathlib import Path

logger = config.get_logger(__name__)


class DataAgentState(TypedDict):
    question: str
    sql: str
    rows: list[dict]
    answer: str


@config.trace_operation("opspilot_sql_query")
def run_read_only_sql(sql: str, db_path: Path | None = None) -> list[dict]:
    """Execute one strictly validated SELECT statement against OpsPilot SQLite."""
    target = db_path or OPSPILOT_DB_PATH
    if not target.exists():
        seed_opspilot_database(target)
    if not is_single_select(sql):
        logger.warning("Blocked non-SELECT SQL attempt.", extra={"guardrail_triggered": True})
        raise ValueError("Security violation: only a single read-only SELECT statement is allowed.")
    rows = fetch_read_only_rows(target, sql)
    config.log_metric("opspilot_sql_rows", float(len(rows)))
    return rows


def question_to_sql(question: str) -> str:
    """Deterministic fallback SQL mapper for common OpsPilot incident/deploy questions."""
    q = question.lower()
    if "deploy" in q or "version" in q or "v2.30.1" in q:
        return (
            "SELECT deploy_id, service_id, version, status, deployed_at "
            "FROM deploys ORDER BY deployed_at DESC LIMIT 10"
        )
    if "service" in q or "owner" in q or "tier" in q:
        return "SELECT service_id, owner_team, tier FROM services ORDER BY tier, service_id"
    if "payments" in q or "spike" in q or "open" in q or "critical" in q:
        return (
            "SELECT incident_id, service_id, title, severity, status, root_cause, created_at "
            "FROM incidents WHERE status = 'open' OR severity = 'critical' ORDER BY created_at DESC LIMIT 10"
        )
    return "SELECT incident_id, service_id, title, severity, status, created_at FROM incidents ORDER BY created_at DESC LIMIT 10"


def build_data_agent(llm=None, db_path: Path | None = None):
    """Build the read-only SQL Data Agent subgraph."""

    def plan_sql_node(state: DataAgentState) -> dict:
        if state.get("sql"):
            return {"sql": state["sql"]}
        if llm is None:
            return {"sql": question_to_sql(state["question"])}
        prompt = (
            "Return ONLY one SQLite SELECT query (no markdown) for tables "
            "services(service_id, owner_team, tier, description), "
            "deploys(deploy_id, service_id, version, deployed_by, status, deployed_at), "
            "incidents(incident_id, service_id, title, severity, status, root_cause, created_at).\n"
            f"Question: {state['question']}"
        )
        raw = str(llm.invoke(prompt).content).strip().strip("`").removeprefix("sql").strip()
        return {"sql": raw}

    def execute_sql_node(state: DataAgentState) -> dict:
        rows = run_read_only_sql(state["sql"], db_path=db_path)
        return {"rows": rows, "answer": json.dumps(rows, indent=2)}

    graph = StateGraph(DataAgentState)
    graph.add_node("plan_sql", plan_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_edge(START, "plan_sql")
    graph.add_edge("plan_sql", "execute_sql")
    graph.add_edge("execute_sql", END)
    return graph.compile()
