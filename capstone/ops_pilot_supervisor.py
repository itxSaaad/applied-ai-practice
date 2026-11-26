"""OpsPilot LangGraph Supervisor orchestrating Knowledge, Data, ADK A2A, and Writer agents."""

from __future__ import annotations

import os
import sqlite3
from typing import TYPE_CHECKING, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
from capstone.a2a_client import call_adk_analyst
from capstone.data_agent import build_data_agent
from capstone.guardrails import redact_pii
from capstone.knowledge_agent import build_knowledge_agent
from capstone.writer_agent import writer_node
from day2_langchain.pattern3_agent_middleware import MAX_MODEL_CALLS
from day5_observability.pattern2_prompt_management import fetch_production_system_prompt

if TYPE_CHECKING:
    from pathlib import Path

logger = config.get_logger(__name__)

DEFAULT_CHECKPOINT_DB = config.ARTIFACTS_DIR / "db" / "opspilot_checkpoints.sqlite"
MAX_SUPERVISOR_STEPS = MAX_MODEL_CALLS


class OpsPilotState(TypedDict, total=False):
    question: str
    route: str
    step_count: int
    knowledge_answer: str
    citations: list[str]
    data_answer: str
    adk_answer: str
    adk_degraded: bool
    draft: str
    approval_status: str
    jira_created: bool
    slack_posted: bool
    final_answer: str


class RouteDecision(BaseModel):
    next_step: Literal["knowledge", "data", "adk_analyst", "writer", "finish"] = Field(
        description="Next OpsPilot specialist to invoke."
    )


def get_checkpointer(db_path: Path | None = None):
    """Return Postgres checkpointer when POSTGRES_URI is configured, else durable SQLite checkpointer."""
    postgres_uri = os.getenv("POSTGRES_URI", "").strip()
    if postgres_uri:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            saver = PostgresSaver.from_conn_string(postgres_uri)
            saver.setup()
            return saver
        except Exception:
            pass

    target = db_path or DEFAULT_CHECKPOINT_DB
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    return SqliteSaver(conn)


def _heuristic_route(state: OpsPilotState) -> str:
    q = state.get("question", "").lower()
    if any(w in q for w in ("postmortem", "jira", "slack", "ticket", "remediate")):
        if not state.get("data_answer"):
            return "data"
        if not state.get("knowledge_answer"):
            return "knowledge"
        if not state.get("approval_status"):
            return "writer"
        return "finish"
    if any(w in q for w in ("adk", "analyst", "correlate")):
        return "finish" if state.get("adk_answer") else "adk_analyst"
    if any(w in q for w in ("sql", "incident", "deploy", "table", "count", "how many", "owner")):
        return "finish" if state.get("data_answer") else "data"
    return "finish" if state.get("knowledge_answer") else "knowledge"


def build_opspilot_graph(*, checkpointer=None, llm=None, db_path: Path | None = None, adk_url: str | None = None):
    """Compile the OpsPilot Supervisor StateGraph with durable checkpointer."""
    knowledge_graph = build_knowledge_agent(llm=llm)
    data_graph = build_data_agent(llm=llm, db_path=db_path)

    def supervisor_node(state: OpsPilotState) -> dict:
        steps = int(state.get("step_count", 0)) + 1
        if steps > MAX_SUPERVISOR_STEPS:
            logger.warning("Supervisor step budget reached; finishing.", extra={"step_count": steps})
            return {"route": "finish", "step_count": steps}
        if llm is None:
            return {"route": _heuristic_route(state), "step_count": steps}
        supervisor_prompt = fetch_production_system_prompt("ops_pilot_supervisor_prompt")
        router = llm.with_structured_output(RouteDecision)
        decision = router.invoke(
            f"{supervisor_prompt}\n\nQuestion: {state.get('question', '')}\n"
            f"Have knowledge: {bool(state.get('knowledge_answer'))}, "
            f"Have data: {bool(state.get('data_answer'))}, "
            f"Have adk: {bool(state.get('adk_answer'))}, "
            f"Have approval: {bool(state.get('approval_status'))}",
            config=config.get_run_config(model=config.LLM_MODEL),
        )
        return {"route": decision.next_step, "step_count": steps}

    def knowledge_step(state: OpsPilotState) -> dict:
        res = knowledge_graph.invoke({"question": state["question"], "documents": [], "rewrite_count": 0})
        return {"knowledge_answer": res["answer"], "citations": res.get("citations", [])}

    def data_step(state: OpsPilotState) -> dict:
        res = data_graph.invoke({"question": state["question"]})
        return {"data_answer": res["answer"]}

    def adk_step(state: OpsPilotState) -> dict:
        res = call_adk_analyst(state["question"], base_url=adk_url)
        return {"adk_answer": res["analysis"], "adk_degraded": bool(res.get("degraded", False))}

    def finalize_step(state: OpsPilotState) -> dict:
        if state.get("final_answer"):
            return {"final_answer": redact_pii(state["final_answer"])}
        parts = [
            state.get("knowledge_answer", ""),
            state.get("data_answer", ""),
            state.get("adk_answer", ""),
        ]
        combined = "\n\n".join(p for p in parts if p).strip() or "No answer produced."
        return {"final_answer": redact_pii(combined)}

    graph = StateGraph(OpsPilotState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("knowledge", knowledge_step)
    graph.add_node("data", data_step)
    graph.add_node("adk_analyst", adk_step)
    graph.add_node("writer", writer_node)
    graph.add_node("finalize", finalize_step)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["route"],
        {
            "knowledge": "knowledge",
            "data": "data",
            "adk_analyst": "adk_analyst",
            "writer": "writer",
            "finish": "finalize",
        },
    )
    graph.add_edge("knowledge", "supervisor")
    graph.add_edge("data", "supervisor")
    graph.add_edge("adk_analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("finalize", END)

    active_cp = checkpointer if checkpointer is not None else get_checkpointer()
    return graph.compile(checkpointer=active_cp)
