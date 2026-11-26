import csv
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

import config
from day2_langchain.pattern2_safe_tool import query_read_only_sqlite
from day2_langchain.pattern5_rag_agent import rag_pipeline
from llm import get_llm
from utils.logging import get_logger

logger = get_logger(__name__)

llm = None


def _get_model():
    return llm if llm is not None else get_llm()


class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_worker: str
    task_result: str


class SubgraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _format_worker_input(role_prompt: str, messages: list[BaseMessage]) -> list[BaseMessage]:
    """Ensures worker invocations conform to Gemini chat role constraints."""
    last_msg = messages[-1] if messages else HumanMessage(content="Perform assigned task.")
    last_text = getattr(last_msg, "text", last_msg.content)
    if isinstance(last_text, list):
        last_text = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in last_text])
    return [
        SystemMessage(content=role_prompt),
        HumanMessage(content=str(last_text)),
    ]


@tool
@config.trace_operation("day3_supervisor_runbook_search", capture_input=False)
def search_runbooks(query: str) -> str:
    """Search the embedded runbook index and return a cited structured answer."""
    return rag_pipeline(query).model_dump_json()


@tool
@config.trace_operation(
    "day3_supervisor_incident_query",
    capture_input=False,
    capture_output=False,
)
def query_incidents_sql(query: str) -> str:
    """Execute a read-only SELECT query against the seeded incidents sample."""
    return query_read_only_sqlite.invoke({"query": query})


def researcher_worker(state: SubgraphState) -> dict:
    question = state["messages"][-1].content
    result = search_runbooks.invoke({"query": str(question)})
    return {"messages": [HumanMessage(content=f"Research evidence:\n{result}")]}


research_builder = StateGraph(SubgraphState)
research_builder.add_node("researcher_worker", researcher_worker)
research_builder.add_edge(START, "researcher_worker")
research_builder.add_edge("researcher_worker", END)
research_subgraph = research_builder.compile()


def analyst_worker(state: SubgraphState) -> dict:
    result = query_incidents_sql.invoke({"query": "SELECT id, title, severity, status FROM incidents LIMIT 10"})
    return {"messages": [HumanMessage(content=f"Incident data:\n{result}")]}


analyst_builder = StateGraph(SubgraphState)
analyst_builder.add_node("analyst_worker", analyst_worker)
analyst_builder.add_edge(START, "analyst_worker")
analyst_builder.add_edge("analyst_worker", END)
analyst_subgraph = analyst_builder.compile()


def writer_worker(state: SubgraphState) -> dict:
    res = _get_model().invoke(_format_worker_input("You are a Technical Writer.", state["messages"]))
    return {"messages": [res]}


writer_builder = StateGraph(SubgraphState)
writer_builder.add_node("writer_worker", writer_worker)
writer_builder.add_edge(START, "writer_worker")
writer_builder.add_edge("writer_worker", END)
writer_subgraph = writer_builder.compile()


class SupervisorRoute(BaseModel):
    next_step: Literal["researcher", "analyst", "writer", "FINISH"] = Field(
        description="The next specialized worker agent to route execution to, or FINISH when complete."
    )


def supervisor_node(state: TeamState) -> Command[Literal["researcher", "analyst", "writer", END]]:
    router = _get_model().with_structured_output(SupervisorRoute)
    system_prompt = (
        "You manage researcher, analyst, and writer workers. Choose researcher for "
        "runbook questions, analyst for incident data, and writer after evidence is "
        "available. Return FINISH only after the writer has produced the final response."
    )
    messages_to_send: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages_to_send.extend(state["messages"])
    decision = router.invoke(messages_to_send)
    if decision is None:
        raise RuntimeError("Supervisor returned no routing decision.")
    next_step = decision.next_step

    if next_step == "FINISH":
        return Command(goto=END)

    logger.info("Supervisor routed task to %s.", next_step)
    return Command(goto=next_step, update={"next_worker": next_step})


main_builder = StateGraph(TeamState)
main_builder.add_node("supervisor", supervisor_node)
main_builder.add_node("researcher", research_subgraph)
main_builder.add_node("analyst", analyst_subgraph)
main_builder.add_node("writer", writer_subgraph)

main_builder.add_edge(START, "supervisor")
main_builder.add_edge("researcher", "supervisor")
main_builder.add_edge("analyst", "supervisor")
main_builder.add_edge("writer", "supervisor")

supervisor_team = main_builder.compile()


if __name__ == "__main__":
    with (config.DATA_DIR / "day3_supervisor_tasks.csv").open(encoding="utf-8", newline="") as file:
        tasks = list(csv.DictReader(file))
    invoke_config = {"recursion_limit": 12, **config.get_run_config()}
    for task in tasks:
        result = supervisor_team.invoke(
            {
                "messages": [HumanMessage(content=task["task"])],
                "next_worker": "",
                "task_result": "",
            },
            config=invoke_config,
        )
        last_msg = result["messages"][-1]
        print(
            f"{task['task_id']} Supervisor Team Final Output:\n",
            getattr(last_msg, "text", last_msg.content),
        )
    config.flush_traces()
