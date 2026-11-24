import logging
import os
import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from typing_extensions import Annotated
from dotenv import load_dotenv

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Supervisor")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", config.LLM_MODEL),
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0,
)


class TeamState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_worker: str
    task_result: str


class SubgraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def _format_worker_input(role_prompt: str, messages: List[BaseMessage]) -> List[BaseMessage]:
    """Ensures worker invocations conform to Gemini chat role constraints."""
    last_msg = messages[-1] if messages else HumanMessage(content="Perform assigned task.")
    last_text = getattr(last_msg, "text", last_msg.content)
    if isinstance(last_text, list):
        last_text = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in last_text])
    return [
        SystemMessage(content=role_prompt),
        HumanMessage(content=str(last_text)),
    ]


def researcher_worker(state: SubgraphState) -> dict:
    res = llm.invoke(_format_worker_input("You are a Researcher with RAG search tools.", state["messages"]))
    return {"messages": [res]}


research_builder = StateGraph(SubgraphState)
research_builder.add_node("researcher_worker", researcher_worker)
research_builder.add_edge(START, "researcher_worker")
research_builder.add_edge("researcher_worker", END)
research_subgraph = research_builder.compile()


def analyst_worker(state: SubgraphState) -> dict:
    res = llm.invoke(_format_worker_input("You are a Data Analyst with read-only SQL tools.", state["messages"]))
    return {"messages": [res]}


analyst_builder = StateGraph(SubgraphState)
analyst_builder.add_node("analyst_worker", analyst_worker)
analyst_builder.add_edge(START, "analyst_worker")
analyst_builder.add_edge("analyst_worker", END)
analyst_subgraph = analyst_builder.compile()


def writer_worker(state: SubgraphState) -> dict:
    res = llm.invoke(_format_worker_input("You are a Technical Writer.", state["messages"]))
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
    try:
        router = llm.with_structured_output(SupervisorRoute)
        system_prompt = (
            "You manage a team of workers: researcher, analyst, and writer. "
            "Select the next worker based on request status. Return FINISH once the writer has produced the final response."
        )
        messages_to_send: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for m in state["messages"]:
            messages_to_send.append(m)

        if messages_to_send and not isinstance(messages_to_send[-1], (HumanMessage, SystemMessage)):
            messages_to_send.append(HumanMessage(content="Evaluate progress above and decide next step: researcher, analyst, writer, or FINISH."))

        decision = router.invoke(messages_to_send)
        next_step = decision.next_step if decision else "FINISH"
    except Exception as e:
        logger.warning(f"Supervisor routing fallback: {e}")
        next_step = "FINISH"

    if next_step == "FINISH":
        return Command(goto=END)

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
    callbacks = config.get_callbacks()
    invoke_config = {"recursion_limit": 12}
    if callbacks:
        invoke_config["callbacks"] = callbacks

    try:
        result = supervisor_team.invoke(
            {
                "messages": [HumanMessage(content="Research why HTTP 504 timeouts happen in microservices and write a 3-bullet mitigation guide.")],
                "next_worker": "",
                "task_result": "",
            },
            config=invoke_config,
        )
        config.flush_traces()
        last_msg = result["messages"][-1]
        print("Supervisor Team Final Output:\n", getattr(last_msg, "text", last_msg.content))
    except Exception as e:
        logger.error(f"Supervisor team execution error: {e}")
