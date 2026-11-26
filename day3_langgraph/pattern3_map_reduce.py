import json
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

import config
from llm import get_llm
from utils.logging import get_logger

logger = get_logger(__name__)

llm = None


def _get_model():
    return llm if llm is not None else get_llm()


class OverallState(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], operator.add]
    final_brief: str


class MapTaskState(TypedDict):
    document: str


def dynamic_map_fanout(state: OverallState) -> list[Send]:
    """Maps dynamic list of input documents into Send objects targeting worker nodes."""
    return [Send("summarize_doc_node", {"document": doc}) for doc in state["documents"]]


def summarize_doc_node(state: MapTaskState) -> dict:
    """Worker map task processing single isolated payload item."""
    prompt = f"Summarize the following incident log snippet concisely:\n{state['document']}"
    res = _get_model().invoke([HumanMessage(content=prompt)])
    res_text = getattr(res, "text", res.content)
    return {"summaries": [str(res_text)]}


def reduce_summaries_node(state: OverallState) -> dict:
    """Aggregate step compiling all parallel map outputs into unified brief."""
    combined_summaries = "\n\n".join(state.get("summaries", []))
    prompt = f"Compile these incident summaries into one structured brief:\n{combined_summaries}"
    res = _get_model().invoke([HumanMessage(content=prompt)])
    res_text = getattr(res, "text", res.content)
    return {"final_brief": str(res_text)}


builder = StateGraph(OverallState)
builder.add_node("summarize_doc_node", summarize_doc_node)
builder.add_node("reduce_summaries_node", reduce_summaries_node)

builder.add_conditional_edges(START, dynamic_map_fanout, ["summarize_doc_node"])
builder.add_edge("summarize_doc_node", "reduce_summaries_node")
builder.add_edge("reduce_summaries_node", END)

map_reduce_graph = builder.compile()


if __name__ == "__main__":
    sample_logs = json.loads((config.DATA_DIR / "sample_incident_logs.json").read_text(encoding="utf-8"))
    invoke_config = config.get_run_config()

    try:
        result = map_reduce_graph.invoke(
            {"documents": sample_logs, "summaries": [], "final_brief": ""},
            config=invoke_config,
        )
        config.flush_traces()
        print("Compiled Incident Brief:\n", result["final_brief"])
        print(f"Summarized {len(result['summaries'])} documents in the parallel map.")
    except Exception as e:
        logger.error(f"Map-reduce execution error: {e}")
