import logging
import operator
import os
import sys
from pathlib import Path
from typing import List, TypedDict
from typing_extensions import Annotated
from dotenv import load_dotenv

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MapReduce")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", config.LLM_MODEL),
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0,
)


class OverallState(TypedDict):
    documents: List[str]
    summaries: Annotated[List[str], operator.add]
    final_brief: str


class MapTaskState(TypedDict):
    document: str


def dynamic_map_fanout(state: OverallState) -> List[Send]:
    """Maps dynamic list of input documents into Send objects targeting worker nodes."""
    return [Send("summarize_doc_node", {"document": doc}) for doc in state["documents"]]


def summarize_doc_node(state: MapTaskState) -> dict:
    """Worker map task processing single isolated payload item."""
    prompt = f"Summarize the following incident log snippet concisely:\n{state['document']}"
    res = llm.invoke([HumanMessage(content=prompt)])
    res_text = getattr(res, "text", res.content)
    return {"summaries": [str(res_text)]}


def reduce_summaries_node(state: OverallState) -> dict:
    """Aggregate step compiling all parallel map outputs into unified brief."""
    combined_summaries = "\n\n".join(state.get("summaries", []))
    prompt = f"Compile these incident summaries into one structured brief:\n{combined_summaries}"
    res = llm.invoke([HumanMessage(content=prompt)])
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
    sample_logs = [
        "08:12 UTC - auth-api CPU spiked to 98% following token cache eviction.",
        "08:19 UTC - checkout-service reported elevated 504 latency waiting on auth-api.",
        "08:27 UTC - Auto-scaler added 4 replicas and cache hit ratio recovered to 99%.",
    ]
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    try:
        result = map_reduce_graph.invoke(
            {"documents": sample_logs, "summaries": [], "final_brief": ""},
            config=invoke_config,
        )
        config.flush_traces()
        print("Compiled Incident Brief:\n", result["final_brief"])
    except Exception as e:
        logger.error(f"Map-reduce execution error: {e}")
