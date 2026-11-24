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
logger = logging.getLogger("CRAG")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", config.LLM_MODEL),
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0,
)


class CRAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[str]
    question: str
    rewrite_count: int
    grounded: bool


class GradeDocument(BaseModel):
    relevant: bool = Field(description="True if document is relevant to the question.")


class CheckGrounding(BaseModel):
    grounded: bool = Field(description="True if the answer is directly supported by retrieved context.")


def route_node(state: CRAGState) -> Command[Literal["retrieve_node", "direct_answer_node"]]:
    """Routes to retrieval or direct answering using Command routing."""
    last_msg = state["messages"][-1]
    last_message = getattr(last_msg, "text", last_msg.content)
    if isinstance(last_message, list):
        last_message = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in last_message])
    last_message = str(last_message)

    if any(k in last_message.lower() for k in ["search", "lookup", "runbook"]):
        return Command(goto="retrieve_node", update={"question": last_message})
    return Command(goto="direct_answer_node")


def retrieve_node(state: CRAGState) -> dict:
    """Simulated vector retrieval step returning context chunks."""
    query = state.get("question", "")
    retrieved_chunks = [
        f"Retrieved context chunk for query: {query}",
        "System documentation: Runbook step 1 - restart service on error 500.",
    ]
    return {"documents": retrieved_chunks}


def grade_node(state: CRAGState) -> Command[Literal["generate_node", "rewrite_node"]]:
    """Grades retrieved documents; routes to rewrite if irrelevant and under rewrite limit."""
    docs = state.get("documents", [])
    rewrite_count = state.get("rewrite_count", 0)

    relevant_docs = []
    try:
        grader = llm.with_structured_output(GradeDocument)
        for doc in docs:
            res = grader.invoke([
                SystemMessage(content="Grade relevance of doc to question. Return boolean."),
                HumanMessage(content=f"Question: {state['question']}\nDoc: {doc}"),
            ])
            if res and res.relevant:
                relevant_docs.append(doc)
    except Exception as e:
        logger.warning(f"Grading fallback applied: {e}")
        relevant_docs = docs

    if relevant_docs:
        return Command(goto="generate_node", update={"documents": relevant_docs})

    if rewrite_count < 2:
        return Command(goto="rewrite_node", update={"rewrite_count": rewrite_count + 1})

    return Command(goto="generate_node", update={"documents": ["No relevant context found after rewrites."]})


def rewrite_node(state: CRAGState) -> dict:
    """Reformulates the search query for vector retrieval."""
    prompt = f"Rewrite this query for better semantic vector retrieval: {state['question']}"
    res = llm.invoke([HumanMessage(content=prompt)])
    res_text = getattr(res, "text", res.content)
    return {"question": str(res_text)}


def generate_node(state: CRAGState) -> dict:
    docs_text = "\n".join(state.get("documents", []))
    prompt = f"Answer question based strictly on context:\nContext:\n{docs_text}\n\nQuestion: {state['question']}"
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [res]}


def check_grounding_node(state: CRAGState) -> Command[Literal[END, "generate_node"]]:
    """Verifies that the generated response is grounded in the source text."""
    last_msg = state["messages"][-1]
    last_response = getattr(last_msg, "text", last_msg.content)
    docs_text = "\n".join(state.get("documents", []))

    try:
        grounding_checker = llm.with_structured_output(CheckGrounding)
        res = grounding_checker.invoke([
            SystemMessage(content="Verify if output is supported by context."),
            HumanMessage(content=f"Context: {docs_text}\nResponse: {last_response}"),
        ])
        is_grounded = res.grounded if res else True
    except Exception as e:
        logger.warning(f"Grounding check fallback applied: {e}")
        is_grounded = True

    if is_grounded or state.get("grounded", False):
        return Command(goto=END, update={"grounded": True})

    return Command(goto="generate_node", update={"grounded": True})


def direct_answer_node(state: CRAGState) -> dict:
    res = llm.invoke(state["messages"])
    return {"messages": [res]}


builder = StateGraph(CRAGState)
builder.add_node("route_node", route_node)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("grade_node", grade_node)
builder.add_node("rewrite_node", rewrite_node)
builder.add_node("generate_node", generate_node)
builder.add_node("check_grounding_node", check_grounding_node)
builder.add_node("direct_answer_node", direct_answer_node)

builder.add_edge(START, "route_node")
builder.add_edge("retrieve_node", "grade_node")
builder.add_edge("rewrite_node", "retrieve_node")
builder.add_edge("generate_node", "check_grounding_node")
builder.add_edge("direct_answer_node", END)

crag_graph = builder.compile()


if __name__ == "__main__":
    callbacks = config.get_callbacks()
    invoke_config = {"recursion_limit": 15}
    if callbacks:
        invoke_config["callbacks"] = callbacks

    try:
        result = crag_graph.invoke(
            {
                "messages": [HumanMessage(content="Lookup the runbook steps for error 500.")],
                "documents": [],
                "question": "",
                "rewrite_count": 0,
                "grounded": False,
            },
            config=invoke_config,
        )
        config.flush_traces()
        last_msg = result["messages"][-1]
        print("CRAG Final Output:", getattr(last_msg, "text", last_msg.content))
    except Exception as e:
        logger.error(f"CRAG execution error: {e}")
