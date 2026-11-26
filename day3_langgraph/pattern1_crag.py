from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

import config
from day2_langchain.pattern5_rag_agent import retrieve_runbook_documents
from llm import get_llm

llm = None


def _get_model():
    return llm if llm is not None else get_llm()


class CRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[str]
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
    """Retrieve source-tagged runbook passages from the shared persistent index."""
    query = state.get("question", "")
    documents = retrieve_runbook_documents(query, k=4)
    retrieved_chunks = [
        f"Source: {document.metadata['source_url']} | "
        f"Section: {document.metadata['section']}\n{document.page_content}"
        for document in documents
    ]
    return {"documents": retrieved_chunks}


def grade_node(state: CRAGState) -> Command[Literal["generate_node", "rewrite_node"]]:
    """Grades retrieved documents; routes to rewrite if irrelevant and under rewrite limit."""
    docs = state.get("documents", [])
    rewrite_count = state.get("rewrite_count", 0)

    grader = _get_model().with_structured_output(GradeDocument)
    relevant_docs = []
    for doc in docs:
        result = grader.invoke(
            [
                SystemMessage(
                    content="Grade whether the document is relevant to the question. "
                    "Return a boolean and do not infer missing facts."
                ),
                HumanMessage(content=f"Question: {state['question']}\nDocument:\n{doc}"),
            ],
            config=config.get_run_config(prompt_version="crag-grade-v1"),
        )
        if result and result.relevant:
            relevant_docs.append(doc)

    if relevant_docs:
        return Command(goto="generate_node", update={"documents": relevant_docs})

    if rewrite_count < 2:
        return Command(goto="rewrite_node", update={"rewrite_count": rewrite_count + 1})

    return Command(goto="generate_node", update={"documents": ["No relevant context found after rewrites."]})


def rewrite_node(state: CRAGState) -> dict:
    """Reformulates the search query for vector retrieval."""
    prompt = f"Rewrite this query for better semantic vector retrieval: {state['question']}"
    res = _get_model().invoke(
        [HumanMessage(content=prompt)],
        config=config.get_run_config(prompt_version="crag-rewrite-v1"),
    )
    res_text = getattr(res, "text", res.content)
    return {"question": str(res_text)}


def generate_node(state: CRAGState) -> dict:
    docs_text = "\n".join(state.get("documents", []))
    prompt = (
        "Answer using only the supplied context. Cite each source using its Source label. "
        "If the context is insufficient, say so.\n"
        f"Context:\n{docs_text}\n\nQuestion: {state['question']}"
    )
    res = _get_model().invoke(
        [HumanMessage(content=prompt)],
        config=config.get_run_config(prompt_version="crag-answer-v1"),
    )
    return {"messages": [res]}


def check_grounding_node(state: CRAGState) -> Command[Literal[END, "generate_node"]]:
    """Verifies that the generated response is grounded in the source text."""
    last_msg = state["messages"][-1]
    last_response = getattr(last_msg, "text", last_msg.content)
    docs_text = "\n".join(state.get("documents", []))

    grounding_checker = _get_model().with_structured_output(CheckGrounding)
    res = grounding_checker.invoke(
        [
            SystemMessage(content="Verify if output is supported by context."),
            HumanMessage(content=f"Context: {docs_text}\nResponse: {last_response}"),
        ],
        config=config.get_run_config(prompt_version="crag-grounding-v1"),
    )
    is_grounded = res.grounded if res else False

    if is_grounded or state.get("grounded", False):
        return Command(goto=END, update={"grounded": True})

    return Command(goto="generate_node", update={"grounded": True})


def direct_answer_node(state: CRAGState) -> dict:
    res = _get_model().invoke(
        state["messages"],
        config=config.get_run_config(prompt_version="crag-direct-v1"),
    )
    return {"messages": [res]}


def build_crag_graph(checkpointer=None):
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
    return builder.compile(checkpointer=checkpointer)


crag_graph = build_crag_graph()


if __name__ == "__main__":
    thread_id = f"crag-{uuid4().hex}"
    invoke_config = {
        "recursion_limit": 15,
        "configurable": {"thread_id": thread_id},
        **config.get_run_config(session_id=thread_id),
    }

    with SqliteSaver.from_conn_string(config.CRAG_CHECKPOINTS_DB_PATH) as checkpointer:
        durable_graph = build_crag_graph(checkpointer=checkpointer)
        result = durable_graph.invoke(
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
