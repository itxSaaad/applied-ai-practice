"""Knowledge Agent: Corrective RAG LangGraph subgraph over OpsPilot runbooks."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document  # noqa: TC002
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
from capstone.seed import load_runbook_documents
from llm import get_llm

logger = config.get_logger(__name__)


class KnowledgeState(TypedDict):
    question: str
    rewritten_question: str
    documents: Annotated[list[Document], operator.add]
    citations: list[str]
    grade: str
    rewrite_count: int
    answer: str


class GradeResult(BaseModel):
    relevant: Literal["yes", "no"] = Field(description="Whether retrieved chunks answer the question.")
    reason: str = Field(description="Short justification.")


def lexical_retrieve(query: str, k: int = 4) -> list[Document]:
    """Deterministic keyword-overlap retriever over tracked runbooks (offline fallback)."""
    terms = {t.lower().strip("?,.") for t in query.split() if len(t) > 2}
    docs = load_runbook_documents()
    scored: list[tuple[int, Document]] = []
    for doc in docs:
        haystack = f"{doc.metadata.get('source', '')} {doc.page_content}".lower()
        score = sum(1 for t in terms if t in haystack)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:k]] or docs[:k]


def build_knowledge_agent(llm=None, retriever=None):
    """Build the Corrective RAG subgraph (retrieve -> grade -> rewrite/generate)."""
    active_retriever = retriever or lexical_retrieve

    def _model():
        return llm or get_llm()

    def retrieve_node(state: KnowledgeState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        docs = active_retriever(query)
        citations = sorted({str(d.metadata.get("source", "unknown")) for d in docs})
        return {"documents": docs, "citations": citations}

    def grade_node(state: KnowledgeState) -> dict:
        docs = state.get("documents", [])
        if not docs:
            return {"grade": "no"}
        if llm is None:
            # Fast lexical check when no custom/fake LLM is injected in offline tests
            terms = {t.lower().strip("?,.") for t in state["question"].split() if len(t) > 3}
            joined = " ".join(d.page_content.lower() for d in docs)
            hit = any(t in joined for t in terms)
            return {"grade": "yes" if hit else "no"}
        grader = _model().with_structured_output(GradeResult)
        context = "\n\n".join(d.page_content for d in docs[:4])
        res = grader.invoke(
            f"Question: {state['question']}\n\nContext:\n{context}\n\nIs the context relevant?",
            config=config.get_run_config(model=config.LLM_MODEL),
        )
        return {"grade": res.relevant}

    def rewrite_node(state: KnowledgeState) -> dict:
        count = state.get("rewrite_count", 0) + 1
        if llm is None:
            return {"rewritten_question": f"runbook procedure {state['question']}", "rewrite_count": count}
        msg = _model().invoke(
            f"Rewrite this on-call search query for better runbook retrieval: {state['question']}",
            config=config.get_run_config(model=config.LLM_MODEL),
        )
        return {"rewritten_question": str(msg.content).strip(), "rewrite_count": count}

    def generate_node(state: KnowledgeState) -> dict:
        docs = state.get("documents", [])[:4]
        citations = sorted({str(d.metadata.get("source", "unknown")) for d in docs})
        context = "\n\n".join(f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs)
        config.log_metric("opspilot_knowledge_docs", float(len(docs)), rewrite_count=state.get("rewrite_count", 0))
        if llm is None:
            summary = docs[0].page_content.strip().splitlines()[0] if docs else "No matching runbook found."
            return {
                "citations": citations,
                "answer": f"{summary} (Sources: {', '.join(citations)})",
            }
        resp = _model().invoke(
            f"Answer the on-call question strictly from the runbook context below and cite sources.\n\n"
            f"Question: {state['question']}\n\nContext:\n{context}",
            config=config.get_run_config(model=config.LLM_MODEL),
        )
        return {"citations": citations, "answer": str(resp.content).strip()}

    def route_after_grade(state: KnowledgeState) -> str:
        if state.get("grade") == "yes" or state.get("rewrite_count", 0) >= 2:
            return "generate"
        return "rewrite"

    graph = StateGraph(KnowledgeState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"rewrite": "rewrite", "generate": "generate"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
