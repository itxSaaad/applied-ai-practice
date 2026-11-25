import logging
import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RagAgent")


# 1. Structured Output Schema
class Citation(BaseModel):
    source_url: str = Field(description="Source URL reference")
    section: str = Field(description="Document section reference")


class Answer(BaseModel):
    answer: str = Field(description="Direct concise answer based strictly on retrieved context")
    citations: List[Citation] = Field(description="List of context citations used")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


# 2. Vector DB Initialization & Ingestion using pinned langchain-ollama package
docs = [
    Document(
        page_content="LangChain v1 agents run on LangGraph under the hood.",
        metadata={"source_url": "https://docs.langchain.com/v1", "section": "architecture"},
    )
]
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=config.OLLAMA_BASE_URL,
    ),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 3. Deterministic LCEL Retrieval Chain
llm = init_chat_model(
    f"google_genai:{os.getenv('LLM_MODEL', config.LLM_MODEL)}",
    temperature=0.0,
)
structured_llm = llm.with_structured_output(Answer)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY context below:\nContext: {context}\nQuestion: {question}"
)


@config.trace_operation("day2_rag_retrieval")
def rag_pipeline(question: str) -> Answer:
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}
    retrieved_docs = retriever.invoke(question)
    context_str = "\n\n".join(
        [f"[URL: {d.metadata['source_url']} | Section: {d.metadata['section']}]\n{d.page_content}" for d in retrieved_docs]
    )
    formatted_prompt = prompt.format(context=context_str, question=question)
    return structured_llm.invoke(formatted_prompt, config=invoke_config)


# 4. Wrap Pipeline as an Agent Tool
@tool
def knowledge_base_tool(query: str) -> str:
    """Queries official documentation to provide verified technical answers with source citations."""
    res = rag_pipeline(query)
    return res.model_dump_json()


# Create Agent incorporating Knowledge Tool
agent = create_agent(
    model=llm,
    tools=[knowledge_base_tool],
    system_prompt="You are a documentation search assistant. Always query the knowledge base tool for technical factual inquiries.",
)


if __name__ == "__main__":
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    try:
        res = agent.invoke(
            {"messages": [("user", "What execution engine do LangChain v1 agents run on?")]},
            config=invoke_config,
        )
        config.flush_traces()
        last_msg = res["messages"][-1]
        print(getattr(last_msg, "text", last_msg.content))
    except Exception as e:
        logger.warning(f"RAG agent test note: {e}")
