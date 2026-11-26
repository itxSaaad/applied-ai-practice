"""Compare document chunk sizes against the shared 25-question retrieval set."""

import csv
import time

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_runbooks() -> list[Document]:
    runbook_dir = config.RUNBOOKS_DIR
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.relative_to(config.ROOT_DIR).as_posix()},
        )
        for path in sorted(runbook_dir.glob("*.md"))
    ]


def load_questions() -> list[dict[str, str]]:
    with (config.DATA_DIR / "docs_qa.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def evaluate_setting(
    documents: list[Document],
    questions: list[dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, float | int]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The runbook corpus produced no chunks.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=config.OLLAMA_BASE_URL,
        ),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    hits = 0
    started = time.perf_counter()
    try:
        for question in questions:
            results = retriever.invoke(question["question"])
            if any(doc.metadata["source"] == question["expected_source"] for doc in results):
                hits += 1
    finally:
        vector_store.delete_collection()
    latency_ms = (time.perf_counter() - started) * 1000 / len(questions)
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
        "hit_rate_at_4": hits / len(questions),
        "average_query_latency_ms": latency_ms,
    }


@config.trace_operation("day2_chunking_study", flush=True)
def run_study() -> list[dict[str, float | int]]:
    documents = load_runbooks()
    questions = load_questions()
    if len(questions) != 25:
        raise ValueError(f"Expected 25 evaluation questions; found {len(questions)}.")

    results = [
        evaluate_setting(documents, questions, chunk_size, overlap)
        for chunk_size in (250, 500, 1000)
        for overlap in (0, 100)
        if overlap < chunk_size
    ]
    for row in results:
        config.log_metric(
            "chunking_hit_rate_at_4",
            float(row["hit_rate_at_4"]),
            chunk_size=row["chunk_size"],
            chunk_overlap=row["chunk_overlap"],
            chunk_count=row["chunk_count"],
            average_query_latency_ms=round(float(row["average_query_latency_ms"]), 2),
        )
    output = config.BENCHMARKS_DIR / "chunking_study_results.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


if __name__ == "__main__":
    for row in run_study():
        print(
            f"size={row['chunk_size']} overlap={row['chunk_overlap']} "
            f"chunks={row['chunk_count']} hit@4={row['hit_rate_at_4']:.1%} "
            f"avg_ms={row['average_query_latency_ms']:.1f}"
        )
    print(f"Results written to {config.BENCHMARKS_DIR / 'chunking_study_results.csv'}")
