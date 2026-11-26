import time

import chromadb
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from utils.logging import get_logger

logger = get_logger(__name__)


def load_project_corpus() -> list[str]:
    source_files = [
        config.ROOT_DIR / "README.md",
        *sorted(config.RUNBOOKS_DIR.glob("*.md")),
        *sorted((config.ROOT_DIR / "day1_foundations").glob("*.py")),
        *sorted((config.ROOT_DIR / "day2_langchain").glob("*.py")),
        *sorted((config.ROOT_DIR / "day3_langgraph").glob("*.py")),
        *sorted((config.ROOT_DIR / "day4_adk").glob("*.py")),
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=260, chunk_overlap=20)
    corpus = [chunk for path in source_files for chunk in splitter.split_text(path.read_text(encoding="utf-8"))]
    if not corpus:
        raise ValueError("No project documentation or code was available to index.")
    logger.info(
        "Indexing %s project chunks from %s source files.",
        len(corpus),
        len(source_files),
    )
    return corpus


@config.trace_operation("day1_semantic_search_benchmark")
def run_benchmark() -> dict[str, float]:
    corpus = load_project_corpus()
    query = "How do continuous dense vectors measure semantic distance?"

    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=config.OLLAMA_BASE_URL,
    )
    corpus_embeddings = embeddings_model.embed_documents(corpus)
    query_embedding = embeddings_model.embed_query(query)

    started = time.perf_counter()
    query_vector = np.array(query_embedding)
    corpus_vectors = np.array(corpus_embeddings)
    dot_products = np.dot(corpus_vectors, query_vector)
    norms_corpus = np.linalg.norm(corpus_vectors, axis=1)
    norm_query = np.linalg.norm(query_vector)
    denominator = norms_corpus * norm_query
    similarities = np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products),
        where=denominator != 0,
    )
    numpy_top_5_idx = np.argsort(similarities)[::-1][:5]
    numpy_latency = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name=f"benchmark_{time.time_ns()}")
    collection.add(
        documents=corpus,
        embeddings=corpus_embeddings,
        ids=[f"doc_{i}" for i in range(len(corpus))],
    )
    chroma_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )
    chroma_latency = (time.perf_counter() - started) * 1000

    numpy_top_texts = {corpus[index] for index in numpy_top_5_idx}
    chroma_top_texts = set(chroma_results["documents"][0])
    recall_at_5 = len(numpy_top_texts.intersection(chroma_top_texts)) / 5.0
    metrics = {
        "numpy_latency_ms": numpy_latency,
        "chroma_latency_ms": chroma_latency,
        "recall_at_5": recall_at_5,
    }

    config.log_metric(
        "semantic_search_recall_at_5",
        recall_at_5,
        corpus_chunks=len(corpus),
        numpy_latency_ms=round(numpy_latency, 2),
        chroma_latency_ms=round(chroma_latency, 2),
    )
    return metrics


if __name__ == "__main__":
    try:
        metrics = run_benchmark()
        print("\n--- Semantic Search Evaluation Results ---")
        print(f"NumPy Brute Force Latency: {metrics['numpy_latency_ms']:.2f} ms")
        print(f"Chroma Indexing & Query Latency: {metrics['chroma_latency_ms']:.2f} ms")
        print(f"Recall@5 Alignment Rate: {metrics['recall_at_5'] * 100:.1f}%")
    finally:
        config.flush_traces()
