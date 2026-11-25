import logging
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
from langchain_ollama import OllamaEmbeddings

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

logger = logging.getLogger("SemanticSearchBenchmark")


@config.trace_operation("day1_semantic_search_benchmark")
def run_benchmark() -> dict[str, float]:
    corpus = [
        "PostgreSQL supports relational schemas, foreign keys, and strict ACID compliance.",
        "Chroma is an open-source file-based local vector store optimized for embeddings.",
        "Self-attention allows transformer models to weight relations across sequence tokens.",
        "Ollama enables local execution of open weights models like Qwen3 and Llama3.",
        "Cosine similarity measures the angular distance between two continuous dense vectors.",
    ] + [
        f"Synthetic filler document chunk number {i} covering general operational topics."
        for i in range(495)
    ]
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
    similarities = dot_products / (norms_corpus * norm_query)
    numpy_top_5_idx = np.argsort(similarities)[::-1][:5]
    numpy_latency = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="benchmark_coll")
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

    logger.info(
        "metric benchmark=semantic_search numpy_latency_ms=%.2f "
        "chroma_latency_ms=%.2f recall_at_5=%.3f",
        numpy_latency,
        chroma_latency,
        recall_at_5,
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
