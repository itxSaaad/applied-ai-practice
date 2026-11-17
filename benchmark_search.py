import time
import numpy as np
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Corpus initialization
corpus = [
    "PostgreSQL supports relational schemas, foreign keys, and strict ACID compliance.",
    "Chroma is an open-source file-based local vector store optimized for embeddings.",
    "Self-attention allows transformer models to weight relations across sequence tokens.",
    "Ollama enables local execution of open weights models like Qwen3 and Llama3.",
    "Cosine similarity measures the angular distance between two continuous dense vectors."
] + [f"Synthetic filler document chunk number {i} covering general operational topics." for i in range(495)]

query = "How do continuous dense vectors measure semantic distance?"

print("Generating embeddings via local Ollama (nomic-embed-text)...")
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

corpus_embeddings = embeddings_model.embed_documents(corpus)
query_embedding = embeddings_model.embed_query(query)

# 2. Brute-Force Cosine Search via Pure NumPy
t0 = time.perf_counter()
A = np.array(query_embedding)
B = np.array(corpus_embeddings)

dot_products = np.dot(B, A)
norms_B = np.linalg.norm(B, axis=1)
norm_A = np.linalg.norm(A)
numpy_similarities = dot_products / (norms_B * norm_A)

numpy_top_5_idx = np.argsort(numpy_similarities)[::-1][:5]
t1 = time.perf_counter()
numpy_latency = (t1 - t0) * 1000

# 3. Ephemeral Vector Database Search via Chroma
t2 = time.perf_counter()
ephemeral_client = chromadb.EphemeralClient()
vector_db = Chroma.from_texts(
    texts=corpus,
    embedding=embeddings_model,
    client=ephemeral_client,
    collection_name="benchmark_coll"
)
chroma_results = vector_db.similarity_search_with_score(query, k=5)
t3 = time.perf_counter()
chroma_latency = (t3 - t2) * 1000

# 4. Recall Evaluation
numpy_top_texts = set([corpus[idx] for idx in numpy_top_5_idx])
chroma_top_texts = set([doc.page_content for doc, _ in chroma_results])

intersection = numpy_top_texts.intersection(chroma_top_texts)
recall_at_5 = len(intersection) / 5.0

print(f"\n--- Semantic Search Evaluation Results ---")
print(f"NumPy Brute Force Latency: {numpy_latency:.2f} ms")
print(f"Chroma HNSW Search Latency: {chroma_latency:.2f} ms")
print(f"Recall@5 Alignment Rate: {recall_at_5 * 100:.1f}%")