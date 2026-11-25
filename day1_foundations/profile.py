import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DAY1_DIR = Path(__file__).resolve().parent

import config
from llm import _extract_clean_text, get_llm


@config.trace_operation("day1_model_profile", flush=True)
def profile_model(use_local: bool, prompt: str, runs: int) -> None:
    model_label = f"Ollama Local ({config.LOCAL_MODEL})" if use_local else f"Gemini Cloud ({config.LLM_MODEL})"
    llm = get_llm(use_local=use_local)
    
    ttfts = []
    latencies = []
    input_tokens_list = []
    output_tokens_list = []
    records = []

    callbacks = config.get_callbacks()
    stream_config = {"callbacks": callbacks} if callbacks else {}

    print(f"Starting profiling: {model_label} across {runs} iterations...")

    for i in range(runs):
        start_time = time.perf_counter()
        first_token_time = None
        full_response = ""
        
        # Stream response to measure Time to First Token (TTFT)
        stream = llm.stream(prompt, config=stream_config)
        last_chunk = None
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            full_response += _extract_clean_text(chunk)
            last_chunk = chunk
            
        end_time = time.perf_counter()
        
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0.0
        total_latency = (end_time - start_time) * 1000
        
        # Parse exact token metrics from returned message metadata
        in_tokens = 0
        out_tokens = 0
        if last_chunk and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
            in_tokens = last_chunk.usage_metadata.get("input_tokens", 0)
            out_tokens = last_chunk.usage_metadata.get("output_tokens", 0)
            
        if in_tokens == 0 or out_tokens == 0:
            try:
                in_tokens = llm.get_num_tokens(prompt)
                out_tokens = llm.get_num_tokens(full_response)
            except Exception:
                in_tokens = len(prompt) // 4
                out_tokens = len(full_response) // 4

        ttfts.append(ttft)
        latencies.append(total_latency)
        input_tokens_list.append(in_tokens)
        output_tokens_list.append(out_tokens)
        
        records.append({
            "iteration": i + 1,
            "model": model_label,
            "ttft_ms": round(ttft, 2),
            "total_latency_ms": round(total_latency, 2),
            "input_tokens": in_tokens,
            "output_tokens": out_tokens
        })

    output_path = DAY1_DIR / f"profile_{'local' if use_local else 'cloud'}.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
        
    print(f"\n--- Summary Statistics: {model_label} ---")
    print(f"Processed Iterations: {runs}")
    print(f"TTFT p50: {np.percentile(ttfts, 50):.2f} ms | p95: {np.percentile(ttfts, 95):.2f} ms")
    print(f"Total Latency p50: {np.percentile(latencies, 50):.2f} ms | p95: {np.percentile(latencies, 95):.2f} ms")
    print(f"Avg Output Tokens: {np.mean(output_tokens_list):.1f}")
    print(f"Raw profiling log written to {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Latency & Token Profiler")
    parser.add_argument("--n", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--local", action="store_true", help="Profile local Ollama model instead of Gemini Cloud")
    args = parser.parse_args()
    
    benchmark_prompt = "Explain the architectural differences between vector search and keyword search in three concise paragraphs."
    profile_model(use_local=args.local, prompt=benchmark_prompt, runs=args.n)
