import argparse
import csv
import time

import numpy as np

import config
from llm import _extract_clean_text, get_llm


def _extract_provider_token_usage(message) -> tuple[int, int] | None:
    """Read exact usage counts from LangChain or provider response metadata."""
    usage_metadata = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    input_tokens = usage_metadata.get(
        "input_tokens",
        response_metadata.get("prompt_eval_count"),
    )
    output_tokens = usage_metadata.get(
        "output_tokens",
        response_metadata.get("eval_count"),
    )
    if input_tokens is None or output_tokens is None:
        return None
    in_count, out_count = int(input_tokens), int(output_tokens)
    if in_count == 0 and out_count == 0:
        return None
    return in_count, out_count


@config.trace_operation("day1_model_profile", flush=True)
def profile_model(use_local: bool, prompt: str, runs: int) -> None:
    if runs <= 0:
        raise ValueError("runs must be a positive integer.")
    model_label = f"Ollama Local ({config.LOCAL_MODEL})" if use_local else f"Gemini Cloud ({config.LLM_MODEL})"
    llm = get_llm(use_local=use_local)

    ttfts = []
    latencies = []
    input_tokens_list = []
    output_tokens_list = []
    records = []

    stream_config = config.get_run_config(
        model=config.LOCAL_MODEL if use_local else config.LLM_MODEL,
        provider="local-ollama" if use_local else "cloud-gemini",
    )

    print(f"Starting profiling: {model_label} across {runs} iterations...")

    for i in range(runs):
        start_time = time.perf_counter()
        first_token_time = None
        full_response = ""
        provider_usage = None

        # Stream response to measure Time to First Token (TTFT)
        stream = llm.stream(prompt, config=stream_config)
        last_chunk = None
        for chunk in stream:
            chunk_usage = _extract_provider_token_usage(chunk)
            if chunk_usage is not None:
                provider_usage = chunk_usage
            chunk_text = _extract_clean_text(chunk)
            if first_token_time is None and chunk_text:
                first_token_time = time.perf_counter()
            full_response += chunk_text
            last_chunk = chunk

        end_time = time.perf_counter()

        if first_token_time is None or not full_response:
            raise RuntimeError(
                f"{model_label} returned no text for iteration {i + 1}; " "latency and token metrics would be invalid."
            )
        ttft = (first_token_time - start_time) * 1000
        total_latency = (end_time - start_time) * 1000

        if provider_usage is None:
            provider_usage = _extract_provider_token_usage(last_chunk)
        if provider_usage is not None:
            in_tokens, out_tokens = provider_usage
            token_source = "provider_usage"
        else:
            try:
                in_tokens = llm.get_num_tokens(prompt)
                out_tokens = llm.get_num_tokens(full_response)
            except ImportError as error:
                raise RuntimeError(
                    "The provider returned no token-usage metadata, and the "
                    "tokenizer fallback is unavailable; accurate token metrics "
                    "cannot be recorded."
                ) from error
            token_source = "tokenizer_estimate"

        ttfts.append(ttft)
        latencies.append(total_latency)
        input_tokens_list.append(in_tokens)
        output_tokens_list.append(out_tokens)

        records.append(
            {
                "iteration": i + 1,
                "model": model_label,
                "ttft_ms": round(ttft, 2),
                "total_latency_ms": round(total_latency, 2),
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "token_source": token_source,
            }
        )
        config.log_metric(
            "model_request_latency_ms",
            round(total_latency, 2),
            model=model_label,
            iteration=i + 1,
            ttft_ms=round(ttft, 2),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            token_source=token_source,
        )

    output_path = config.BENCHMARKS_DIR / f"profile_{'local' if use_local else 'cloud'}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    summary = [
        f"Model: {model_label}",
        f"Runs: {runs}",
        f"TTFT p50/p95: {np.percentile(ttfts, 50):.2f}/{np.percentile(ttfts, 95):.2f} ms",
        f"Total latency p50/p95: {np.percentile(latencies, 50):.2f}/{np.percentile(latencies, 95):.2f} ms",
        f"Mean tokens in/out: {np.mean(input_tokens_list):.1f}/{np.mean(output_tokens_list):.1f}",
    ]
    summary_path = config.BENCHMARKS_DIR / f"profile_{'local' if use_local else 'cloud'}_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    for line in summary:
        print(line)
    config.log_metric(
        "model_call_count",
        len(records),
        model=model_label,
        provider="local-ollama" if use_local else "cloud-gemini",
    )
    print(f"CSV: {output_path}")
    print(f"Summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM Latency & Token Profiler")
    parser.add_argument("--n", type=int, default=20, help="Number of benchmark iterations")
    provider = parser.add_mutually_exclusive_group()
    provider.add_argument(
        "--local",
        dest="provider",
        action="store_const",
        const="local",
        help="Profile only the local Ollama model.",
    )
    provider.add_argument(
        "--cloud",
        dest="provider",
        action="store_const",
        const="cloud",
        help="Profile only the configured cloud model.",
    )
    provider.add_argument(
        "--both",
        dest="provider",
        action="store_const",
        const="both",
        help="Profile both providers (the default).",
    )
    parser.set_defaults(provider="both")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    benchmark_prompt = (
        "Explain the architectural differences between vector search and keyword search in three concise paragraphs."
    )
    if args.n <= 0:
        parser.error("--n must be a positive integer.")
    providers = {
        "local": (True,),
        "cloud": (False,),
        "both": (True, False),
    }[args.provider]
    for use_local in providers:
        profile_model(use_local=use_local, prompt=benchmark_prompt, runs=args.n)
