import argparse
import csv
import json
import time

import config
from llm import _extract_clean_text, get_llm


@config.trace_operation("day1_temperature_benchmark", flush=True)
def evaluate_temperature_sampling(temp: float, iterations: int = 5):
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer.")
    llm = get_llm(use_local=False, temperature=temp)
    prompt = (
        "Respond ONLY with a raw JSON object containing the keys: "
        "'status' (string 'success'), 'code' (integer 200), and 'summary' (string under 5 words). "
        "Do not include Markdown formatting or backticks."
    )

    valid_json_count = 0
    unique_responses = set()
    invoke_config = config.get_run_config()

    for _ in range(iterations):
        res = llm.invoke(prompt, config=invoke_config)
        text = _extract_clean_text(res).strip()
        unique_responses.add(text)

        try:
            parsed = json.loads(text)
            if (
                isinstance(parsed, dict)
                and parsed.get("status") == "success"
                and parsed.get("code") == 200
                and isinstance(parsed.get("summary"), str)
            ):
                valid_json_count += 1
        except json.JSONDecodeError:
            pass
        time.sleep(0.5)

    valid_rate = (valid_json_count / iterations) * 100
    return valid_rate, len(unique_responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature Sampling Benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per temperature")
    args = parser.parse_args()

    temperatures = [0.0, 0.7, 1.2]
    results = []
    print("--- Temperature & JSON Sampling Validity Benchmark ---")
    print("| Temperature | Valid JSON Rate (%) | Unique Response Variations |")
    print("| :--- | :--- | :--- |")

    for t in temperatures:
        rate, unique_count = evaluate_temperature_sampling(temp=t, iterations=args.iterations)
        print(f"| {t:<11} | {rate:<19.1f} | {unique_count:<26} |")
        results.append(
            {
                "temperature": t,
                "iterations": args.iterations,
                "valid_json_rate_percent": rate,
                "unique_answers": unique_count,
            }
        )
    print(
        "\nConclusion: Temperature changes the variety of generated answers, while "
        "temperature zero can still produce provider-level variation. "
        "Choose a setting from both the measured JSON-valid rate and distinct-answer count."
    )
    results_path = config.BENCHMARKS_DIR / "temperature_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Measurements written to {results_path}")
