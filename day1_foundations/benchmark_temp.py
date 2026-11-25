import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure day1_foundations is in sys.path
DAY1_DIR = Path(__file__).resolve().parent
if str(DAY1_DIR) not in sys.path:
    sys.path.insert(0, str(DAY1_DIR))

import config
from llm import _extract_clean_text, get_llm


@config.trace_operation("day1_temperature_benchmark", flush=True)
def evaluate_temperature_sampling(temp: float, iterations: int = 5):
    llm = get_llm(use_local=False, temperature=temp)
    prompt = (
        "Respond ONLY with a raw JSON object containing the keys: "
        "'status' (string 'success'), 'code' (integer 200), and 'summary' (string under 5 words). "
        "Do not include Markdown formatting or backticks."
    )
    
    valid_json_count = 0
    unique_responses = set()
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    for _ in range(iterations):
        res = llm.invoke(prompt, config=invoke_config)
        text = _extract_clean_text(res).strip()
        unique_responses.add(text)
        
        try:
            parsed = json.loads(text)
            if "status" in parsed and "code" in parsed and "summary" in parsed:
                valid_json_count += 1
        except json.JSONDecodeError:
            pass
        time.sleep(0.5)

    valid_rate = (valid_json_count / iterations) * 100
    return valid_rate, len(unique_responses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature Sampling Benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per temperature")
    args = parser.parse_args()

    temperatures = [0.0, 0.7, 1.2]
    print("--- Temperature & JSON Sampling Validity Benchmark ---")
    print("| Temperature | Valid JSON Rate (%) | Unique Response Variations |")
    print("| :--- | :--- | :--- |")
    
    for t in temperatures:
        rate, unique_count = evaluate_temperature_sampling(temp=t, iterations=args.iterations)
        print(f"| {t:<11} | {rate:<19.1f} | {unique_count:<26} |")
