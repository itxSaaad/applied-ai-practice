import json
from llm import get_llm

def evaluate_temperature_sampling(temp: float, iterations: int = 10):
    llm = get_llm(use_local=False, temperature=temp)
    prompt = (
        "Respond ONLY with a raw JSON object containing the keys: "
        "'status' (string 'success'), 'code' (integer 200), and 'summary' (string under 5 words). "
        "Do not include Markdown formatting or backticks."
    )
    
    valid_json_count = 0
    unique_responses = set()

    for _ in range(iterations):
        res = llm.invoke(prompt)
        text = str(res.content).strip()
        unique_responses.add(text)
        
        try:
            parsed = json.loads(text)
            if "status" in parsed and "code" in parsed and "summary" in parsed:
                valid_json_count += 1
        except json.JSONDecodeError:
            pass

    valid_rate = (valid_json_count / iterations) * 100
    return valid_rate, len(unique_responses)

if __name__ == "__main__":
    temperatures = [0.0, 0.7, 1.2]
    print("--- Temperature & JSON Sampling Validity Benchmark ---")
    print("| Temperature | Valid JSON Rate (%) | Unique Response Variations |")
    print("| :--- | :--- | :--- |")
    
    for t in temperatures:
        rate, unique_count = evaluate_temperature_sampling(temp=t, iterations=10)
        print(f"| {t:<11} | {rate:<19.1f} | {unique_count:<26} |")