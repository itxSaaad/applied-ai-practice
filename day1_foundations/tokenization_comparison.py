"""Compare provider token counts on English, Urdu, and source-code inputs."""

import config
from llm import get_llm

SAMPLES = {
    "english": "A vector index retrieves semantically similar documents.",
    "urdu": "مصنوعی ذہانت میں متن کی نمائندگی کے لیے ایمبیڈنگ استعمال ہوتی ہے۔",
    "code": "def cosine(a, b): return dot(a, b) / (norm(a) * norm(b))",
}


@config.trace_operation("day1_tokenization_comparison")
def compare_token_counts() -> list[dict[str, int | str]]:
    models = {
        "local": get_llm(use_local=True),
        "cloud": get_llm(use_local=False),
    }
    results = []
    for sample_name, text in SAMPLES.items():
        for model_name, model in models.items():
            results.append(
                {
                    "sample": sample_name,
                    "model": model_name,
                    "characters": len(text),
                    "tokens": model.get_num_tokens(text),
                }
            )
    return results


if __name__ == "__main__":
    for result in compare_token_counts():
        print(f"{result['sample']}: {result['model']} " f"{result['characters']} characters, {result['tokens']} tokens")
