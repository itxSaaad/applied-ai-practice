"""Provider-free, 10-example quality gate for pull requests."""

import argparse
import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import config
from day5_observability.pattern2_prompt_management import DEFAULT_SUPERVISOR_PROMPT

MINIMUM_PASS_RATE = 80.0
REQUIRED_PROMPT_CONCEPTS = {
    "untrusted": ("untrusted",),
    "evidence": ("evidence",),
    "citations": ("cite", "citation"),
}
STOP_WORDS = {
    "what",
    "does",
    "should",
    "how",
    "the",
    "for",
    "from",
    "with",
    "and",
    "are",
    "is",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "we",
    "do",
    "i",
}


def prompt_contract_passes(prompt: str) -> bool:
    """Require evidence, citation, and untrusted-content controls in the prompt."""
    normalized = prompt.lower()
    return all(any(term in normalized for term in alternatives) for alternatives in REQUIRED_PROMPT_CONCEPTS.values())


def evaluate_offline_subset(
    prompt: str = DEFAULT_SUPERVISOR_PROMPT,
    *,
    limit: int = 10,
) -> dict[str, float | int]:
    """Measure expected-source retrieval on the first N shared QA examples."""
    if limit <= 0:
        raise ValueError("Evaluation subset size must be positive.")
    with (config.DATA_DIR / "docs_qa.csv").open(encoding="utf-8", newline="") as file:
        cases = list(csv.DictReader(file))[:limit]
    if len(cases) != limit:
        raise ValueError(f"Expected {limit} evaluation rows; found {len(cases)}.")

    documents = []
    for path in sorted(config.RUNBOOKS_DIR.glob("*.md")):
        documents.append(
            (
                path.relative_to(config.ROOT_DIR).as_posix(),
                path.read_text(encoding="utf-8"),
            )
        )
    prompt_valid = prompt_contract_passes(prompt)
    passed = 0
    for case in cases:
        terms = {
            term
            for term in re.findall(r"\b[a-z0-9]+\b", case["question"].lower())
            if len(term) > 2 and term not in STOP_WORDS
        }
        ranked = sorted(
            documents,
            key=lambda item: len(terms & set(re.findall(r"\b[a-z0-9]+\b", item[1].lower()))),
            reverse=True,
        )
        source_retrieved = any(source == case["expected_source"] for source, _ in ranked[:4])
        passed += int(prompt_valid and source_retrieved)

    pass_rate = passed / len(cases) * 100
    result = {
        "passed": passed,
        "total": len(cases),
        "pass_rate": pass_rate,
        "threshold": MINIMUM_PASS_RATE,
        "prompt_contract_passed": int(prompt_valid),
    }
    if pass_rate < MINIMUM_PASS_RATE:
        raise AssertionError(f"Offline evaluation gate failed: {pass_rate:.1f}% is below " f"{MINIMUM_PASS_RATE:.1f}%.")
    config.log_metric(
        "offline_eval_pass_rate",
        pass_rate,
        passed=passed,
        total=len(cases),
        threshold=MINIMUM_PASS_RATE,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=config.DATA_DIR / "prompts" / "ops_pilot_supervisor.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.BENCHMARKS_DIR / "eval_summary.json",
    )
    args = parser.parse_args()
    prompt = args.prompt.read_text(encoding="utf-8")
    result = evaluate_offline_subset(prompt, limit=args.limit)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": config.COURSE_RUN_ID,
        "course_day": "day5",
        "source_file": "day5_observability/pattern5_ci_gate.py",
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f"{args.output.name}.{uuid4().hex}.tmp")
    temporary_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(args.output)
    print(
        f"Offline evaluation: {result['passed']}/{result['total']} "
        f"({result['pass_rate']:.1f}%; threshold {result['threshold']:.1f}%)"
    )
    print(f"Evaluation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
