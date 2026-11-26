"""Compare prompt versions against the same LangSmith dataset."""

from collections.abc import Callable, Mapping

from day5_observability.pattern3_evaluators import run_benchmark_experiment


def compare_prompt_versions(
    dataset_name: str,
    targets: Mapping[str, Callable],
) -> dict[str, object]:
    if len(targets) < 2:
        raise ValueError("At least two named prompt versions are required.")
    return {version: run_benchmark_experiment(dataset_name, target) for version, target in targets.items()}
