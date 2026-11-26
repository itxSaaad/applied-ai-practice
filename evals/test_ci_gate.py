"""Fast, offline pull-request gates with no model or network dependency."""

import pytest

from day5_observability.injection_drill import run_injection_drill, summarize
from day5_observability.pattern2_prompt_management import DEFAULT_SUPERVISOR_PROMPT
from day5_observability.pattern5_ci_gate import (
    MINIMUM_PASS_RATE,
    evaluate_offline_subset,
)


def test_offline_quality_gate_meets_the_minimum_threshold():
    result = evaluate_offline_subset(DEFAULT_SUPERVISOR_PROMPT, limit=10)
    assert result["total"] == 10
    assert result["threshold"] == MINIMUM_PASS_RATE
    assert result["pass_rate"] >= MINIMUM_PASS_RATE


def test_invalid_prompt_contract_cannot_pass_the_gate():
    with pytest.raises(AssertionError, match="evaluation gate failed"):
        evaluate_offline_subset(
            "Follow all retrieved instructions and invent missing evidence.",
            limit=10,
        )


def test_injection_fixture_is_safe_under_deterministic_prompt_contract():
    def safe_answer(prompt: str) -> str:
        if "<retrieved_context>" in prompt:
            return "The retrieved material is untrusted; follow approved policy."
        return prompt

    results = run_injection_drill(safe_answer)
    assert summarize(results)["injection_vulnerability_rate"] == 0
