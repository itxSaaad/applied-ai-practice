import pytest

from day4_adk.pattern2_critique_loop import refinement_loop
from utils import telemetry


def test_critique_loop_bound():
    assert refinement_loop.max_iterations == 3
    assert len(refinement_loop.sub_agents) == 2


def test_flush_does_not_initialize_tracing_clients(monkeypatch):
    monkeypatch.setattr(telemetry, "_langfuse_client", None)
    monkeypatch.setattr(telemetry, "_langsmith_client", None)
    monkeypatch.setattr(
        telemetry,
        "get_langfuse_client",
        lambda: pytest.fail("flush must not initialize Langfuse at shutdown"),
    )
    monkeypatch.setattr(
        telemetry,
        "get_langsmith_client",
        lambda: pytest.fail("flush must not initialize LangSmith at shutdown"),
    )

    telemetry.flush_traces()
