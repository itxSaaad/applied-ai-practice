"""Offline unit tests for client-side Gemini rate limiting."""

import time

import pytest

from utils import rate_limit


def test_model_rate_limits_match_configured_quotas():
    flash_lite = rate_limit.get_model_rate_limit("gemini-3.5-flash-lite")
    assert flash_lite is not None
    assert (flash_lite.rpm, flash_lite.tpm, flash_lite.rpd) == (15, 250_000, 500)

    flash_38 = rate_limit.get_model_rate_limit("google_genai:gemini-3.8-flash")
    assert flash_38 is not None
    assert (flash_38.rpm, flash_38.tpm, flash_38.rpd) == (5, 250_000, 20)

    assert rate_limit.get_model_rate_limit("ollama:qwen3:8b") is None


def test_sliding_window_limiter_throttles_after_rpm_exhausted(monkeypatch):
    clock = [100.0]
    sleeps: list[float] = []

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(time, "sleep", fake_sleep)

    limiter = rate_limit.SlidingWindowRateLimiter(window_seconds=60.0)
    # gemini-3.8-flash has 5 RPM
    for _ in range(5):
        assert limiter.acquire("gemini-3.8-flash", estimated_tokens=100) == 0.0

    waited = limiter.acquire("gemini-3.8-flash", estimated_tokens=100)
    assert waited >= 60.0
    assert sleeps


def test_sliding_window_limiter_enforces_rpd(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    limiter = rate_limit.SlidingWindowRateLimiter(window_seconds=1.0)
    # gemini-3.8-flash has 20 RPD
    for _ in range(20):
        limiter.acquire("gemini-3.8-flash", estimated_tokens=10)
        clock[0] += 2.0

    with pytest.raises(RuntimeError, match="Daily request quota"):
        limiter.acquire("gemini-3.8-flash", estimated_tokens=10)


def test_rate_limit_error_detection_and_retry(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    rate_limit.get_rate_limiter().reset()

    attempts = [0]

    def flaky():
        attempts[0] += 1
        if attempts[0] < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return "ok"

    result = rate_limit.call_with_rate_limit_retry("gemini-3.5-flash-lite", flaky, max_retries=2, base_delay=0.1)
    assert result == "ok"
    assert attempts[0] == 2
    assert len(sleeps) == 1
