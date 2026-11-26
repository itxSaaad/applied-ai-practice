"""Client-side RPM/TPM/RPD rate limiting and 429 retry helpers for Gemini API calls."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_google_genai import ChatGoogleGenerativeAI

from utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelRateLimit:
    """Per-model rate-limit quotas (RPM, TPM, RPD)."""

    rpm: int
    tpm: int
    rpd: int


# Limits sourced from the project's Google AI Studio quota dashboard.
MODEL_RATE_LIMITS: dict[str, ModelRateLimit] = {
    "gemini-3.5-flash-lite": ModelRateLimit(rpm=15, tpm=250_000, rpd=500),
    "gemini-3.1-flash-lite": ModelRateLimit(rpm=15, tpm=250_000, rpd=500),
    "gemini-3.8-flash": ModelRateLimit(rpm=5, tpm=250_000, rpd=20),
    "gemini-3.6-flash": ModelRateLimit(rpm=5, tpm=250_000, rpd=20),
    "gemini-3-flash": ModelRateLimit(rpm=5, tpm=250_000, rpd=20),
    "gemini-4-hydrogen": ModelRateLimit(rpm=5, tpm=250_000, rpd=20),
    "gemini-2.5-flash": ModelRateLimit(rpm=5, tpm=250_000, rpd=20),
    "gemini-2.5-flash-lite": ModelRateLimit(rpm=10, tpm=250_000, rpd=20),
    "gemini-robotics-er-2-preview": ModelRateLimit(rpm=5, tpm=250_000, rpd=20),
    "gemini-3.8-flash-tts": ModelRateLimit(rpm=3, tpm=10_000, rpd=10),
    "gemini-3.8-flash-lite-tts": ModelRateLimit(rpm=3, tpm=10_000, rpd=10),
    "gemini-3.1-flash-tts": ModelRateLimit(rpm=3, tpm=10_000, rpd=10),
    "gemini-2.5-flash-tts": ModelRateLimit(rpm=3, tpm=10_000, rpd=10),
    "gemini-embedding-1": ModelRateLimit(rpm=100, tpm=30_000, rpd=1_000),
    "gemini-embedding-2": ModelRateLimit(rpm=100, tpm=30_000, rpd=1_000),
    "gemma-4-26b": ModelRateLimit(rpm=30, tpm=16_000, rpd=14_400),
    "gemma-4-31b": ModelRateLimit(rpm=30, tpm=16_000, rpd=14_400),
}

DEFAULT_GEMINI_LIMIT = ModelRateLimit(rpm=5, tpm=250_000, rpd=20)
_WINDOW_SECONDS = 61.0
_DAY_SECONDS = 86_400.0


def normalize_model_name(model_name: str) -> str:
    """Strip provider prefixes such as 'google_genai:' or 'models/'."""
    normalized = model_name.strip().lower()
    for prefix in ("google_genai:", "models/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def get_model_rate_limit(model_name: str) -> ModelRateLimit | None:
    """Return the configured rate limit for a Gemini/Gemma model, or None for local models."""
    normalized = normalize_model_name(model_name)
    if normalized in MODEL_RATE_LIMITS:
        return MODEL_RATE_LIMITS[normalized]
    if normalized.startswith(("gemini-", "gemma-")):
        return DEFAULT_GEMINI_LIMIT
    return None


def estimate_tokens(payload: Any, *, output_buffer: int = 256) -> int:
    """Conservatively estimate request tokens from character length."""
    if payload is None:
        return output_buffer
    if isinstance(payload, str):
        chars = len(payload)
    elif isinstance(payload, list):
        chars = sum(len(str(getattr(item, "content", item))) for item in payload)
    else:
        chars = len(str(payload))
    return max(1, chars // 4) + output_buffer


class _ModelBucket:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.window: deque[tuple[float, int]] = deque()
        self.daily: deque[float] = deque()


class SlidingWindowRateLimiter:
    """Thread-safe and async-friendly sliding-window rate limiter per model."""

    def __init__(self, window_seconds: float = _WINDOW_SECONDS) -> None:
        self._window_seconds = window_seconds
        self._buckets_lock = threading.Lock()
        self._buckets: dict[str, _ModelBucket] = {}

    def _get_bucket(self, normalized_model: str) -> _ModelBucket:
        with self._buckets_lock:
            bucket = self._buckets.get(normalized_model)
            if bucket is None:
                bucket = _ModelBucket()
                self._buckets[normalized_model] = bucket
            return bucket

    def _try_acquire(self, model_name: str, estimated_tokens: int) -> tuple[bool, float]:
        limit = get_model_rate_limit(model_name)
        if limit is None:
            return True, 0.0

        normalized = normalize_model_name(model_name)
        bucket = self._get_bucket(normalized)
        with bucket.lock:
            now = time.monotonic()
            cutoff = now - self._window_seconds
            day_cutoff = now - _DAY_SECONDS

            while bucket.window and bucket.window[0][0] <= cutoff:
                bucket.window.popleft()
            while bucket.daily and bucket.daily[0] <= day_cutoff:
                bucket.daily.popleft()

            if len(bucket.daily) >= limit.rpd:
                raise RuntimeError(f"Daily request quota ({limit.rpd} RPD) reached for {normalized}.")

            current_rpm = len(bucket.window)
            current_tpm = sum(tokens for _, tokens in bucket.window)
            tokens_ok = (current_tpm + estimated_tokens) <= limit.tpm or current_rpm == 0

            if current_rpm < limit.rpm and tokens_ok:
                bucket.window.append((now, estimated_tokens))
                bucket.daily.append(now)
                return True, 0.0

            # Base wait is a small constant; add jitter to avoid thundering herd.
            # Jitter is a random offset in the range [-10%, +10%] of the base wait.
            base_wait = 0.1
            if bucket.window:
                # Calculate the time until the oldest entry slides out of the window
                # plus a small safety margin (0.05 s).
                base_wait = max(
                    base_wait,
                    bucket.window[0][0] + self._window_seconds - now + 0.05,
                )
            # Apply exponential back‑off factor on consecutive throttles.
            # The bucket tracks the number of recent throttles via its window length.
            throttle_factor = min(2 ** (len(bucket.window) - 1), 8)
            jitter = random.uniform(-0.1, 0.1) * base_wait
            wait_seconds = max(base_wait * throttle_factor + jitter, 0.0)
            return False, wait_seconds

    def acquire(self, model_name: str, estimated_tokens: int = 256) -> float:
        """Block until a request slot is available; return total seconds waited."""
        waited = 0.0
        while True:
            acquired, wait_seconds = self._try_acquire(model_name, estimated_tokens)
            if acquired:
                return waited
            logger.debug(
                "Rate limit throttle for %s: sleeping %.2fs",
                model_name,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            waited += wait_seconds

    async def async_acquire(self, model_name: str, estimated_tokens: int = 256) -> float:
        """Asynchronously wait until a request slot is available."""
        waited = 0.0
        while True:
            acquired, wait_seconds = self._try_acquire(model_name, estimated_tokens)
            if acquired:
                return waited
            logger.debug(
                "Async rate limit throttle for %s: sleeping %.2fs",
                model_name,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)
            waited += wait_seconds

    def reset(self) -> None:
        """Clear all recorded request windows (primarily for unit tests)."""
        with self._buckets_lock:
            self._buckets.clear()


_DEFAULT_LIMITER = SlidingWindowRateLimiter()


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Return the process-wide Gemini rate limiter."""
    return _DEFAULT_LIMITER


def acquire_rate_limit(model_name: str, payload: Any = None) -> float:
    """Acquire a synchronous rate-limit slot for the given model."""
    return _DEFAULT_LIMITER.acquire(model_name, estimate_tokens(payload))


async def async_acquire_rate_limit(model_name: str, payload: Any = None) -> float:
    """Acquire an asynchronous rate-limit slot for the given model."""
    return await _DEFAULT_LIMITER.async_acquire(model_name, estimate_tokens(payload))


def is_rate_limit_error(error: BaseException) -> bool:
    """Return True if an exception indicates a 429 / quota / rate-limit response."""
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "429",
            "resource_exhausted",
            "resourceexhausted",
            "rate limit",
            "ratelimit",
            "quota",
            "too many requests",
        )
    )


def call_with_rate_limit_retry[T](
    model_name: str,
    fn: Callable[[], T],
    *,
    payload: Any = None,
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> T:
    """Execute a synchronous call under rate limiting with exponential backoff on 429."""
    for attempt in range(max_retries + 1):
        acquire_rate_limit(model_name, payload)
        try:
            return fn()
        except Exception as error:
            if not is_rate_limit_error(error) or attempt >= max_retries:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0.0, 1.0)
            logger.warning(
                "Rate-limited by %s (attempt %d/%d); retrying in %.1fs.",
                model_name,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable retry loop exit.")


class RateLimitedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI subclass enforcing per-model RPM/TPM/RPD limits."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return call_with_rate_limit_retry(
            self.model,
            lambda: super(RateLimitedChatGoogleGenerativeAI, self)._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ),
            payload=messages,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        max_retries = 3
        for attempt in range(max_retries + 1):
            await async_acquire_rate_limit(self.model, messages)
            try:
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as error:
                if not is_rate_limit_error(error) or attempt >= max_retries:
                    raise
                delay = 5.0 * (2**attempt) + random.uniform(0.0, 1.0)
                logger.warning(
                    "Async rate-limited by %s (attempt %d/%d); retrying in %.1fs.",
                    self.model,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable async retry loop exit.")

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        acquire_rate_limit(self.model, messages)
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        await async_acquire_rate_limit(self.model, messages)
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk
