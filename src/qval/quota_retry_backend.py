"""Backend wrapper that sleeps and retries on quota-exhaustion errors.

Codex enforces a 5-hour rolling usage window; crossing it surfaces as a
``QuotaExhaustedError`` from the backend. This wrapper intercepts those
errors on every ``generate_chat`` / ``generate_chat_batch`` call and
transparently retries on a fixed schedule:

  * one initial 5-minute sleep (catches transient "too many requests"
    bursts and short rate-limit backoffs),
  * then up to five 1-hour sleeps (each covers a Codex quota window).

If all retries exhaust, the ``QuotaExhaustedError`` propagates so the
pipeline can exit gracefully. When ``policy='abort'``, the wrapper does
not retry at all and lets the error propagate on first occurrence.

The wrapper mirrors ``LoggingBackend``'s delegation pattern so wrapping is
transparent: the same backend flows through trajectory collection, MC
estimation, and method evaluation with no changes to those call sites.

Wrap order at construction time:
``SecondElicitationBackendWrapper(``
``LoggingBackend(QuotaRetryBackendWrapper(raw_backend, policy=...))``
``)``, so quota retries do not spam the experiment log while
second-elicitation follow-up calls are logged as ordinary model calls.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    ScoringResult,
)

from qval.error_handling import is_quota_exhausted_error

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# One 5-minute retry then five 1-hour retries. Total worst-case sleep:
# 5 min + 5 x 1 h = 5 h 5 min across 7 total attempts, after which the
# error propagates for graceful abort.
_QUOTA_SLEEP_SCHEDULE: tuple[float, ...] = (
    300.0,
    3600.0,
    3600.0,
    3600.0,
    3600.0,
    3600.0,
)


class QuotaRetryBackendWrapper:
    """Wraps any ``ModelBackend`` with quota-aware sleep-and-retry logic."""

    def __init__(self, backend: ModelBackend, *, policy: str) -> None:
        if policy not in ("sleep_and_retry", "abort"):
            raise ValueError(
                "QuotaRetryBackendWrapper policy must be 'sleep_and_retry' "
                f"or 'abort', got {policy!r}"
            )
        self._backend = backend
        self._policy = policy
        self.enable_thinking = getattr(backend, "enable_thinking", None)

    @property
    def capabilities(self):
        return self._backend.capabilities

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "QuotaRetryBackendWrapper":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.close()

    def generate(
        self, prompts: list[str], params: SamplingParams
    ) -> list[GenerationResult]:
        return self._retry(lambda: self._backend.generate(prompts, params))

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        return self._retry(lambda: self._backend.generate_chat(messages, params))

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        return self._retry(
            lambda: self._backend.generate_chat_batch(messages_batch, params)
        )

    def score_chat(
        self,
        messages: list[ChatMessage],
        continuation: str,
    ) -> ScoringResult:
        return self._retry(lambda: self._backend.score_chat(messages, continuation))

    def score_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        continuations: list[str],
    ) -> list[ScoringResult]:
        return self._retry(
            lambda: self._backend.score_chat_batch(messages_batch, continuations)
        )

    def _retry(self, fn: Callable[[], _T]) -> _T:
        if self._policy == "abort":
            return fn()
        total_attempts = len(_QUOTA_SLEEP_SCHEDULE) + 1
        for attempt in range(total_attempts):
            try:
                return fn()
            except BaseException as exc:
                if not is_quota_exhausted_error(exc):
                    raise
                if attempt == total_attempts - 1:
                    logger.error(
                        "Quota-exhausted error persisted after %d attempts "
                        "(total sleep %.0fs). Giving up. Error: %s",
                        total_attempts,
                        sum(_QUOTA_SLEEP_SCHEDULE),
                        exc,
                    )
                    raise
                sleep_seconds = _QUOTA_SLEEP_SCHEDULE[attempt]
                logger.warning(
                    "Quota-exhausted error on attempt %d/%d, sleeping %.0fs "
                    "before retry. Error: %s",
                    attempt + 1,
                    total_attempts,
                    sleep_seconds,
                    exc,
                )
                time.sleep(sleep_seconds)
                logger.info(
                    "Woke from quota sleep - starting attempt %d/%d.",
                    attempt + 2,
                    total_attempts,
                )
        raise RuntimeError(
            "QuotaRetryBackendWrapper._retry fell through its loop; this is a bug."
        )


def wrap_with_quota_retry(
    backend: ModelBackend,
    *,
    policy: str,
) -> QuotaRetryBackendWrapper:
    """Return ``backend`` wrapped in ``QuotaRetryBackendWrapper`` with *policy*.

    Kept as a separate helper so backend-build sites stay one line each
    regardless of future wrapping complexity.
    """
    return QuotaRetryBackendWrapper(backend, policy=policy)
