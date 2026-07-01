"""Error classification and resilient generation for backend failures.

Provides error classifiers (``is_recoverable_backend_error``,
``is_transient_api_error``), ``retry_on_transient`` (generic retry helper),
and ``generate_batch_with_transient_retry`` which wraps
``generate_chat_batch`` with retry logic for transient errors and
offending-index removal for prompt-too-long errors.
"""

from __future__ import annotations

from importlib import import_module
import logging
import time
from typing import Callable, TypeVar

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

# Retry parameters for generate_batch_with_transient_retry
TRANSIENT_MAX_RETRIES = 10
TRANSIENT_RETRY_BASE_DELAY = 2.0  # seconds
TRANSIENT_RETRY_MAX_DELAY = 8.0  # seconds

# Retry parameters for MC rollout collection
MC_ROLLOUT_MAX_RETRIES = 6
MC_ROLLOUT_RETRY_BASE_DELAY = 4.0  # seconds

_protocol = import_module("llenvs.inference.protocol")
if not hasattr(_protocol, "QuotaExhaustedError"):
    class QuotaExhaustedError(RuntimeError):
        """Fallback quota exception for older llenvs versions."""

    _protocol.QuotaExhaustedError = QuotaExhaustedError


def _status_code_for_error(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def is_transient_api_error(exc: BaseException) -> bool:
    """Return ``True`` for transient API errors that may succeed on retry.

    Transient errors are non-deterministic failures where retrying the same
    request may succeed.  This is complementary to
    :func:`is_recoverable_backend_error` which identifies deterministic,
    input-specific failures (same input will always fail).

    Covers connection drops, timeouts, server-side errors (429, 500, 502,
    503, 529), malformed API responses (``JSONDecodeError`` from
    truncated/corrupted response bodies, or ``MalformedResponseError``
    when a provider returns HTTP 200 with a structurally invalid body —
    e.g. OpenRouter returning ``choices: null`` when an upstream provider
    failed).  A 429 reaching this layer means the SDK's built-in
    rate-limit retries were already exhausted.
    """
    import json

    # Malformed API response (truncated JSON, proxy error page, etc.)
    if isinstance(exc, json.JSONDecodeError):
        return True

    status = _status_code_for_error(exc)
    if status in (429, 500, 502, 503, 529):
        return True

    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError", "MalformedResponseError"}:
        return True

    text = str(exc).lower()
    if any(
        needle in text
        for needle in (
            "rate limited",
            "rate limit",
            "temporary failure",
            "temporarily unavailable",
            "overloaded",
        )
    ):
        return True

    return False


def retry_on_transient(
    fn: Callable[[], _T],
    *,
    max_retries: int = TRANSIENT_MAX_RETRIES,
    base_delay: float = TRANSIENT_RETRY_BASE_DELAY,
    description: str = "operation",
) -> _T:
    """Retry *fn* on transient API errors with capped exponential backoff.

    Non-transient exceptions propagate immediately.  After exhausting
    retries, the last transient exception is re-raised.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_transient_api_error(exc):
                raise
            last_exc = exc
            if attempt < max_retries:
                delay = _transient_retry_delay(
                    retry_number=attempt + 1,
                    base_delay=base_delay,
                )
                _log.warning(
                    "Transient API error in %s (attempt %d/%d), retrying in %.1fs: %s",
                    description,
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _transient_retry_delay(
    *,
    retry_number: int,
    base_delay: float,
) -> float:
    """Return capped retry delay for a 1-indexed retry attempt."""
    return min(base_delay * (2 ** (retry_number - 1)), TRANSIENT_RETRY_MAX_DELAY)


def is_recoverable_backend_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* is a safe per-point backend failure to skip."""
    try:
        from llenvs.inference.protocol import (
            RecoverableInputError,
            RetryExhaustedTransientError,
        )
    except ImportError:
        from llenvs.inference.protocol import PromptTooLongError

        return isinstance(exc, PromptTooLongError)

    return isinstance(exc, (RecoverableInputError, RetryExhaustedTransientError))


def is_quota_exhausted_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* is (or wraps) a backend quota-exhaustion error.

    Matches the raw ``QuotaExhaustedError`` and also ``PartialBatchError``
    where every failed slot is itself a ``QuotaExhaustedError`` (the case
    where all concurrent items in a batch hit quota at once, e.g., Codex).

    If the installed ``llenvs`` version predates ``QuotaExhaustedError``,
    the function safely returns ``False`` rather than raising ``ImportError``
    — so crash handlers that rely on this classifier never themselves crash
    on version skew.
    """
    try:
        from llenvs.inference.protocol import QuotaExhaustedError
    except ImportError:
        return False

    if isinstance(exc, QuotaExhaustedError):
        return True

    try:
        from llenvs.inference.protocol import PartialBatchError
    except ImportError:
        return False

    if isinstance(exc, PartialBatchError) and exc.failures:
        return all(
            isinstance(failure, QuotaExhaustedError)
            for failure in exc.failures.values()
        )
    return False


def _validate_partial_batch_contract(
    exc: BaseException,
    *,
    active_count: int,
) -> None:
    from llenvs.inference.protocol import PartialBatchError

    if not isinstance(exc, PartialBatchError):
        return
    if len(exc.results) != active_count:
        raise RuntimeError(
            "PartialBatchError contract violation during retry handling: "
            f"expected {active_count} result slots, got {len(exc.results)}"
        )
    if any(idx < 0 or idx >= active_count for idx in exc.failures):
        raise RuntimeError(
            "PartialBatchError contract violation during retry handling: "
            "failure indices fell outside the active batch"
        )


def check_abort_threshold(
    num_aborted: int,
    total_points: int,
    max_aborted_points: float,
    phase: str,
) -> None:
    """Raise RuntimeError if the abort rate exceeds the configured threshold."""
    if total_points == 0:
        return
    abort_rate = num_aborted / total_points
    if abort_rate > max_aborted_points:
        raise RuntimeError(
            f"Abort rate {abort_rate:.1%} ({num_aborted}/{total_points}) "
            f"in phase '{phase}' exceeds max_aborted_points="
            f"{max_aborted_points:.1%}. This indicates a systemic issue "
            f"(e.g., most prompts are too long for the model)."
        )


def generate_batch_with_transient_retry(
    backend: object,
    messages: list,
    params: object,
    *,
    max_retries: int = TRANSIENT_MAX_RETRIES,
    base_delay: float = TRANSIENT_RETRY_BASE_DELAY,
) -> list:
    """Call ``generate_chat_batch`` with transient retry and offending-index handling.

    Returns a list the same length as *messages*.  Positions that could not be
    generated (transient failure after retries, prompt too long) are ``None``;
    the caller decides how to handle them (NaN for predictions, skip for
    ranking candidates).

    Non-recoverable, non-transient errors are **re-raised**.  Callers that
    want to swallow fatal errors (e.g., ranking collection) should wrap the
    call in their own ``try/except``.
    """
    from llenvs.inference.protocol import GenerationResult

    results: list[GenerationResult | None] = [None] * len(messages)
    active_indices = list(range(len(messages)))
    active_messages = list(messages)
    transient_attempts = 0

    while active_indices:
        try:
            gen_results = backend.generate_chat_batch(active_messages, params)
            for local_idx, gen_result in enumerate(gen_results):
                results[active_indices[local_idx]] = gen_result
            return results
        except Exception as exc:
            partial_cls = None
            try:
                from llenvs.inference.protocol import (
                    PartialBatchError as _PartialBatchError,
                )
            except ImportError:
                _PartialBatchError = None
            partial_cls = _PartialBatchError

            # --- Partial success batch error ---
            if partial_cls is not None and isinstance(exc, partial_cls):
                _validate_partial_batch_contract(exc, active_count=len(active_indices))
                transient_pairs: list[tuple[int, object]] = []
                fatal_failure: BaseException | None = None

                for local_idx, entry in enumerate(exc.results):
                    if local_idx >= len(active_indices):
                        break
                    if not isinstance(entry, BaseException):
                        results[active_indices[local_idx]] = entry

                for local_idx, failure in exc.failures.items():
                    if local_idx >= len(active_indices):
                        continue
                    if is_recoverable_backend_error(failure):
                        continue
                    if is_transient_api_error(failure):
                        transient_pairs.append(
                            (active_indices[local_idx], active_messages[local_idx])
                        )
                        continue
                    fatal_failure = failure
                    break

                if fatal_failure is not None:
                    raise fatal_failure

                if not transient_pairs:
                    return results

                transient_attempts += 1
                if transient_attempts <= max_retries:
                    delay = _transient_retry_delay(
                        retry_number=transient_attempts,
                        base_delay=base_delay,
                    )
                    _log.warning(
                        "Transient API error in partial batch (attempt %d/%d), "
                        "retrying %d failed prompts in %.1fs",
                        transient_attempts,
                        max_retries,
                        len(transient_pairs),
                        delay,
                    )
                    time.sleep(delay)
                    active_indices = [idx for idx, _ in transient_pairs]
                    active_messages = [msg for _, msg in transient_pairs]
                    continue
                _log.error(
                    "Transient API error persisted after %d retries for %d "
                    "partial-batch prompts",
                    max_retries,
                    len(transient_pairs),
                )
                return results

            # --- Deterministic per-input error (PromptTooLongError) ---
            if is_recoverable_backend_error(exc):
                offending = getattr(exc, "offending_indices", None)
                if offending and len(offending) < len(active_indices):
                    offending_set = set(offending)
                    _log.warning(
                        "Removed %d too-long prompts, retrying %d remaining",
                        len(offending_set),
                        len(active_indices) - len(offending_set),
                    )
                    pairs = [
                        (idx, msg)
                        for i, (idx, msg) in enumerate(
                            zip(active_indices, active_messages)
                        )
                        if i not in offending_set
                    ]
                    active_indices = [p[0] for p in pairs]
                    active_messages = [p[1] for p in pairs]
                    continue  # does not consume transient retry budget
                _log.warning(
                    "Recoverable error without actionable offending "
                    "indices, skipping batch: %s",
                    exc,
                )
                return results

            # --- Transient error (connection, timeout, rate limit) ---
            if is_transient_api_error(exc):
                transient_attempts += 1
                if transient_attempts <= max_retries:
                    delay = _transient_retry_delay(
                        retry_number=transient_attempts,
                        base_delay=base_delay,
                    )
                    _log.warning(
                        "Transient API error (attempt %d/%d), retrying in %.1fs: %s",
                        transient_attempts,
                        max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                _log.error(
                    "Transient API error persisted after %d retries, "
                    "giving up on batch: %s",
                    max_retries,
                    exc,
                )
                return results

            # --- Unknown / fatal error — re-raise ---
            raise

    return results


def _looks_like_tmux_session_dead_message(message: str) -> bool:
    lowered = message.lower()
    return (
        "tmux session died during command execution" in lowered
        or ("no server running" in lowered and "tmux" in lowered)
        or "can't find session" in lowered
        or "can't find pane" in lowered
    )


def classify_rollout_error(exc: BaseException) -> str | None:
    """Classify a rollout environment error for per-point handling.

    Returns:
        ``"jericho_native_fault"`` for recoverable Jericho native faults
        (e.g. SIGFPE or emulator halt),
        ``"timeout"`` for recoverable replay/step timeout failures that should
        abort only the affected point, ``"shell_continuation"`` when replay
        entered bash continuation mode because a command was syntactically
        incomplete, ``"tmux_session_dead"`` when a Harbor tmux shell dies
        during replay/step execution, or ``None`` when the error should still
        be treated as job-fatal.
    """
    try:
        from llenvs.adapters.jericho import JerichoEmulatorHaltedError
    except ImportError:  # pragma: no cover - llenvs is an installed dependency
        JerichoEmulatorHaltedError = ()

    message = str(exc)
    if isinstance(exc, JerichoEmulatorHaltedError):
        return "jericho_native_fault"
    if message.startswith("SIGFPE:") or message.startswith("Jericho emulator halted"):
        return "jericho_native_fault"
    if "shell continuation prompt" in message:
        return "shell_continuation"
    if _looks_like_tmux_session_dead_message(message):
        return "tmux_session_dead"
    if "timed out after" in message:
        return "timeout"
    return None


def is_recoverable_rollout_error(exc: BaseException) -> bool:
    """Return ``True`` if a rollout environment error should abort only a point."""
    return classify_rollout_error(exc) is not None


class EnvError(RuntimeError):
    """An environment operation failed due to a native-code crash.

    Wraps OS-level signals (e.g. SIGFPE from Jericho's Frotz interpreter)
    that would otherwise kill the process with no Python traceback.
    The message records what actually happened.
    """


def install_sigfpe_handler() -> None:
    """Convert SIGFPE into a Python :class:`EnvError`.

    Native C libraries (e.g. Jericho's Frotz Z-machine interpreter) can
    trigger SIGFPE on integer division by zero.  By default this kills the
    process.  Installing this handler lets existing ``try/except`` blocks
    catch the error and skip the offending input.
    """
    import signal

    def _handler(signum: int, frame: object) -> None:
        raise EnvError("SIGFPE: integer division by zero in native environment code")

    signal.signal(signal.SIGFPE, _handler)
