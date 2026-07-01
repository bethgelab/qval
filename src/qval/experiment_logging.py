"""Experiment logging: captures every LLM call with full prompts, responses, and timing."""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
)

from qval.types import RawPointReturns


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert a list of ChatMessages to JSON-serializable dicts."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _summarize_messages(messages: list[ChatMessage]) -> dict[str, Any]:
    """Summarize prompt size characteristics for debugging failures."""
    content_chars = [len(m.content or "") for m in messages]
    return {
        "num_messages": len(messages),
        "roles": [m.role for m in messages],
        "content_chars": content_chars,
        "total_content_chars": sum(content_chars),
    }


def _serialize_unique_message_groups(
    messages_batch: list[list[ChatMessage]],
) -> list[dict[str, Any]]:
    """Compress a batch into unique prompts plus repetition counts.

    Batched MC rollouts often repeat the exact same prompt many times to obtain
    independent samples. Grouping identical prompts keeps JSONL logs much
    smaller while preserving the full prompt text and its multiplicity.
    """
    groups: list[dict[str, Any]] = []
    group_by_prompt: dict[str, int] = {}

    for msgs in messages_batch:
        serialized = _serialize_messages(msgs)
        key = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
        group_idx = group_by_prompt.get(key)
        if group_idx is None:
            group_by_prompt[key] = len(groups)
            groups.append(
                {
                    "prompt": serialized,
                    "count": 1,
                    "message_summary": _summarize_messages(msgs),
                }
            )
        else:
            groups[group_idx]["count"] += 1

    return groups


_FAILURE_SAMPLE_LIMIT = 5


def _extract_status_code(exc: BaseException) -> int | None:
    """Extract HTTP status from an OpenAI-SDK-shaped exception.

    The SDK exposes ``status_code`` on the exception itself, but on some
    transport-layer errors only ``response.status_code`` is set.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _extract_body(exc: BaseException) -> Any:
    """Extract the provider response body verbatim, or ``None``.

    Prefers the ``body`` attribute (already-parsed JSON on OpenAI SDK
    errors); falls back to ``response.json()`` and swallows decode errors
    so a malformed body doesn't kill logging itself.
    """
    body = getattr(exc, "body", None)
    if body is not None:
        return body
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return response.json()
    except Exception:
        return None


def _serialize_error(exc: BaseException) -> dict[str, Any]:
    """Convert an exception to a JSON-safe payload, preserving HTTP status,
    response bodies, per-item failures from PartialBatchError, OpenRouter's
    top-level ``provider_error`` (MalformedResponseError), and backend/model
    identifiers when present.

    The recursive traversal of ``PartialBatchError.failures`` groups
    duplicates by ``(type, status_code, message)`` and emits at most
    ``_FAILURE_SAMPLE_LIMIT`` distinct samples — enough to diagnose
    "all 100 items failed for the same reason" cases without bloating logs
    when failures vary.
    """
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }

    status_code = _extract_status_code(exc)
    if status_code is not None:
        payload["status_code"] = status_code

    body = _extract_body(exc)
    if body is not None:
        payload["body"] = sanitize_for_json(body)

    failures = getattr(exc, "failures", None)
    if isinstance(failures, dict) and failures:
        results = getattr(exc, "results", None)
        payload["failure_count"] = len(failures)
        if isinstance(results, list):
            payload["result_count"] = len(results)
        payload["failure_samples"] = _summarize_failures(failures)

    provider_error = getattr(exc, "provider_error", None)
    if provider_error is not None:
        payload["provider_error"] = sanitize_for_json(provider_error)

    backend_name = getattr(exc, "backend_name", None)
    if isinstance(backend_name, str) and backend_name:
        payload["backend_name"] = backend_name
    model_name = getattr(exc, "model_name", None)
    if isinstance(model_name, str) and model_name:
        payload["model_name"] = model_name

    return payload


def _summarize_failures(
    failures: dict[int, BaseException],
) -> list[dict[str, Any]]:
    """Group per-item failures by (type, status_code, message), keep the
    earliest index per group, sort by frequency, cap at the sample limit.
    """
    groups: dict[tuple[str, Any, str], dict[str, Any]] = {}
    for idx, sub_exc in failures.items():
        if not isinstance(sub_exc, BaseException):
            continue
        sub_serialized = _serialize_error(sub_exc)
        key = (
            sub_serialized["type"],
            sub_serialized.get("status_code"),
            sub_serialized["message"],
        )
        existing = groups.get(key)
        if existing is None:
            groups[key] = {
                "count": 1,
                "first_index": idx,
                "error": sub_serialized,
            }
        else:
            existing["count"] += 1
            if idx < existing["first_index"]:
                existing["first_index"] = idx
    sorted_groups = sorted(
        groups.values(),
        key=lambda g: (-g["count"], g["first_index"]),
    )
    return sorted_groups[:_FAILURE_SAMPLE_LIMIT]


def _stderr_prompt_too_long(
    exc: BaseException,
    phase: str,
    backend_wrapper: Any,
    params: SamplingParams | None = None,
) -> None:
    """Print structured diagnostics to stderr for PromptTooLongError."""
    from llenvs.inference.protocol import PromptTooLongError

    if not isinstance(exc, PromptTooLongError):
        return

    import sys

    lines = [
        "=" * 72,
        "PROMPT TOO LONG — diagnostic block",
        f"  Phase:         {phase}",
        f"  Model:         {exc.model_name or getattr(backend_wrapper, 'model_name', '')}",
        f"  Max model len: {exc.max_model_len}",
        f"  Batch size:    {exc.batch_size}",
    ]
    if params is not None:
        lines.append(
            f"  Sampling:      {json.dumps(_serialize_sampling_params(params), sort_keys=True)}"
        )
    if exc.prompt_token_lengths:
        lines.append("  Per-prompt token lengths:")
        for i, length in enumerate(exc.prompt_token_lengths):
            marker = " *** OVER" if i in exc.offending_indices else ""
            lines.append(f"    [{i}] {length} tokens{marker}")
    if exc.offending_prompts:
        for idx, prompt_text in zip(exc.offending_indices, exc.offending_prompts):
            lines.append(f"  --- Offending prompt [{idx}] (full text) ---")
            lines.append(prompt_text)
            lines.append(f"  --- End prompt [{idx}] ---")
    lines.append("=" * 72)
    print("\n".join(lines), file=sys.stderr, flush=True)


def _serialize_result(result: GenerationResult) -> dict[str, Any]:
    """Convert a GenerationResult to a JSON-serializable dict."""
    return {
        "text": result.text,
        "finish_reason": result.finish_reason.name if result.finish_reason else None,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "metadata": result.metadata,
    }


def _serialize_sampling_params(params: SamplingParams) -> dict[str, Any]:
    """Convert SamplingParams to a JSON-serializable dict."""
    d = asdict(params)
    # Convert tuples to lists for JSON
    if "stop_sequences" in d:
        d["stop_sequences"] = list(d["stop_sequences"])
    return d


def _serialize_backend_info(backend: Any) -> dict[str, Any]:
    """Extract stable backend metadata for log records."""
    return {
        "model_name": getattr(backend, "model_name", None),
        "enable_thinking": getattr(backend, "enable_thinking", None),
    }


def _normalized_sampling_settings(
    params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract truncation-relevant settings from serialized sampling params."""
    if params is None:
        return None
    thinking_budget_suffix = params.get("thinking_budget_suffix")
    second_elicitation_suffix = params.get("second_elicitation_suffix")
    return {
        "max_tokens": params.get("max_tokens"),
        "thinking_budget": params.get("thinking_budget"),
        "thinking_budget_soft_ratio": params.get("thinking_budget_soft_ratio"),
        "thinking_budget_suffix": thinking_budget_suffix,
        "thinking_budget_suffix_enabled": thinking_budget_suffix is not None,
        "second_elicitation_enabled": second_elicitation_suffix is not None,
        "second_elicitation_max_tokens": params.get("second_elicitation_max_tokens")
        if second_elicitation_suffix is not None
        else None,
        "second_elicitation_suffix": second_elicitation_suffix,
    }


def _normalize_backend_settings(
    backend: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract truncation-relevant backend settings from serialized backend info."""
    if backend is None:
        return None
    return {
        "model_name": backend.get("model_name"),
        "enable_thinking": backend.get("enable_thinking"),
    }


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN and Inf float values with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    return obj


def compute_numeric_summary(values: list[float] | list[int]) -> dict[str, Any]:
    """Compute count/mean/std/min/max for a numeric sequence."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    values_f = [float(value) for value in values]
    return {
        "count": len(values_f),
        "mean": statistics.fmean(values_f),
        "std": statistics.pstdev(values_f),
        "min": min(values_f),
        "max": max(values_f),
    }


def compute_boolean_summary(values: list[bool]) -> dict[str, Any]:
    """Compute count, true count, and true rate for a boolean sequence."""
    true_count = sum(1 for value in values if value)
    total = len(values)
    return {
        "count": total,
        "true_count": true_count,
        "true_rate": (true_count / total) if total else None,
    }


def summarize_trajectory_results(
    results: list[Any],
    *,
    trajectory_returns: list[float] | None = None,
) -> dict[str, Any]:
    """Summarize trajectory returns, lengths, and success outcomes."""
    if trajectory_returns is None:
        returns = [float(result.total_reward) for result in results]
    else:
        returns = [float(value) for value in trajectory_returns]
    step_counts = [len(result.trajectory.transitions) for result in results]
    success_summary = compute_boolean_summary([bool(result.success) for result in results])
    return {
        "num_trajectories": len(results),
        "returns": compute_numeric_summary(returns),
        "steps": compute_numeric_summary(step_counts),
        "success_count": success_summary["true_count"],
        "success_rate": success_summary["true_rate"],
    }


def summarize_raw_rollouts(raw_returns: list[RawPointReturns]) -> dict[str, Any]:
    """Summarize rollout returns, rollout lengths, and per-point rollout counts."""
    rollout_returns: list[float] = []
    rollout_step_counts: list[int] = []
    rollouts_per_point: list[int] = []

    for raw in raw_returns:
        point_rollout_count = len(raw.returns) + len(raw.next_state_returns)
        rollouts_per_point.append(point_rollout_count)
        rollout_returns.extend(raw.returns)
        rollout_returns.extend(raw.next_state_returns)
        rollout_step_counts.extend(raw.step_counts)
        rollout_step_counts.extend(raw.next_state_step_counts)

    return {
        "num_points": len(raw_returns),
        "num_rollouts": len(rollout_returns),
        "returns": compute_numeric_summary(rollout_returns),
        "steps": compute_numeric_summary(rollout_step_counts),
        "rollouts_per_point": compute_numeric_summary(rollouts_per_point),
    }


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------


class ExperimentLogger:
    """Central collector for experiment log records.

    Maintains a list of call records tagged with phase labels
    (e.g., "trajectory_collection", "mc_estimation", "method:llm_direct").
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._phase: str = ""
        self._next_index: int = 0
        self._jsonl_sink_path: Path | None = None

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    @property
    def current_phase(self) -> str:
        return self._phase

    def set_phase(self, name: str) -> None:
        """Set the current phase tag for subsequent log records."""
        self._phase = name

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Context manager that sets phase and restores the previous one on exit."""
        prev = self._phase
        self._phase = name
        try:
            yield
        finally:
            self._phase = prev

    def next_index(self) -> int:
        """Return the next monotonic index and increment the counter."""
        idx = self._next_index
        self._next_index += 1
        return idx

    def log_call(self, record: dict[str, Any]) -> None:
        """Append a call record."""
        self._records.append(record)
        self._append_to_sink(record)

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a non-call event (e.g., phase summary).

        Event records are distinct from LLM call records — they have an
        ``event_type`` key instead of ``call_type`` and are excluded from
        the resource summary.
        """
        record = {
            "index": self.next_index(),
            "phase": self._phase,
            "event_type": event_type,
            "timestamp": datetime.now().astimezone().isoformat(),
            **data,
        }
        self._records.append(record)
        self._append_to_sink(record)

    def attach_jsonl_sink(self, path: str | Path) -> None:
        """Stream records to a JSONL file as they are logged."""
        sink_path = Path(path)
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        sink_path.write_text("")
        self._jsonl_sink_path = sink_path
        for record in self._records:
            self._append_to_sink(record)

    def _append_to_sink(self, record: dict[str, Any]) -> None:
        if self._jsonl_sink_path is None:
            return
        with self._jsonl_sink_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_progress(
        self,
        *,
        name: str,
        completed: int,
        total: int,
        logger_name: str,
        force: bool = False,
    ) -> None:
        """Log a progress update as an info line only."""
        percent = round((completed / total * 100.0) if total > 0 else 0.0, 1)
        logging.getLogger(logger_name).info(
            "%s: %d/%d (%.1f%%)",
            name,
            completed,
            total,
            percent,
        )

    def make_progress_reporter(
        self,
        *,
        name: str,
        logger_name: str,
        min_step_pct: float = 0.0,
    ) -> Any:
        """Create a progress callback.

        By default (``min_step_pct == 0``) it logs on every change in the
        completed count. With ``min_step_pct > 0`` intermediate updates are
        throttled: a line is logged only once the percentage advances by at
        least ``min_step_pct`` since the last logged line. The first and final
        (``completed >= total``) updates always log.
        """
        return _ProgressReporter(
            experiment_logger=self,
            name=name,
            logger_name=logger_name,
            min_step_pct=min_step_pct,
        )

    def get_phase_records(self, phase: str) -> list[dict[str, Any]]:
        """Return all records matching the given phase."""
        return [r for r in self._records if r.get("phase") == phase]

    def annotate_last(self, phase: str, **kwargs: Any) -> None:
        """Add annotation to the last record in the given phase."""
        for record in reversed(self._records):
            if record.get("phase") == phase:
                record["annotation"] = kwargs
                return

    def annotate_phase_records(
        self, phase: str, annotations: list[dict[str, Any]]
    ) -> None:
        """Annotate all records in a phase, zipping with the annotations list."""
        phase_records = self.get_phase_records(phase)
        for record, ann in zip(phase_records, annotations):
            record["annotation"] = ann

    def compute_resource_summary(self) -> dict[str, Any]:
        """Aggregate token counts, call counts, and timing by phase.

        Only counts LLM call records (those with a ``call_type`` key).
        Event records logged via ``log_event()`` are excluded.
        """
        by_phase: dict[str, dict[str, float]] = {}
        total_calls = 0
        total_prompt = 0
        total_completion = 0
        total_elapsed = 0.0

        for record in self._records:
            if "call_type" not in record:
                continue  # skip event records

            phase = record.get("phase", "unknown")
            prompt_tokens = record.get("prompt_tokens", 0)
            completion_tokens = record.get("completion_tokens", 0)
            elapsed = record.get("elapsed_seconds", 0.0)

            total_calls += 1
            total_prompt += prompt_tokens
            total_completion += completion_tokens
            total_elapsed += elapsed

            if phase not in by_phase:
                by_phase[phase] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed_seconds": 0.0,
                }
            by_phase[phase]["calls"] += 1
            by_phase[phase]["prompt_tokens"] += prompt_tokens
            by_phase[phase]["completion_tokens"] += completion_tokens
            by_phase[phase]["elapsed_seconds"] += elapsed

        return {
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_elapsed_seconds": round(total_elapsed, 3),
            "by_phase": by_phase,
        }

    def save_logs(self, path: str | Path) -> None:
        """Write all records as JSONL (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self._records:
                f.write(json.dumps(record, default=str) + "\n")

    def save_summary(
        self,
        path: str | Path,
        *,
        experiment_info: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        environments: dict[str, Any] | None = None,
    ) -> None:
        """Write a summary JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment": experiment_info or {},
            "config": config or {},
            "environments": environments or {},
            "resources": self.compute_resource_summary(),
        }

        with open(path, "w") as f:
            json.dump(summary, f, indent=4, default=str)


# ---------------------------------------------------------------------------
# LoggingBackend
# ---------------------------------------------------------------------------


def compute_stats(
    logger: ExperimentLogger,
    phase_prefix: str,
) -> dict[str, Any]:
    """Compute per-generation statistics for a phase prefix.

    Inspects all LLM call records whose phase starts with *phase_prefix*.
    For ``generate_chat`` records, checks the single response's
    ``finish_reason``.  For ``generate_chat_batch`` records, checks each
    response in the batch.

    Returns a dict with per-generation truncation counts, percentages,
    finish-reason breakdowns, token totals, and relevant generation settings.
    """
    total_generations = 0
    truncated_generations = 0
    generate_chat_calls = 0
    generate_chat_batch_calls = 0
    batched_response_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    finish_reason_counts: dict[str, int] = {}
    sampling_variants: list[dict[str, Any]] = []
    backend_variants: list[dict[str, Any]] = []

    def _add_variant(
        variants: list[dict[str, Any]], value: dict[str, Any] | None
    ) -> None:
        if value is None:
            return
        if value not in variants:
            variants.append(value)

    for record in logger.records:
        if "call_type" not in record:
            continue
        phase = record.get("phase", "")
        if not phase.startswith(phase_prefix):
            continue
        if "error" in record:
            continue

        call_type = record["call_type"]
        total_prompt_tokens += int(record.get("prompt_tokens", 0) or 0)
        total_completion_tokens += int(record.get("completion_tokens", 0) or 0)
        _add_variant(
            sampling_variants,
            _normalized_sampling_settings(record.get("sampling_params")),
        )
        _add_variant(
            backend_variants,
            _normalize_backend_settings(record.get("backend")),
        )

        if call_type == "generate_chat":
            generate_chat_calls += 1
            total_generations += 1
            resp = record.get("response", {})
            finish_reason = resp.get("finish_reason") or "UNKNOWN"
            finish_reason_counts[finish_reason] = (
                finish_reason_counts.get(finish_reason, 0) + 1
            )
            if finish_reason == "MAX_TOKENS":
                truncated_generations += 1
        elif call_type == "generate_chat_batch":
            generate_chat_batch_calls += 1
            responses = record.get("responses", [])
            batched_response_count += len(responses)
            total_generations += len(responses)
            for resp in responses:
                finish_reason = resp.get("finish_reason") or "UNKNOWN"
                finish_reason_counts[finish_reason] = (
                    finish_reason_counts.get(finish_reason, 0) + 1
                )
                if finish_reason == "MAX_TOKENS":
                    truncated_generations += 1

    pct = (
        truncated_generations / total_generations * 100
        if total_generations > 0
        else 0.0
    )
    result = {
        "total_generations": total_generations,
        "truncated_generations": truncated_generations,
        "truncated_generations_pct": round(pct, 2),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "generate_chat_calls": generate_chat_calls,
        "generate_chat_batch_calls": generate_chat_batch_calls,
        "batched_response_count": batched_response_count,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "sampling_settings": sampling_variants[0]
        if len(sampling_variants) == 1
        else None,
        "sampling_settings_variants": sampling_variants
        if len(sampling_variants) > 1
        else [],
        "backend_settings": backend_variants[0] if len(backend_variants) == 1 else None,
        "backend_settings_variants": backend_variants
        if len(backend_variants) > 1
        else [],
    }
    return result


class LoggingBackend:
    """Wraps any ModelBackend and logs all generate_chat / generate_chat_batch calls.

    Delegates to the inner backend and captures timing, full messages,
    sampling params, and full responses into the ExperimentLogger.
    """

    def __init__(self, backend: ModelBackend, logger: ExperimentLogger) -> None:
        self._backend = backend
        self._logger = logger
        self.enable_thinking = getattr(backend, "enable_thinking", None)

    @property
    def capabilities(self):
        return self._backend.capabilities

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> LoggingBackend:
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
        return self._backend.generate(prompts, params)

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        index = self._logger.next_index()
        timestamp = datetime.now().astimezone().isoformat()

        t0 = time.monotonic()
        try:
            result = self._backend.generate_chat(messages, params)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            record = {
                "index": index,
                "phase": self._logger.current_phase,
                "call_type": "generate_chat",
                "timestamp": timestamp,
                "elapsed_seconds": round(elapsed, 4),
                "messages": _serialize_messages(messages),
                "message_summary": _summarize_messages(messages),
                "sampling_params": _serialize_sampling_params(params),
                "backend": _serialize_backend_info(self),
                "error": _serialize_error(exc),
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            self._logger.log_call(record)
            _stderr_prompt_too_long(
                exc,
                self._logger.current_phase,
                self,
                params=params,
            )
            raise
        elapsed = time.monotonic() - t0

        record = {
            "index": index,
            "phase": self._logger.current_phase,
            "call_type": "generate_chat",
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed, 4),
            "messages": _serialize_messages(messages),
            "sampling_params": _serialize_sampling_params(params),
            "backend": _serialize_backend_info(self),
            "response": _serialize_result(result),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }
        self._logger.log_call(record)

        return result

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        index = self._logger.next_index()
        timestamp = datetime.now().astimezone().isoformat()

        t0 = time.monotonic()
        try:
            results = self._backend.generate_chat_batch(messages_batch, params)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            record = {
                "index": index,
                "phase": self._logger.current_phase,
                "call_type": "generate_chat_batch",
                "timestamp": timestamp,
                "elapsed_seconds": round(elapsed, 4),
                "batch_size": len(messages_batch),
                "messages_format": "unique_prompts",
                "messages": _serialize_unique_message_groups(messages_batch),
                "sampling_params": _serialize_sampling_params(params),
                "backend": _serialize_backend_info(self),
                "error": _serialize_error(exc),
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            self._logger.log_call(record)
            _stderr_prompt_too_long(
                exc,
                self._logger.current_phase,
                self,
                params=params,
            )
            raise
        elapsed = time.monotonic() - t0

        total_prompt = sum(r.prompt_tokens for r in results)
        total_completion = sum(r.completion_tokens for r in results)

        record = {
            "index": index,
            "phase": self._logger.current_phase,
            "call_type": "generate_chat_batch",
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed, 4),
            "batch_size": len(messages_batch),
            "messages_format": "unique_prompts",
            "messages": _serialize_unique_message_groups(messages_batch),
            "sampling_params": _serialize_sampling_params(params),
            "backend": _serialize_backend_info(self),
            "responses": [_serialize_result(r) for r in results],
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        }
        self._logger.log_call(record)

        return results


class _ProgressReporter:
    """Progress callback that emits when completion changes.

    With ``min_step_pct > 0`` intermediate updates are throttled so a line is
    logged only once the percentage advances by at least that many points since
    the last logged line. The first and final updates always log.
    """

    def __init__(
        self,
        *,
        experiment_logger: ExperimentLogger,
        name: str,
        logger_name: str,
        min_step_pct: float = 0.0,
    ) -> None:
        self._experiment_logger = experiment_logger
        self._name = name
        self._logger_name = logger_name
        self._min_step_pct = min_step_pct
        self._last_completed = -1
        self._last_logged_pct: float | None = None

    def __call__(self, completed: int, total: int) -> None:
        if total <= 0:
            return

        if self._last_completed == completed:
            return
        self._last_completed = completed

        is_final = completed >= total
        percent = completed / total * 100.0
        if (
            not is_final
            and self._min_step_pct > 0.0
            and self._last_logged_pct is not None
            and percent - self._last_logged_pct < self._min_step_pct
        ):
            return

        self._experiment_logger.log_progress(
            name=self._name,
            completed=completed,
            total=total,
            logger_name=self._logger_name,
            force=is_final,
        )
        self._last_logged_pct = percent
