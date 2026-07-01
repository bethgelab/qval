"""Backend wrapper for truncation follow-up generations.

The wrapper activates only when ``SamplingParams.second_elicitation_suffix``
is set.  It delegates the initial generation unchanged, then performs a
no-thinking follow-up call for any result that stopped at ``MAX_TOKENS`` and
merges the two texts.
"""

from __future__ import annotations

from dataclasses import replace
import logging

from llenvs.core.tools import ToolDefinition
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    ModelBackend,
    PartialBatchError,
    SamplingParams,
    ScoringResult,
    StopReason,
)

from qval.error_handling import (
    generate_batch_with_transient_retry,
    is_recoverable_backend_error,
    is_transient_api_error,
    retry_on_transient,
)

logger = logging.getLogger(__name__)

_ELICITATION_PROMPT = (
    "Please provide the final answer now. Follow the formatting instructions "
    "specified above exactly."
)


def _is_max_tokens(result: GenerationResult) -> bool:
    finish_reason = result.finish_reason
    return finish_reason == StopReason.MAX_TOKENS or getattr(
        finish_reason, "name", None
    ) == "MAX_TOKENS"


def _build_elicitation_messages(
    messages: list[ChatMessage],
    first_result: GenerationResult,
    suffix: str,
) -> list[ChatMessage]:
    continued = list(messages)
    continued.append(
        ChatMessage(role="assistant", content=(first_result.text or "") + suffix)
    )
    continued.append(ChatMessage(role="user", content=_ELICITATION_PROMPT))
    return continued


def _elicitation_params(params: SamplingParams) -> SamplingParams:
    return replace(
        params,
        max_tokens=params.second_elicitation_max_tokens,
        thinking_budget=None,
        thinking_budget_per_block=False,
        thinking_budget_soft_ratio=None,
        thinking_budget_suffix=None,
        disable_thinking=True,
        second_elicitation_suffix=None,
    )


def _merge_elicitation(
    first: GenerationResult,
    second: GenerationResult,
    suffix: str,
) -> GenerationResult:
    merged_text = (first.text or "") + suffix + (second.text or "")
    merged_meta = {**first.metadata, **second.metadata, "second_elicitation": True}
    return GenerationResult(
        text=merged_text,
        finish_reason=second.finish_reason,
        tool_calls=second.tool_calls,
        token_logprobs=None,
        prompt_tokens=first.prompt_tokens + second.prompt_tokens,
        completion_tokens=first.completion_tokens + second.completion_tokens,
        metadata=merged_meta,
    )


class SecondElicitationBackendWrapper(ModelBackend):
    """Apply second elicitation to direct backend generation calls.

    This wrapper intentionally sits outside ``LoggingBackend`` in the pipeline
    scripts.  That way both the first generation and the follow-up generation
    are logged as ordinary model calls with their own messages, sampling
    params, and outputs.
    """

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend
        self.enable_thinking = getattr(backend, "enable_thinking", None)

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    @property
    def capabilities(self):
        return self._backend.capabilities

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "SecondElicitationBackendWrapper":
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
        first = self._backend.generate_chat(messages, params)
        if params.second_elicitation_suffix is None or not _is_max_tokens(first):
            return first

        return self._second_elicit_one(messages, first, params)

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        try:
            first_results = self._backend.generate_chat_batch(messages_batch, params)
        except PartialBatchError as exc:
            if params.second_elicitation_suffix is None:
                raise
            transformed_results = list(exc.results)
            for idx, value in enumerate(exc.results):
                if idx in exc.failures or isinstance(value, BaseException):
                    continue
                if not isinstance(value, GenerationResult):
                    continue
                if not _is_max_tokens(value):
                    continue
                transformed_results[idx] = self._second_elicit_one(
                    messages_batch[idx],
                    value,
                    params,
                )
            raise PartialBatchError(transformed_results, exc.failures) from exc

        if params.second_elicitation_suffix is None:
            return first_results

        return self._second_elicit_batch(messages_batch, first_results, params)

    def score_chat(
        self,
        messages: list[ChatMessage],
        continuation: str,
    ) -> ScoringResult:
        return self._backend.score_chat(messages, continuation)

    def score_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        continuations: list[str],
    ) -> list[ScoringResult]:
        return self._backend.score_chat_batch(messages_batch, continuations)

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> GenerationResult:
        return self._backend.generate_with_tools(messages, tools, params, tool_choice)

    def generate_with_tools_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        tools: list[ToolDefinition],
        params: SamplingParams,
        tool_choice: str = "auto",
    ) -> list[GenerationResult]:
        return self._backend.generate_with_tools_batch(
            messages_batch,
            tools,
            params,
            tool_choice,
        )

    def generate_chat_with_prefix(
        self,
        messages: list[ChatMessage],
        assistant_prefix: str,
        params: SamplingParams,
    ) -> GenerationResult:
        return self._backend.generate_chat_with_prefix(
            messages,
            assistant_prefix,
            params,
        )

    def generate_chat_with_prefix_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        assistant_prefixes: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        return self._backend.generate_chat_with_prefix_batch(
            messages_batch,
            assistant_prefixes,
            params,
        )

    def continue_from_prefix(
        self,
        prefix: str,
        params: SamplingParams,
        num_continuations: int = 1,
    ) -> list[GenerationResult]:
        return self._backend.continue_from_prefix(
            prefix,
            params,
            num_continuations,
        )

    def _second_elicit_one(
        self,
        messages: list[ChatMessage],
        first: GenerationResult,
        params: SamplingParams,
    ) -> GenerationResult:
        suffix = params.second_elicitation_suffix or ""
        elicitation_messages = _build_elicitation_messages(messages, first, suffix)
        elicitation_params = _elicitation_params(params)
        try:
            second = retry_on_transient(
                lambda: self._backend.generate_chat(
                    elicitation_messages,
                    elicitation_params,
                ),
                description="second elicitation",
            )
        except Exception as exc:
            if is_recoverable_backend_error(exc) or is_transient_api_error(exc):
                logger.warning(
                    "Second elicitation failed; keeping truncated first result: %s",
                    exc,
                )
                return first
            raise
        return _merge_elicitation(first, second, suffix)

    def _second_elicit_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        first_results: list[GenerationResult],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        needs_elicitation = [
            (idx, messages, result)
            for idx, (messages, result) in enumerate(
                zip(messages_batch, first_results, strict=False)
            )
            if _is_max_tokens(result)
        ]
        if not needs_elicitation:
            return first_results

        suffix = params.second_elicitation_suffix or ""
        elicitation_params = _elicitation_params(params)
        elicitation_messages = [
            _build_elicitation_messages(messages, first, suffix)
            for _, messages, first in needs_elicitation
        ]
        second_results = generate_batch_with_transient_retry(
            self._backend,
            elicitation_messages,
            elicitation_params,
        )

        merged_results = list(first_results)
        for (original_idx, _, first), second in zip(
            needs_elicitation,
            second_results,
            strict=False,
        ):
            if second is None:
                continue
            merged_results[original_idx] = _merge_elicitation(first, second, suffix)
        return merged_results


def wrap_with_second_elicitation(
    backend: ModelBackend,
) -> SecondElicitationBackendWrapper:
    """Return ``backend`` wrapped with generic second-elicitation handling."""
    return SecondElicitationBackendWrapper(backend)


__all__ = [
    "SecondElicitationBackendWrapper",
    "wrap_with_second_elicitation",
]
