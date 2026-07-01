"""LLM-based method that generates a Python signal function once, then executes it.

Note: This method operates on text-only state representations. Images from
visual environments are not passed to the generated Python function. For
vision-aware evaluation, use :class:`LLMDirectMethod` or
:class:`LLMRankingMethod` which support interleaved image content blocks.
"""

from __future__ import annotations

import collections
import itertools
import json
import logging
import math
import re
import statistics
import string
from typing import Any, Callable, NamedTuple

from llenvs.inference.protocol import ChatMessage, ModelBackend, SamplingParams

from qval.dense_signal import DenseSignalMethod
from qval.methods.code_parsing import extract_code, validate_code, validate_signature
from qval.methods.serialization import serialize_action, serialize_observation
from qval.prompt_presets import PromptPreset
from qval.prompts import (
    build_codegen_system_prompt,
    build_codegen_user_prompt,
    codegen_param_names,
)
from qval.types import EvaluationPoint, MethodContext

logger = logging.getLogger(__name__)

# Safe builtins for exec sandbox. Curated to block obvious footguns
# (open, exec, subprocess, introspection-based sandbox escapes) while
# allowing pure, common Python idioms that LLM-generated code naturally
# reaches for. This is footgun prevention, not a security boundary.
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    # Exception classes — required so try/except in generated code works.
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError,
    "RuntimeError": RuntimeError,
}

# Modules available in generated code
_ALLOWED_MODULES = {
    "collections": collections,
    "itertools": itertools,
    "json": json,
    "math": math,
    "re": re,
    "statistics": statistics,
    "string": string,
}

class CodeSample(NamedTuple):
    """A successfully compiled code sample from multi-sample generation."""

    code: str
    signal_fn: Callable[..., float]


class LLMCodeGenMethod(DenseSignalMethod):
    """Prompts an LLM once to generate a Python signal function, then executes it.

    On first evaluation (or explicit ``generate()``), asks the LLM to write a
    ``signal_function`` whose signature depends on the signal type:

    - STATE_VALUE / POTENTIAL: ``signal_function(state: str) -> float``
    - Q_VALUE / ADVANTAGE: ``signal_function(state: str, action: str) -> float``
    - SHAPED_REWARD: ``signal_function(state: str, action: str, next_state: str) -> float``

    The code is compiled in a restricted sandbox and cached. Subsequent
    evaluations just call the compiled function with the appropriate arguments.
    """

    def __init__(
        self,
        backend: ModelBackend,
        context: MethodContext,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_preset: PromptPreset,
    ) -> None:
        super().__init__(context)
        self._backend = backend
        self._sampling_params = sampling_params or SamplingParams()
        self._prompt_preset = prompt_preset
        self._generated_code: str | None = None
        self._signal_fn: Callable[..., float] | None = None
        self.last_aborted_indices: set[int] = set()
        self._param_names = codegen_param_names(
            context.signal_type, context.include_next_state,
        )

    @property
    def generated_code(self) -> str | None:
        """The generated Python code, or None if not yet generated."""
        return self._generated_code

    def generate(self) -> None:
        """Explicitly trigger code generation from the LLM."""
        from qval.error_handling import retry_on_transient

        messages = self._build_messages()
        result = retry_on_transient(
            lambda: self._backend.generate_chat(messages, self._sampling_params),
            description="codegen generate",
        )

        code = extract_code(result.text).code
        validate_code(code)
        self._signal_fn = self._compile_code(code, len(self._param_names))
        self._generated_code = code

    def evaluate(self, point: EvaluationPoint) -> float:
        if self._signal_fn is None:
            self.generate()

        args: list[str] = []
        for name in self._param_names:
            if name == "state":
                args.append(serialize_observation(point.state))
            elif name == "action":
                args.append(serialize_action(point.action))
            elif name == "next_state":
                args.append(serialize_observation(point.next_state))

        try:
            return float(self._signal_fn(*args))
        except Exception:
            logger.warning("Runtime error in generated signal function", exc_info=True)
            return math.nan

    def evaluate_batch(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        from qval.error_handling import (
            is_recoverable_backend_error,
            is_transient_api_error,
        )

        if not points:
            self.last_aborted_indices = set()
            return []

        if self._signal_fn is None:
            try:
                self.generate()
            except Exception as exc:
                if not (
                    is_recoverable_backend_error(exc)
                    or is_transient_api_error(exc)
                ):
                    raise
                logger.warning(
                    "Recoverable backend failure during code generation; "
                    "returning NaN for %d evaluation points",
                    len(points),
                    exc_info=True,
                )
                self.last_aborted_indices = set(range(len(points)))
                if progress_callback:
                    progress_callback(len(points), len(points))
                return [math.nan] * len(points)

        self.last_aborted_indices = set()
        results: list[float] = []
        total = len(points)
        for index, point in enumerate(points, start=1):
            results.append(self.evaluate(point))
            if progress_callback:
                progress_callback(index, total)
        return results

    def generate_samples(self, num_samples: int) -> list[CodeSample | None]:
        """Generate multiple independent code samples in one batched LLM call.

        Returns one slot per requested sample. Both recoverable backend
        failures and invalid generated code (parse, validation, compile,
        or signature errors — including partial/truncated output)
        produce ``None`` slots, matching the NaN-on-failure contract
        used elsewhere in the pipeline. Downstream, each ``None`` slot
        is persisted as an all-NaN prediction artifact marked with
        ``generation_aborted: true``.

        Does NOT modify ``self._signal_fn`` or ``self._generated_code``.
        """
        from qval.error_handling import generate_batch_with_transient_retry

        messages = self._build_messages()
        results = generate_batch_with_transient_retry(
            self._backend, [messages] * num_samples, self._sampling_params,
        )
        samples: list[CodeSample | None] = []
        for i, result in enumerate(results):
            if result is None:
                logger.warning(
                    "Sample %d/%d: backend error (skipped)", i + 1, num_samples
                )
                samples.append(None)
                continue
            try:
                code = extract_code(result.text).code
                validate_code(code)
                fn = self._compile_code(code, len(self._param_names))
            except ValueError as exc:
                # Covers parse errors (ast.parse), missing signal_function,
                # sandbox compile errors, and SignatureError (a ValueError
                # subclass). Truncated/partial model output ends up here.
                finish_reason = getattr(result, "finish_reason", None)
                logger.warning(
                    "Sample %d/%d: invalid generated code (finish_reason=%s): %s",
                    i + 1, num_samples, finish_reason, exc,
                )
                samples.append(None)
                continue
            samples.append(CodeSample(code=code, signal_fn=fn))
        return samples

    def evaluate_batch_with_fn(
        self,
        points: list[EvaluationPoint],
        signal_fn: Callable[..., float],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        """Evaluate points using an explicit compiled function."""
        results: list[float] = []
        total = len(points)
        for index, point in enumerate(points, start=1):
            args: list[str] = []
            for name in self._param_names:
                if name == "state":
                    args.append(serialize_observation(point.state))
                elif name == "action":
                    args.append(serialize_action(point.action))
                elif name == "next_state":
                    args.append(serialize_observation(point.next_state))
            try:
                results.append(float(signal_fn(*args)))
            except Exception:
                logger.warning("Runtime error in generated signal function", exc_info=True)
                results.append(math.nan)
            if progress_callback:
                progress_callback(index, total)
        return results

    def _build_messages(self) -> list[ChatMessage]:
        system_content = build_codegen_system_prompt(
            self.context, self._prompt_preset,
        )
        user_content = build_codegen_user_prompt(
            self.context.signal_type, self._prompt_preset,
            include_next_state=self.context.include_next_state,
            enable_thinking=self.context.enable_thinking,
        )

        return [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]

    @staticmethod
    def _compile_code(code: str, required_args: int = 3) -> Callable[..., float]:
        """Compile generated code in a restricted sandbox."""

        def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in _ALLOWED_MODULES:
                return _ALLOWED_MODULES[name]
            raise ImportError(f"Import of '{name}' is not allowed")

        namespace: dict[str, Any] = {
            "__builtins__": {**_SAFE_BUILTINS, "__import__": _restricted_import},
            **_ALLOWED_MODULES,
        }

        try:
            exec(compile(code, "<generated>", "exec"), namespace)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in generated code: {e}") from e
        except Exception as e:
            raise ValueError(f"Compilation error in generated code: {e}") from e

        fn = namespace["signal_function"]
        validate_signature(fn, required_args=required_args)

        return fn
