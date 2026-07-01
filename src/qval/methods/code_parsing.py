"""Code extraction, static validation, and signature validation for LLM-generated code."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass

from llenvs.core.cleaning import strip_special_tokens, strip_thinking_tokens

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:python)?\s*\n(.*?)(?:```|\Z)",
    re.DOTALL,
)


class SignatureError(ValueError):
    """Raised when a function signature doesn't match the expected arity."""


@dataclass(frozen=True)
class CodeParsingResult:
    """Result of extracting code from an LLM response.

    Attributes:
        code: The extracted (and possibly concatenated) code string.
        blocks: Individual code blocks found in the response.
        source: How the code was found — ``"fenced"`` or ``"unfenced"``.
    """

    code: str
    blocks: list[str]
    source: str


def extract_code(response: str) -> CodeParsingResult:
    """Extract Python code from an LLM response.

    Strips ``<think>…</think>`` reasoning blocks (including unclosed
    blocks from MAX_TOKENS truncation) and special tokens before
    matching. Accepts either a closed code fence
    (```` ```python ... ``` ````) or an unclosed opening fence from a
    truncated response — in the second case everything between the
    opening fence and end of text is treated as the code block.
    If multiple blocks are found, they are concatenated with blank lines.
    If no fenced blocks are found, falls back to the stripped raw
    response.
    """
    cleaned = strip_special_tokens(strip_thinking_tokens(response))
    matches = _CODE_FENCE_PATTERN.findall(cleaned)
    if matches:
        blocks = [m.strip() for m in matches if m.strip()]
        if blocks:
            return CodeParsingResult(
                code="\n\n".join(blocks),
                blocks=blocks,
                source="fenced",
            )

    stripped = cleaned.strip()
    return CodeParsingResult(
        code=stripped,
        blocks=[stripped] if stripped else [],
        source="unfenced",
    )


def validate_code(code: str) -> None:
    """Validate that code is syntactically correct and defines a top-level ``signal_function``.

    Raises:
        ValueError: If the code has syntax errors or lacks a top-level ``signal_function``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated code: {e}") from e

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "signal_function":
            return

    raise ValueError(
        "Generated code does not define a top-level 'signal_function'."
    )


def validate_signature(
    fn: object,
    *,
    required_args: int = 3,
    function_name: str = "signal_function",
) -> None:
    """Validate that a callable accepts the expected number of positional arguments.

    A function is accepted if it can be called with exactly ``required_args``
    positional arguments. This means:

    - It has exactly ``required_args`` required positional params, OR
    - It has fewer required positional params but uses ``*args``, OR
    - It has ``required_args`` or more positional params where extras have defaults.

    Raises:
        SignatureError: If the function cannot accept ``required_args`` positional args.
    """
    sig = inspect.signature(fn)  # type: ignore[arg-type]

    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    required_positional = 0
    total_positional = 0
    has_var_positional = False

    for param in sig.parameters.values():
        if param.kind in positional_kinds:
            total_positional += 1
            if param.default is inspect.Parameter.empty:
                required_positional += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True

    if has_var_positional:
        if required_positional <= required_args:
            return
        raise SignatureError(
            f"'{function_name}' has {required_positional} required positional "
            f"parameters before *args, but must accept {required_args} "
            f"positional arguments."
        )

    if required_positional > required_args:
        raise SignatureError(
            f"'{function_name}' requires {required_positional} positional "
            f"arguments, but must accept exactly {required_args}."
        )

    if total_positional < required_args:
        raise SignatureError(
            f"'{function_name}' accepts at most {total_positional} positional "
            f"arguments, but must accept {required_args}."
        )
