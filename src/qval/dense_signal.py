"""Protocols and base class for dense signal functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from qval.types import EvaluationPoint, MethodContext


class DenseSignalFunction(Protocol):
    """Interface for methods that produce dense signal values.

    A dense signal function takes a (state, action, next_state) triple
    and returns a scalar signal value. The semantics of the value depend
    on the signal type (state-value, Q-value, advantage).
    """

    def __call__(self, state: Any, action: Any, next_state: Any) -> float: ...


@runtime_checkable
class BatchDenseSignalFunction(Protocol):
    """Optional interface for methods that support batch evaluation.

    Methods implementing this protocol can evaluate multiple points at once,
    which enables better GPU utilization for LLM-based methods.
    """

    def __call__(self, state: Any, action: Any, next_state: Any) -> float: ...

    def evaluate_batch(self, points: list[EvaluationPoint]) -> list[float]: ...


class DenseSignalMethod(ABC):
    """Base class for dense signal methods.

    Provides a richer abstraction than DenseSignalFunction for methods that
    need context (task description, reward description, example trajectories)
    and setup phases (e.g., LLM code generation).

    Subclasses implement ``evaluate()`` and inherit ``__call__`` (for
    DenseSignalFunction compatibility) and ``evaluate_batch`` (for
    BatchDenseSignalFunction compatibility) automatically.
    """

    def __init__(self, context: MethodContext) -> None:
        self.context = context

    @abstractmethod
    def evaluate(self, point: EvaluationPoint) -> float:
        """Evaluate a single (state, action, next_state) point."""
        ...

    def evaluate_batch(self, points: list[EvaluationPoint]) -> list[float]:
        """Evaluate multiple points. Override for batched implementations."""
        return [self.evaluate(p) for p in points]

    def __call__(self, state: Any, action: Any, next_state: Any) -> float:
        """DenseSignalFunction-compatible interface."""
        point = EvaluationPoint(
            state=state,
            action=action,
            next_state=next_state,
            trajectory_index=-1,
            step_index=-1,
        )
        return self.evaluate(point)
