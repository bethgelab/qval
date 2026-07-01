"""Random baseline that emits uniform [0, 1) predictions, ignoring inputs."""

from __future__ import annotations

import random
from typing import Callable

from qval.dense_signal import DenseSignalMethod
from qval.types import EvaluationPoint, MethodContext


class RandomBaselineMethod(DenseSignalMethod):
    """Uniform random baseline.

    Returns ``random.Random(seed).random()`` — independent of state, action,
    next_state, signal type, or any other input. Provides a correlation floor
    against which informed methods can be compared.
    """

    def __init__(self, context: MethodContext, *, seed: int | None = None) -> None:
        super().__init__(context)
        self._rng = random.Random(seed)
        self._seed = seed
        self.last_aborted_indices: set[int] = set()

    def evaluate(self, point: EvaluationPoint) -> float:
        return self._rng.random()

    def evaluate_batch(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        self.last_aborted_indices = set()
        total = len(points)
        out = [self._rng.random() for _ in range(total)]
        if progress_callback is not None and total > 0:
            progress_callback(total, total)
        return out
