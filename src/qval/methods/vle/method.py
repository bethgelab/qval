"""VLE-based dense signal method: cosine-similarity style value estimation."""

from __future__ import annotations

import logging
from typing import Callable

import torch

from llenvs.core.state import Observation, State

from qval.dense_signal import DenseSignalMethod
from qval.methods.serialization import extract_images
from qval.types import EvaluationPoint, MethodContext, SignalType

from .encoders import VLEEncoder, decode_image_content
from .heads import (
    HEAD_COSINE,
    HEAD_GOAL_BASELINE,
    HEAD_SOFTMAX_GOALS,
    HEAD_THRESHOLDED_BINARY,
    VLEScoringHead,
    build_head,
)

METHOD_VLM_RM = "vlm_rm"
METHOD_VLM_SOR = "vlm_sor"
VLE_METHOD_TYPES = {METHOD_VLM_RM, METHOD_VLM_SOR}

logger = logging.getLogger(__name__)


_IMAGE_SOURCES = ("state", "next_state")
_MULTI_IMAGE_STRATEGIES = ("last", "first", "mean")


class VLEMethod(DenseSignalMethod):
    """Score each evaluation point with a frozen VLM embedding model.

    Goal text comes from (in priority order) an explicit ``goal_text``,
    per-point ``obs.task.text`` when ``goal_per_point=True``, or
    ``context.task_description``. Points with no image (or no per-point
    goal text in per-point mode) return ``NaN``.
    """

    def __init__(
        self,
        *,
        encoder: VLEEncoder,
        head: VLEScoringHead,
        context: MethodContext,
        image_source: str = "state",
        multi_image_strategy: str = "last",
        goal_text: str | None = None,
        goal_per_point: bool = False,
    ) -> None:
        super().__init__(context)
        if image_source not in _IMAGE_SOURCES:
            raise ValueError(
                f"image_source must be one of {_IMAGE_SOURCES}, got {image_source!r}"
            )
        if multi_image_strategy not in _MULTI_IMAGE_STRATEGIES:
            raise ValueError(
                "multi_image_strategy must be one of "
                f"{_MULTI_IMAGE_STRATEGIES}, got {multi_image_strategy!r}"
            )
        if goal_per_point and goal_text is not None:
            raise ValueError(
                "VLEMethod: goal_text and goal_per_point are mutually exclusive"
            )
        resolved_static_goal = (
            None if goal_per_point else (goal_text or context.task_description)
        )
        if not goal_per_point and not resolved_static_goal:
            raise ValueError(
                "VLEMethod static goal mode requires either goal_text or "
                "MethodContext.task_description to be non-empty"
            )
        self._encoder = encoder
        self._head = head
        self._image_source = image_source
        self._multi_image_strategy = multi_image_strategy
        self._goal_per_point = bool(goal_per_point)
        self.last_aborted_indices: set[int] = set()
        head.prepare(encoder, resolved_static_goal)

    def evaluate(self, point: EvaluationPoint) -> float:
        return self.evaluate_batch([point])[0]

    def evaluate_batch(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        if not points:
            self.last_aborted_indices = set()
            return []

        total = len(points)
        per_point_pils: list[list] = []
        per_point_goal: list[str | None] = []
        aborted: set[int] = set()
        for idx, point in enumerate(points):
            source = point.state if self._image_source == "state" else point.next_state
            images = extract_images(source).state
            goal_text = (
                _extract_point_goal_text(point) if self._goal_per_point else None
            )
            missing_image = not images
            missing_goal = self._goal_per_point and not goal_text
            if missing_image or missing_goal:
                per_point_pils.append([])
                per_point_goal.append(None)
                aborted.add(idx)
                continue
            per_point_pils.append(self._select_pils(images))
            per_point_goal.append(goal_text)

        flat: list = []
        counts: list[int] = []
        for pils in per_point_pils:
            counts.append(len(pils))
            flat.extend(pils)

        results: list[float] = [float("nan")] * total
        if flat:
            embs = self._encoder.encode_image(flat)
            pooled_rows: list[torch.Tensor] = []
            valid_idx: list[int] = []
            offset = 0
            for i, n in enumerate(counts):
                if n == 0:
                    continue
                sub = embs[offset : offset + n]
                offset += n
                if self._multi_image_strategy == "mean" and n > 1:
                    pooled = sub.mean(dim=0)
                    norm = torch.linalg.norm(pooled)
                    if float(norm) > 1e-8:
                        pooled = pooled / norm
                else:
                    pooled = sub[0]
                pooled_rows.append(pooled)
                valid_idx.append(i)
            stacked = torch.stack(pooled_rows, dim=0)

            goal_emb: torch.Tensor | None = None
            goal_texts: list[str] | None = None
            if self._goal_per_point:
                goal_texts = [per_point_goal[i] for i in valid_idx]  # type: ignore[misc]
                assert all(t is not None for t in goal_texts)
                goal_emb = self._encoder.encode_text(goal_texts)

            scores = (
                self._head.score(stacked, goal_emb, goal_texts)
                .detach()
                .cpu()
                .tolist()
            )
            for idx, val in zip(valid_idx, scores):
                results[idx] = float(val)

        self.last_aborted_indices = aborted
        if aborted:
            logger.warning(
                "VLEMethod: %d/%d points had no images/goal and were marked NaN",
                len(aborted),
                total,
            )
        if progress_callback is not None:
            progress_callback(total, total)
        return results

    def _select_pils(self, images: tuple) -> list:
        if self._multi_image_strategy == "first":
            return [decode_image_content(images[0])]
        if self._multi_image_strategy == "last":
            return [decode_image_content(images[-1])]
        return [decode_image_content(img) for img in images]


def _extract_point_goal_text(point: EvaluationPoint) -> str | None:
    """Read per-episode goal text off ``point.state.observation.task.text``."""
    state = point.state
    if not isinstance(state, State):
        return None
    obs = state.observation
    if not isinstance(obs, Observation) or obs.task is None:
        return None
    text = obs.task.text
    return text.strip() if text and text.strip() else None


def build_vle_method(
    *,
    encoder: VLEEncoder,
    context: MethodContext,
    method_type: str,
    baseline_prompt: str | None = None,
    alpha: float | None = None,
    negative_goals: tuple[str, ...] | None = None,
    temperature: float | None = None,
    beta: float | None = None,
    image_source: str | None = None,
    multi_image_strategy: str | None = None,
    goal_text: str | None = None,
    goal_per_point: bool = False,
) -> VLEMethod:
    """Build a ``VLEMethod`` for ``vlm_rm`` (Rocamonde et al. 2023) or
    ``vlm_sor`` (Baumli et al. 2023).

    ``vlm_rm``: raw cosine by default; ``baseline_prompt`` + ``alpha``
    switch to goal-baseline regularization. ``vlm_sor``: softmax over
    goal + ``negative_goals``; setting ``beta`` switches to the
    thresholded binary variant. ``image_source`` defaults to
    ``next_state`` for Q and ``state`` otherwise.
    """
    head_name = _resolve_head_name(
        method_type=method_type,
        baseline_prompt=baseline_prompt,
        alpha=alpha,
        beta=beta,
    )
    head = build_head(
        head_name,
        baseline_prompt=baseline_prompt,
        alpha=alpha,
        negative_goals=negative_goals,
        temperature=temperature,
        beta=beta,
    )
    if image_source is None:
        image_source = (
            "next_state" if context.signal_type == SignalType.Q_VALUE else "state"
        )
    return VLEMethod(
        encoder=encoder,
        head=head,
        context=context,
        image_source=image_source,
        multi_image_strategy=multi_image_strategy or "last",
        goal_text=goal_text,
        goal_per_point=goal_per_point,
    )


def _resolve_head_name(
    *,
    method_type: str,
    baseline_prompt: str | None,
    alpha: float | None,
    beta: float | None,
) -> str:
    if method_type == METHOD_VLM_RM:
        if baseline_prompt is not None and alpha is not None:
            return HEAD_GOAL_BASELINE
        return HEAD_COSINE
    if method_type == METHOD_VLM_SOR:
        if beta is not None:
            return HEAD_THRESHOLDED_BINARY
        return HEAD_SOFTMAX_GOALS
    raise ValueError(
        f"Unknown VLE method_type: {method_type!r}. Valid: {sorted(VLE_METHOD_TYPES)}"
    )
