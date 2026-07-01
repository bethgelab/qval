"""Scoring heads that turn VLE embeddings into a scalar signal."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .encoders import VLEEncoder


HEAD_COSINE = "cosine"
HEAD_GOAL_BASELINE = "goal_baseline"
HEAD_SOFTMAX_GOALS = "softmax_goals"
HEAD_THRESHOLDED_BINARY = "thresholded_binary"

VALID_HEADS = {
    HEAD_COSINE,
    HEAD_GOAL_BASELINE,
    HEAD_SOFTMAX_GOALS,
    HEAD_THRESHOLDED_BINARY,
}


class VLEScoringHead(ABC):
    """Post-processing applied to an image embedding to yield a scalar.

    Static mode: pass ``goal_text`` to ``prepare()``; score ignores
    ``goal_emb``. Per-point mode: omit ``goal_text`` and pass ``goal_emb``
    ``(B, D)`` to ``score()``.
    """

    @abstractmethod
    def prepare(self, encoder: VLEEncoder, goal_text: str | None = None) -> None: ...

    @abstractmethod
    def score(
        self,
        image_emb: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
        goal_texts: list[str] | None = None,
    ) -> torch.Tensor:
        """Map ``(B, D)`` image embeddings to a ``(B,)`` score tensor."""


def _resolve_goal(cached: torch.Tensor | None, passed: torch.Tensor | None, head_name: str) -> torch.Tensor:
    if passed is not None:
        return passed
    if cached is not None:
        return cached
    raise RuntimeError(
        f"{head_name}: no goal available — pass goal_text at prepare() "
        "for static mode, or goal_emb at score() for per-point mode."
    )


class CosineHead(VLEScoringHead):
    """Raw cosine similarity ``cos(goal, image)`` (Rocamonde et al. 2023)."""

    def __init__(self) -> None:
        self._goal: torch.Tensor | None = None

    def prepare(self, encoder: VLEEncoder, goal_text: str | None = None) -> None:
        self._goal = encoder.encode_text([goal_text])[0] if goal_text else None

    def score(
        self,
        image_emb: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
        goal_texts: list[str] | None = None,
    ) -> torch.Tensor:
        del goal_texts
        g = _resolve_goal(self._goal, goal_emb, "CosineHead")
        if g.ndim == 1:
            return image_emb @ g
        return (image_emb * g).sum(dim=-1)


class GoalBaselineRegHead(VLEScoringHead):
    """Goal-baseline regularization from Rocamonde et al. 2023.

    ``score = 1 - 0.5 * ||alpha * proj_L(s) + (1 - alpha) * s - g||^2``
    with ``L`` the line through goal ``g`` and baseline ``b``.
    """

    def __init__(self, *, alpha: float, baseline_prompt: str) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not baseline_prompt:
            raise ValueError("baseline_prompt must be a non-empty string")
        self._alpha = float(alpha)
        self._baseline_prompt = baseline_prompt
        self._goal: torch.Tensor | None = None
        self._baseline: torch.Tensor | None = None
        self._direction: torch.Tensor | None = None

    def prepare(self, encoder: VLEEncoder, goal_text: str | None = None) -> None:
        self._baseline = encoder.encode_text([self._baseline_prompt])[0]
        if goal_text is not None:
            self._goal = encoder.encode_text([goal_text])[0]
            diff = self._goal - self._baseline
            norm = torch.linalg.norm(diff)
            if float(norm) < 1e-8:
                raise ValueError(
                    "Goal and baseline embeddings are identical; cannot form "
                    "projection line for goal-baseline regularization."
                )
            self._direction = diff / norm
        else:
            self._goal = None
            self._direction = None

    def score(
        self,
        image_emb: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
        goal_texts: list[str] | None = None,
    ) -> torch.Tensor:
        del goal_texts
        if self._baseline is None:
            raise RuntimeError("GoalBaselineRegHead.prepare() was not called")
        g = _resolve_goal(self._goal, goal_emb, "GoalBaselineRegHead")
        b = self._baseline
        if g.ndim == 1:
            assert self._direction is not None
            d = self._direction
            proj = b + ((image_emb - b) @ d).unsqueeze(-1) * d
        else:
            diff = g - b
            norm = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-8)
            d = diff / norm
            proj = b + ((image_emb - b) * d).sum(dim=-1, keepdim=True) * d
        mixed = self._alpha * proj + (1.0 - self._alpha) * image_emb
        delta = mixed - g
        sq = (delta * delta).sum(dim=-1)
        return 1.0 - 0.5 * sq


class SoftmaxGoalsHead(VLEScoringHead):
    """Softmax-over-goals (Baumli et al. 2023, eq. 1).

    Returns the probability mass assigned to the task goal. Any negative
    whose string equals the current goal is dropped from the denominator
    so the sum matches ``Σ_{l' ∈ ℒ}`` when ℒ is passed via negatives.
    """

    def __init__(
        self,
        *,
        negative_goals: tuple[str, ...],
        temperature: float = 0.07,
    ) -> None:
        if not negative_goals:
            raise ValueError("negative_goals must be a non-empty tuple")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._negative_goals = tuple(negative_goals)
        self._temperature = float(temperature)
        self._goal: torch.Tensor | None = None
        self._negatives: torch.Tensor | None = None
        self._static_goal_text: str | None = None

    def prepare(self, encoder: VLEEncoder, goal_text: str | None = None) -> None:
        self._negatives = encoder.encode_text(list(self._negative_goals))
        self._goal = encoder.encode_text([goal_text])[0] if goal_text else None
        self._static_goal_text = goal_text

    def score(
        self,
        image_emb: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
        goal_texts: list[str] | None = None,
    ) -> torch.Tensor:
        if self._negatives is None:
            raise RuntimeError("SoftmaxGoalsHead.prepare() was not called")
        g = _resolve_goal(self._goal, goal_emb, "SoftmaxGoalsHead")
        n = self._negatives
        neg_logits = image_emb @ n.T
        if g.ndim == 1:
            goal_logits = (image_emb @ g).unsqueeze(-1)
        else:
            goal_logits = (image_emb * g).sum(dim=-1, keepdim=True)
        dedup_mask = self._build_dedup_mask(image_emb.shape[0], goal_texts)
        if dedup_mask is not None:
            neg_logits = neg_logits.masked_fill(dedup_mask, float("-inf"))
        logits = torch.cat([goal_logits, neg_logits], dim=1) / self._temperature
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 0]

    def _build_dedup_mask(
        self, batch_size: int, goal_texts: list[str] | None
    ) -> torch.Tensor | None:
        if goal_texts is None:
            if self._static_goal_text is None:
                return None
            goal_texts = [self._static_goal_text] * batch_size
        rows: list[list[bool]] = []
        any_true = False
        for text in goal_texts:
            row = [text == neg for neg in self._negative_goals]
            if any(row):
                any_true = True
            rows.append(row)
        if not any_true:
            return None
        return torch.tensor(rows, dtype=torch.bool, device=self._negatives.device)


class ThresholdedBinaryHead(SoftmaxGoalsHead):
    """Thresholded binary reward from Baumli et al. 2023: ``1[p(goal|image) > beta]``."""

    def __init__(
        self,
        *,
        negative_goals: tuple[str, ...],
        temperature: float,
        beta: float,
    ) -> None:
        super().__init__(negative_goals=negative_goals, temperature=temperature)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self._beta = float(beta)

    def score(
        self,
        image_emb: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
        goal_texts: list[str] | None = None,
    ) -> torch.Tensor:
        prob = super().score(image_emb, goal_emb, goal_texts)
        return (prob > self._beta).float()


def build_head(
    head: str,
    *,
    baseline_prompt: str | None = None,
    alpha: float | None = None,
    negative_goals: tuple[str, ...] | None = None,
    temperature: float | None = None,
    beta: float | None = None,
) -> VLEScoringHead:
    """Instantiate a scoring head from a string key + hyperparameters."""
    if head == HEAD_COSINE:
        return CosineHead()
    if head == HEAD_GOAL_BASELINE:
        if baseline_prompt is None or alpha is None:
            raise ValueError("head 'goal_baseline' requires baseline_prompt and alpha")
        return GoalBaselineRegHead(alpha=alpha, baseline_prompt=baseline_prompt)
    if head == HEAD_SOFTMAX_GOALS:
        if negative_goals is None or temperature is None:
            raise ValueError("head 'softmax_goals' requires negative_goals and temperature")
        return SoftmaxGoalsHead(negative_goals=negative_goals, temperature=temperature)
    if head == HEAD_THRESHOLDED_BINARY:
        if negative_goals is None or temperature is None or beta is None:
            raise ValueError(
                "head 'thresholded_binary' requires negative_goals, temperature, beta"
            )
        return ThresholdedBinaryHead(
            negative_goals=negative_goals, temperature=temperature, beta=beta,
        )
    raise ValueError(f"Unknown VLE head: {head!r}. Valid: {sorted(VALID_HEADS)}")
