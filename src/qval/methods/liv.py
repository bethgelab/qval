"""LIV (Language-Image Value learning) dense signal method.

Ma et al. 2023 (arxiv.org/abs/2306.00958). CLIP RN50 backbone fine-tuned
on EpicKitchens with a goal-conditioned RL objective; scores states
against either an image goal (like VIP) or a text goal (like CLIP).
The paper's native similarity is cosine; negative L2 is also supported
per the reference ``sim()``.

Depends on the ``clip`` package from
https://github.com/openai/CLIP (installed as a git URL) — this is what
the official LIV repo uses to build the RN50 architecture. Checkpoint
(``jasonyma/LIV/model.pt``, ~244 MB) is pulled from HuggingFace Hub.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
from PIL.Image import Image as PILImage

from llenvs.core.state import ImageContent

from qval.config import BackendConfig
from qval.dense_signal import DenseSignalMethod
from qval.methods.serialization import extract_images
from qval.methods.vip import (
    _pick_state_image,
    make_env_image_resolver,
    make_trajectory_end_resolver,
)
from qval.methods.vle.encoders import decode_image_content
from qval.types import EvaluationPoint, MethodContext, SignalType

logger = logging.getLogger(__name__)


LIV_BACKEND_TYPE = "liv"
METHOD_LIV_IMG = "liv_img"
METHOD_LIV_TXT = "liv_txt"
LIV_METHOD_TYPES = {METHOD_LIV_IMG, METHOD_LIV_TXT}

LIV_SIM_COSINE = "cosine"
LIV_SIM_L2 = "l2"
LIV_SIMILARITIES = {LIV_SIM_COSINE, LIV_SIM_L2}

_LIV_IMAGE_STRATEGIES = ("last", "first")
_LIV_IMG_GOAL_SOURCES = ("trajectory_end", "env_image")

_LIV_HF_REPO = "jasonyma/LIV"
_LIV_HF_FILE = "model.pt"


def _resolve_device(device: str | None) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _strip_state_dict_prefixes(sd: dict, *prefixes: str) -> dict:
    out = dict(sd)
    for p in prefixes:
        out = {k[len(p) :] if k.startswith(p) else k: v for k, v in out.items()}
    return out


class LIVEncoder(ABC):
    """Unnormalized CLIP-like image and text embeddings in a shared space."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode_image(self, images: list[PILImage]) -> torch.Tensor: ...

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor: ...


class ClipLIVEncoder(LIVEncoder):
    """Wraps ``clip.load('RN50')`` with LIV's pretrained state_dict."""

    def __init__(
        self,
        *,
        device: str = "auto",
        model_id: str = "RN50",
        batch_size: int = 32,
    ) -> None:
        import clip
        from huggingface_hub import hf_hub_download

        resolved = _resolve_device(device)
        model, preprocess = clip.load(model_id, device=resolved, jit=False)

        ckpt_path = hf_hub_download(
            _LIV_HF_REPO,
            _LIV_HF_FILE,
            cache_dir=Path.home() / ".cache" / "huggingface" / "liv"
            if "HF_HOME" not in __import__("os").environ
            else None,
        )
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw["liv"] if isinstance(raw, dict) and "liv" in raw else raw
        sd = _strip_state_dict_prefixes(sd, "module.", "model.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logger.warning(
                "LIV state_dict load: missing=%s unexpected=%s",
                len(missing),
                len(unexpected),
            )

        self._clip = clip
        self._model = model.eval()
        self._preprocess = preprocess
        self._device = resolved
        self._batch_size = int(batch_size)
        self._dim = int(model.visual.output_dim)

    @property
    def dim(self) -> int:
        return self._dim

    @torch.inference_mode()
    def encode_image(self, images: list[PILImage]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i in range(0, len(images), self._batch_size):
            batch = images[i : i + self._batch_size]
            tensors = torch.stack([self._preprocess(img) for img in batch]).to(
                self._device
            )
            feats = self._model.encode_image(tensors).float().cpu()
            parts.append(feats)
        return torch.cat(parts, dim=0) if parts else torch.empty((0, self._dim))

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            tokens = self._clip.tokenize(batch, truncate=True).to(self._device)
            feats = self._model.encode_text(tokens).float().cpu()
            parts.append(feats)
        return torch.cat(parts, dim=0) if parts else torch.empty((0, self._dim))


def get_liv_encoder(cfg: BackendConfig) -> LIVEncoder:
    if cfg.type == LIV_BACKEND_TYPE:
        return ClipLIVEncoder(device=cfg.device)
    raise ValueError(
        f"Backend type {cfg.type!r} is not a LIV encoder. Expected {LIV_BACKEND_TYPE!r}."
    )


def _sim(a: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    """Similarity between per-row ``a`` and per-row ``b``, both ``(B, D)`` or
    ``a`` ``(B, D)`` with ``b`` broadcast ``(D,)``. Returns ``(B,)``."""
    if metric == LIV_SIM_COSINE:
        a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if b_n.ndim == 1:
            return a_n @ b_n
        return (a_n * b_n).sum(dim=-1)
    if metric == LIV_SIM_L2:
        if b.ndim == 1:
            return -torch.linalg.norm(a - b, dim=-1)
        return -torch.linalg.norm(a - b, dim=-1)
    raise ValueError(f"Unknown similarity: {metric!r}")


def make_static_text_resolver(text: str) -> Callable[[EvaluationPoint], str]:
    def resolve(_: EvaluationPoint) -> str:
        return text

    return resolve


def _point_task_text(point: EvaluationPoint) -> str | None:
    from llenvs.core.state import Observation, State

    state = point.state
    if not isinstance(state, State):
        return None
    obs = state.observation
    if not isinstance(obs, Observation) or obs.task is None:
        return None
    text = obs.task.text
    return text.strip() if text and text.strip() else None


def make_per_point_text_resolver() -> Callable[[EvaluationPoint], str | None]:
    def resolve(point: EvaluationPoint) -> str | None:
        return _point_task_text(point)

    return resolve


class LIVMethod(DenseSignalMethod):
    """LIV value / shaped reward against an image or text goal."""

    def __init__(
        self,
        *,
        encoder: LIVEncoder,
        context: MethodContext,
        goal_resolver: Callable[[EvaluationPoint], Any],
        goal_modality: str,
        similarity: str = LIV_SIM_COSINE,
        multi_image_strategy: str = "last",
    ) -> None:
        super().__init__(context)
        if goal_modality not in ("image", "text"):
            raise ValueError(
                f"goal_modality must be 'image' or 'text', got {goal_modality!r}"
            )
        if similarity not in LIV_SIMILARITIES:
            raise ValueError(
                f"similarity must be one of {sorted(LIV_SIMILARITIES)}, "
                f"got {similarity!r}"
            )
        if multi_image_strategy not in _LIV_IMAGE_STRATEGIES:
            raise ValueError(
                f"multi_image_strategy must be one of {_LIV_IMAGE_STRATEGIES}, "
                f"got {multi_image_strategy!r}"
            )
        if context.signal_type not in (
            SignalType.STATE_VALUE,
            SignalType.Q_VALUE,
            SignalType.SHAPED_REWARD,
        ):
            raise ValueError(
                "LIVMethod supports signal_type in "
                "{STATE_VALUE, Q_VALUE, SHAPED_REWARD}, "
                f"got {context.signal_type}"
            )
        self._encoder = encoder
        self._goal_resolver = goal_resolver
        self._goal_modality = goal_modality
        self._similarity = similarity
        self._strategy = multi_image_strategy
        self.last_aborted_indices: set[int] = set()

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

        signal = self.context.signal_type
        need_state = signal in (SignalType.STATE_VALUE, SignalType.SHAPED_REWARD)
        need_next = signal in (SignalType.Q_VALUE, SignalType.SHAPED_REWARD)

        per_point: list[tuple[ImageContent | None, ImageContent | None, Any]] = []
        aborted: set[int] = set()
        for i, point in enumerate(points):
            st = (
                _pick_state_image(point.state, self._strategy)
                if need_state
                else None
            )
            nx = (
                _pick_state_image(point.next_state, self._strategy)
                if need_next
                else None
            )
            goal = self._goal_resolver(point)
            bad = (
                (need_state and st is None)
                or (need_next and nx is None)
                or goal is None
                or (self._goal_modality == "text" and not goal)
            )
            if bad:
                aborted.add(i)
                per_point.append((None, None, None))
            else:
                per_point.append((st, nx, goal))

        img_cache: dict[str, torch.Tensor] = {}
        img_keys: list[str] = []
        img_pils: list[PILImage] = []
        text_cache: dict[str, torch.Tensor] = {}
        text_keys: list[str] = []

        def _register_image(img: ImageContent | None) -> None:
            if img is None or img.data in img_cache:
                return
            img_cache[img.data] = torch.empty(0)
            img_keys.append(img.data)
            img_pils.append(decode_image_content(img))

        for st, nx, goal in per_point:
            _register_image(st)
            _register_image(nx)
            if self._goal_modality == "image":
                _register_image(goal)
            elif goal is not None and goal not in text_cache:
                text_cache[goal] = torch.empty(0)
                text_keys.append(goal)

        if img_pils:
            embs = self._encoder.encode_image(img_pils)
            for k, e in zip(img_keys, embs):
                img_cache[k] = e
        if text_keys:
            embs = self._encoder.encode_text(text_keys)
            for k, e in zip(text_keys, embs):
                text_cache[k] = e

        results: list[float] = [float("nan")] * len(points)
        for i, (st, nx, goal) in enumerate(per_point):
            if i in aborted:
                continue
            g = img_cache[goal.data] if self._goal_modality == "image" else text_cache[goal]
            if signal == SignalType.STATE_VALUE:
                s = img_cache[st.data]
                results[i] = float(_sim(s.unsqueeze(0), g, self._similarity)[0])
            elif signal == SignalType.Q_VALUE:
                s = img_cache[nx.data]
                results[i] = float(_sim(s.unsqueeze(0), g, self._similarity)[0])
            else:
                s = img_cache[st.data]
                n = img_cache[nx.data]
                v_s = float(_sim(s.unsqueeze(0), g, self._similarity)[0])
                v_n = float(_sim(n.unsqueeze(0), g, self._similarity)[0])
                results[i] = v_n - v_s

        self.last_aborted_indices = aborted
        if aborted:
            logger.warning(
                "LIVMethod: %d/%d points aborted (missing image or goal)",
                len(aborted),
                len(points),
            )
        if progress_callback is not None:
            progress_callback(len(points), len(points))
        return results


def build_liv_method(
    *,
    encoder: LIVEncoder,
    context: MethodContext,
    method_type: str,
    similarity: str | None = None,
    goal_source: str | None = None,
    env_goal_image_path: str | None = None,
    goal_text: str | None = None,
    goal_per_point: bool = False,
    multi_image_strategy: str | None = None,
    dataset: Any = None,
) -> LIVMethod:
    """Build a LIVMethod for ``liv_img`` or ``liv_txt``."""
    sim = similarity or LIV_SIM_COSINE
    strat = multi_image_strategy or "last"

    if method_type == METHOD_LIV_IMG:
        if goal_source == "trajectory_end":
            if dataset is None:
                raise ValueError(
                    "liv_img with goal_source='trajectory_end' requires a dataset"
                )
            resolver = make_trajectory_end_resolver(dataset, strategy=strat)
        elif goal_source == "env_image":
            if not env_goal_image_path:
                raise ValueError(
                    "liv_img with goal_source='env_image' requires liv_env_goal_image_path"
                )
            resolver = make_env_image_resolver(env_goal_image_path)
        else:
            raise ValueError(
                f"Unknown liv_goal_source: {goal_source!r}. "
                f"Valid: {list(_LIV_IMG_GOAL_SOURCES)}"
            )
        return LIVMethod(
            encoder=encoder,
            context=context,
            goal_resolver=resolver,
            goal_modality="image",
            similarity=sim,
            multi_image_strategy=strat,
        )

    if method_type == METHOD_LIV_TXT:
        if goal_text and goal_per_point:
            raise ValueError(
                "liv_txt: liv_goal_text and liv_goal_per_point are mutually exclusive"
            )
        if goal_per_point:
            resolver = make_per_point_text_resolver()
        elif goal_text:
            resolver = make_static_text_resolver(goal_text)
        elif context.task_description:
            resolver = make_static_text_resolver(context.task_description)
        else:
            raise ValueError(
                "liv_txt needs a goal: set liv_goal_text, liv_goal_per_point, "
                "or context.task_description"
            )
        return LIVMethod(
            encoder=encoder,
            context=context,
            goal_resolver=resolver,
            goal_modality="text",
            similarity=sim,
            multi_image_strategy=strat,
        )

    raise ValueError(
        f"Unknown LIV method_type: {method_type!r}. Valid: {sorted(LIV_METHOD_TYPES)}"
    )
