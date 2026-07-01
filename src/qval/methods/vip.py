"""VIP (Value-Implicit Pre-Training) dense signal method.

Ma et al. 2022 (arxiv.org/abs/2210.00030). Frozen ResNet-50 pre-trained on
Ego4D; value is negative L2 distance in embedding space between current
state and goal images. Image-only, no text.

Vendored checkpoint (CC BY-NC 4.0): the one-time download is hard-coded
to the PyTorch S3 URL used by the official ``vip.load_vip``. No runtime
dependency on the ``vip`` pip package.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from PIL import Image
from PIL.Image import Image as PILImage
from torch.hub import download_url_to_file
from torchvision import transforms
from torchvision.models import resnet50

from llenvs.core.state import ImageContent, Observation, State

from qval.config import BackendConfig
from qval.dense_signal import DenseSignalMethod
from qval.methods.serialization import extract_images
from qval.methods.vle.encoders import decode_image_content
from qval.types import EvaluationPoint, MethodContext, SignalType

logger = logging.getLogger(__name__)


VIP_BACKEND_TYPE = "vip"
METHOD_VIP = "vip"

_VIP_CHECKPOINT_URL = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
_VIP_EMBED_DIM = 1024
_VIP_GOAL_SOURCE_TRAJ_END = "trajectory_end"
_VIP_GOAL_SOURCE_ENV_IMG = "env_image"
_VIP_GOAL_SOURCES = (_VIP_GOAL_SOURCE_TRAJ_END, _VIP_GOAL_SOURCE_ENV_IMG)

_VIP_IMAGE_STRATEGIES = ("last", "first")


class VIPEncoder(ABC):
    """Produces raw (unnormalized) image embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode_image(self, images: list[PILImage]) -> torch.Tensor:
        """Return image embeddings of shape ``(N, dim)``."""


def _resolve_device(device: str | None) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _checkpoint_cache_path() -> Path:
    root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return Path(root) / "vip" / "model.pt"


def _strip_state_dict_prefix(sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


class ResNetVIPEncoder(VIPEncoder):
    """ResNet-50 + 1024-d projection, loaded from the official VIP checkpoint."""

    def __init__(
        self,
        *,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        resolved = _resolve_device(device)
        convnet = resnet50(weights=None)
        convnet.fc = nn.Linear(convnet.fc.in_features, _VIP_EMBED_DIM)

        path = _checkpoint_cache_path()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading VIP checkpoint to %s", path)
            download_url_to_file(_VIP_CHECKPOINT_URL, str(path))
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = raw["vip"] if isinstance(raw, dict) and "vip" in raw else raw
        state_dict = _strip_state_dict_prefix(state_dict, "module.")
        state_dict = _strip_state_dict_prefix(state_dict, "convnet.")
        missing, unexpected = convnet.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                "VIP state_dict load: missing=%s unexpected=%s",
                len(missing),
                len(unexpected),
            )

        convnet.to(resolved).eval()
        self._convnet = convnet
        self._device = resolved
        self._batch_size = int(batch_size)
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @property
    def dim(self) -> int:
        return _VIP_EMBED_DIM

    @torch.inference_mode()
    def encode_image(self, images: list[PILImage]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i in range(0, len(images), self._batch_size):
            batch = images[i : i + self._batch_size]
            tensors = torch.stack([self._transform(img) for img in batch]).to(
                self._device
            )
            feats = self._convnet(tensors).float().cpu()
            parts.append(feats)
        return torch.cat(parts, dim=0) if parts else torch.empty((0, self.dim))


def get_vip_encoder(cfg: BackendConfig) -> VIPEncoder:
    """Dispatch a ``BackendConfig`` entry to a concrete ``VIPEncoder``."""
    if cfg.type == VIP_BACKEND_TYPE:
        return ResNetVIPEncoder(device=cfg.device)
    raise ValueError(
        f"Backend type {cfg.type!r} is not a VIP encoder. Expected {VIP_BACKEND_TYPE!r}."
    )


def _pil_to_image_content(pil: PILImage) -> ImageContent:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return ImageContent(
        data=base64.b64encode(buf.getvalue()).decode("ascii"),
        media_type="image/png",
        source="task",
    )


def _pick_state_image(state: Any, strategy: str) -> ImageContent | None:
    imgs = extract_images(state).state
    if not imgs:
        return None
    if strategy == "first":
        return imgs[0]
    return imgs[-1]


def _trajectory_end_state(trajectory_result: Any) -> Any | None:
    """Return the final ``State`` of a trajectory, or ``None`` if unavailable."""
    traj = getattr(trajectory_result, "trajectory", None)
    if traj is None:
        return None
    transitions = getattr(traj, "transitions", None)
    if transitions:
        return transitions[-1].next_state
    return getattr(traj, "initial_state", None)


def make_trajectory_end_resolver(
    dataset: Any,
    strategy: str = "last",
) -> Callable[[EvaluationPoint], ImageContent | None]:
    """Build ``point -> goal ImageContent`` from each trajectory's final state image."""
    index: dict[int, ImageContent] = {}
    for i, tr in enumerate(dataset.trajectory_results):
        state = _trajectory_end_state(tr)
        if state is None:
            continue
        img = _pick_state_image(state, strategy)
        if img is not None:
            index[i] = img

    def resolve(point: EvaluationPoint) -> ImageContent | None:
        return index.get(point.trajectory_index)

    return resolve


def make_env_image_resolver(
    path: Path | str,
) -> Callable[[EvaluationPoint], ImageContent]:
    """Build ``point -> goal ImageContent`` loading a single PNG from ``path``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"VIP env goal image not found: {p}")
    pil = Image.open(p).convert("RGB")
    goal = _pil_to_image_content(pil)

    def resolve(_: EvaluationPoint) -> ImageContent:
        return goal

    return resolve


class VIPMethod(DenseSignalMethod):
    """VIP value / potential-shaped reward via image-to-goal L2 distance."""

    def __init__(
        self,
        *,
        encoder: VIPEncoder,
        context: MethodContext,
        goal_resolver: Callable[[EvaluationPoint], ImageContent | None],
        multi_image_strategy: str = "last",
    ) -> None:
        super().__init__(context)
        if multi_image_strategy not in _VIP_IMAGE_STRATEGIES:
            raise ValueError(
                f"multi_image_strategy must be one of {_VIP_IMAGE_STRATEGIES}, "
                f"got {multi_image_strategy!r}"
            )
        if context.signal_type not in (
            SignalType.STATE_VALUE,
            SignalType.Q_VALUE,
            SignalType.SHAPED_REWARD,
        ):
            raise ValueError(
                f"VIPMethod supports signal_type in "
                "{STATE_VALUE, Q_VALUE, SHAPED_REWARD}, "
                f"got {context.signal_type}"
            )
        self._encoder = encoder
        self._goal_resolver = goal_resolver
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

        per_point: list[tuple[ImageContent | None, ImageContent | None, ImageContent | None]] = []
        aborted: set[int] = set()
        for i, point in enumerate(points):
            st = _pick_state_image(point.state, self._strategy) if need_state else None
            nx = _pick_state_image(point.next_state, self._strategy) if need_next else None
            gl = self._goal_resolver(point)
            bad = (
                (need_state and st is None)
                or (need_next and nx is None)
                or gl is None
            )
            if bad:
                aborted.add(i)
                per_point.append((None, None, None))
            else:
                per_point.append((st, nx, gl))

        cache: dict[str, torch.Tensor] = {}
        unique_keys: list[str] = []
        unique_pils: list[PILImage] = []
        for st, nx, gl in per_point:
            for img in (st, nx, gl):
                if img is None or img.data in cache:
                    continue
                cache[img.data] = torch.empty(0)
                unique_keys.append(img.data)
                unique_pils.append(decode_image_content(img))

        if unique_pils:
            embs = self._encoder.encode_image(unique_pils)
            for k, e in zip(unique_keys, embs):
                cache[k] = e

        results: list[float] = [float("nan")] * len(points)
        for i, (st, nx, gl) in enumerate(per_point):
            if i in aborted:
                continue
            assert gl is not None
            g = cache[gl.data]
            if signal == SignalType.STATE_VALUE:
                assert st is not None
                results[i] = -float(torch.linalg.norm(cache[st.data] - g))
            elif signal == SignalType.Q_VALUE:
                assert nx is not None
                results[i] = -float(torch.linalg.norm(cache[nx.data] - g))
            else:
                assert st is not None and nx is not None
                v_s = -float(torch.linalg.norm(cache[st.data] - g))
                v_n = -float(torch.linalg.norm(cache[nx.data] - g))
                results[i] = v_n - v_s

        self.last_aborted_indices = aborted
        if aborted:
            logger.warning(
                "VIPMethod: %d/%d points aborted (missing state/next_state/goal image)",
                len(aborted),
                len(points),
            )
        if progress_callback is not None:
            progress_callback(len(points), len(points))
        return results


def build_vip_method(
    *,
    encoder: VIPEncoder,
    context: MethodContext,
    goal_source: str,
    env_goal_image_path: str | None = None,
    multi_image_strategy: str | None = None,
    dataset: Any = None,
) -> VIPMethod:
    """Build a VIPMethod with the configured goal-image resolver."""
    if goal_source == _VIP_GOAL_SOURCE_TRAJ_END:
        if dataset is None:
            raise ValueError(
                "vip_goal_source='trajectory_end' requires a dataset object"
            )
        resolver = make_trajectory_end_resolver(
            dataset, strategy=multi_image_strategy or "last"
        )
    elif goal_source == _VIP_GOAL_SOURCE_ENV_IMG:
        if not env_goal_image_path:
            raise ValueError(
                "vip_goal_source='env_image' requires vip_env_goal_image_path"
            )
        resolver = make_env_image_resolver(env_goal_image_path)
    else:
        raise ValueError(
            f"Unknown vip_goal_source: {goal_source!r}. Valid: {list(_VIP_GOAL_SOURCES)}"
        )
    return VIPMethod(
        encoder=encoder,
        context=context,
        goal_resolver=resolver,
        multi_image_strategy=multi_image_strategy or "last",
    )
