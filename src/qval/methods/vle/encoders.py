"""Vision-language embedding encoders and image decoding."""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as PILImage

from llenvs.core.state import ImageContent
from qval.config import BackendConfig


VLE_BACKEND_TYPE = "huggingface_vle"


class VLEEncoder(ABC):
    """Produces L2-normalized text and image embeddings in a shared space."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> str: ...

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalized text embeddings, shape ``(N, dim)``."""

    @abstractmethod
    def encode_image(self, images: list[PILImage]) -> torch.Tensor:
        """Return L2-normalized image embeddings, shape ``(N, dim)``."""


def decode_image_content(img: ImageContent) -> PILImage:
    """Decode a base64 ``ImageContent`` into a PIL RGB image."""
    raw = base64.b64decode(img.data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _resolve_device(device: str | None) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _as_tensor(feats) -> torch.Tensor:
    """Unwrap a HF model output to the pooled embedding tensor."""
    if isinstance(feats, torch.Tensor):
        return feats
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(feats, attr, None)
        if isinstance(val, torch.Tensor):
            if val.ndim == 3:
                return val[:, 0, :]
            return val
    raise TypeError(
        f"Cannot extract an embedding tensor from {type(feats).__name__}"
    )


def _resolve_dtype(dtype: str | None, device: str) -> torch.dtype:
    if dtype in (None, "", "auto"):
        return torch.float16 if device.startswith("cuda") else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(
            f"Unsupported VLE dtype: {dtype!r}. Valid: {sorted(mapping)}"
        )
    return mapping[dtype]


class HFCLIPEncoder(VLEEncoder):
    """CLIP/SigLIP-family encoder loaded via ``transformers.AutoModel``."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        resolved_device = _resolve_device(device)
        torch_dtype = _resolve_dtype(dtype, resolved_device)
        self._model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        self._model.to(resolved_device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._device = resolved_device
        self._batch_size = int(batch_size)
        if not (
            hasattr(self._model, "get_text_features")
            and hasattr(self._model, "get_image_features")
        ):
            raise TypeError(
                f"Model {model_name!r} does not expose get_text_features / "
                "get_image_features; only CLIP/SigLIP-family models are supported."
            )
        with torch.inference_mode():
            probe = self._processor(
                text=["."], return_tensors="pt", padding=True
            ).to(resolved_device)
            feats = _as_tensor(self._model.get_text_features(**probe))
        self._dim = int(feats.shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return self._device

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                inputs = self._processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._device)
                feats = _as_tensor(self._model.get_text_features(**inputs))
                parts.append(F.normalize(feats.float(), dim=-1))
        return torch.cat(parts, dim=0)

    def encode_image(self, images: list[PILImage]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, len(images), self._batch_size):
                batch = images[i : i + self._batch_size]
                inputs = self._processor(
                    images=batch, return_tensors="pt"
                ).to(self._device)
                feats = _as_tensor(self._model.get_image_features(**inputs))
                parts.append(F.normalize(feats.float(), dim=-1))
        return torch.cat(parts, dim=0)


def get_vle_encoder(cfg: BackendConfig) -> VLEEncoder:
    """Dispatch a ``BackendConfig`` entry to a concrete ``VLEEncoder``."""
    if cfg.type == VLE_BACKEND_TYPE:
        return HFCLIPEncoder(
            model_name=cfg.model,
            device=cfg.device,
            dtype=cfg.dtype,
        )
    raise ValueError(
        f"Backend type {cfg.type!r} is not a VLE encoder. "
        f"Expected {VLE_BACKEND_TYPE!r}."
    )
