"""Vision-language embedding (VLE) dense signal methods.

Zero-shot estimators with frozen CLIP/SigLIP encoders. Implements
``vlm_rm`` (Rocamonde et al. 2023) and ``vlm_sor`` (Baumli et al. 2023).
"""

from .encoders import (
    VLE_BACKEND_TYPE,
    HFCLIPEncoder,
    VLEEncoder,
    decode_image_content,
    get_vle_encoder,
)
from .heads import (
    HEAD_COSINE,
    HEAD_GOAL_BASELINE,
    HEAD_SOFTMAX_GOALS,
    HEAD_THRESHOLDED_BINARY,
    VALID_HEADS,
    CosineHead,
    GoalBaselineRegHead,
    SoftmaxGoalsHead,
    ThresholdedBinaryHead,
    VLEScoringHead,
    build_head,
)
from .method import (
    METHOD_VLM_RM,
    METHOD_VLM_SOR,
    VLE_METHOD_TYPES,
    VLEMethod,
    build_vle_method,
)

__all__ = [
    "HEAD_COSINE",
    "HEAD_GOAL_BASELINE",
    "HEAD_SOFTMAX_GOALS",
    "HEAD_THRESHOLDED_BINARY",
    "HFCLIPEncoder",
    "METHOD_VLM_RM",
    "METHOD_VLM_SOR",
    "VALID_HEADS",
    "VLE_BACKEND_TYPE",
    "VLE_METHOD_TYPES",
    "VLEEncoder",
    "VLEMethod",
    "VLEScoringHead",
    "CosineHead",
    "GoalBaselineRegHead",
    "SoftmaxGoalsHead",
    "ThresholdedBinaryHead",
    "build_head",
    "build_vle_method",
    "decode_image_content",
    "get_vle_encoder",
]
