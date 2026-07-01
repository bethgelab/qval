"""The single source of truth: concrete model / environment / method catalogs.

Editing this file (and regenerating configs) is the intended way to add or
change an experiment dimension. Everything downstream — naming, config
generation, evaluation discovery, and the paper-figure registries — derives
from here.

The data is seeded to reproduce the on-disk naming conventions exactly (see
``docs/naming_conventions.md`` and the per-catalog layouts under
``catalogs/<name>/data/``).
"""

from __future__ import annotations

from qval.registry.atoms import (
    EnvironmentSpec,
    MethodSpec,
    Modality,
    ModelSpec,
    Registry,
    Signal,
)

# Catalog identity. The active-catalog machinery prefers this over the
# directory name, so the bundled benchmark keeps the name ``qval_benchmark``
# regardless of where the file lives on disk.
NAME = "qval_benchmark"

QV = Signal.QV
SV = Signal.SV
TEXT = Modality.TEXT
VISION = Modality.VISION

# Canonical LLM eval actors, in the order used by the historical
# ``EVAL_ACTORS`` vocabulary (order is cosmetic — parsing is longest-match).
_LLM_ACTORS_VISION = ("g4-26-or", "g4-31-or", "q35-9-or", "q35-27-or", "q35-35-or", "q35-122-or")
# Text and vision LLM actors are the same set.
_LLM_ACTORS_TEXT = _LLM_ACTORS_VISION


MODELS: tuple[ModelSpec, ...] = (
    # ---- LLM eval actors. backend_base = text/thinking (direct/gvl/etc);
    #      backend_codegen = text/thinking codegen (codegen/eureka);
    #      backend_vision = thinking vision (OpenRouter provider-pinned for
    #      multi-image prompts). See configs/backends.yaml for the full matrix. ----
    ModelSpec("g4-26-or", "Gemma4 26B-A4B", supports_vision=True,
              backend_base="gemma4_26_thinking_text_or",
              backend_codegen="gemma4_26_thinking_codegen_text_or",
              backend_vision="gemma4_26_thinking_vision_or"),
    ModelSpec("g4-31-or", "Gemma4 31B", supports_vision=True,
              backend_base="gemma4_31_thinking_text_or",
              backend_codegen="gemma4_31_thinking_codegen_text_or",
              backend_vision="gemma4_31_thinking_vision_or"),
    ModelSpec("q35-9-or", "Qwen3.5 9B", supports_vision=True,
              backend_base="qwen35_9_thinking_text_or",
              backend_codegen="qwen35_9_thinking_codegen_text_or",
              backend_vision="qwen35_9_thinking_vision_or"),
    ModelSpec("q35-27-or", "Qwen3.5 27B", supports_vision=True,
              backend_base="qwen35_27_thinking_text_or",
              backend_codegen="qwen35_27_thinking_codegen_text_or",
              backend_vision="qwen35_27_thinking_vision_or"),
    ModelSpec("q35-35-or", "Qwen3.5 35B-A3B", supports_vision=True,
              backend_base="qwen35_35_thinking_text_or",
              backend_codegen="qwen35_35_thinking_codegen_text_or",
              backend_vision="qwen35_35_thinking_vision_or"),
    ModelSpec("q35-122-or", "Qwen3.5 122B-A10B", supports_vision=True,
              backend_base="qwen35_122_thinking_text_or",
              backend_codegen="qwen35_122_thinking_codegen_text_or",
              backend_vision="qwen35_122_thinking_vision_or"),
    # ---- non-LLM eval actors. clip/siglip backbones feed the vle-* methods;
    #      "self" carries no backend (vip/liv supply their own, see METHODS). ----
    ModelSpec("clip", "CLIP", is_embedding_backbone=True, supports_vision=True,
              supports_thinking=False, backend_base="clip_vit_l14"),
    ModelSpec("siglip", "SigLIP", is_embedding_backbone=True, supports_vision=True,
              supports_thinking=False, backend_base="siglip_base"),
    ModelSpec("self", "Self", is_pretrained=True, supports_vision=True, supports_thinking=False),
    # ---- ground-truth-producing actors ----
    ModelSpec("scripted", "scripted", is_gt_only=True, supports_thinking=False),
    ModelSpec("codex-55", "Codex 5.5", is_gt_only=True, supports_thinking=False),
    ModelSpec("ds32", "DeepSeek v3.2", is_gt_only=True, supports_thinking=False),
    ModelSpec("opus-47", "Opus 4.7", is_gt_only=True, supports_thinking=False),
    ModelSpec("random", "Random", is_gt_only=True, supports_thinking=False),
)


# ---------------------------------------------------------------------------
# Environment-specific vle-* hyperparameters. The negative-goal lists and the
# baseline prompt are dataset-specific and are NOT recoverable from the
# canonical artifacts (the producing configs are gone). Pre-filled best-effort
# and flagged — review before running.
# ---------------------------------------------------------------------------
# Recovered verbatim from the surviving alfworld_vision vle_all.yaml — the unique
# task objectives in the canonical all-types-40ms dataset.
_ALFWORLD_VLE_NEGATIVES = (
    "put a mug in desk.",
    "put a pencil in shelf.",
    "put a peppershaker in drawer.",
    "put a saltshaker in drawer.",
    "put a soapbottle in toilet.",
    "put a vase in safe.",
    "put some mug on desk.",
    "put some peppershaker on drawer.",
    "put some saltshaker on cabinet.",
    "put some saltshaker on drawer.",
    "put some soapbottle on toilet.",
    "put some vase on safe.",
    "put some watch on safe.",
)
_ALFWORLD_VLE_BASELINE = "a generic household room"

# frozen_lake / open_apps vle negatives + baseline, recovered verbatim from the
# surviving 8x8_vle_all.yaml and vle_all.yaml configs.
_FROZEN_LAKE_VLE_NEGATIVES = (
    "the agent fell into a hole",
    "an empty frozen lake",
    "a random unrelated scene",
)
_FROZEN_LAKE_VLE_BASELINE = "a blank frozen lake grid"

_OPEN_APPS_VLE_NEGATIVES = (
    "Goal: Add 'Call Mom' to my todo list.",
    "Goal: Go to the Calendar app and add my meeting with Dennis on April 1st of 2026. The title should be 'Dennis-Bob'",
    "Goal: Go to the Calendar app and add my meeting with Einstein on April 1st of 2027. The title should be 'Einstein-Bob'. Set the description as 'paper reading', omit the URL and set the location to New York City. Make sure to add Einstein as an invitee.",
    "Goal: Make a calendar entry for Christmas shopping on the 14th (2025). The title should read 'Shopping for Christmas gifts'.",
    "Goal: Mark 'Water plants' as done in my todo list.",
    "Goal: Remove the WACV 2026 Abstract Deadline event from my calendar.",
    "Goal: Save Paris, France to my favorite places",
    "Goal: Send a message to Bob 'Let's meet'",
)
_OPEN_APPS_VLE_BASELINE = "a generic web browser home page"


def _vle_method_params(
    negatives: tuple[str, ...], baseline: str
) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
    """The env-specific vle-* fields: baseline prompt for vle-gbr, negative
    goals for the vlm_sor methods (vle-thresh / vle-softmax)."""
    return (
        ("vle-gbr", (("vle_baseline_prompt", baseline),)),
        ("vle-thresh", (("vle_negative_goals", negatives),)),
        ("vle-softmax", (("vle_negative_goals", negatives),)),
    )


ENVIRONMENTS: tuple[EnvironmentSpec, ...] = (
    EnvironmentSpec(
        env="alfworld",
        display="ALFWorld",
        dataset_id="all-types-40ms",
        dataset_path="shared/data/datasets/alfworld/200pt_all-types_40ms_ds32_20260414_035138.pkl",
        supports_vision=True,
        gt_actors=("scripted",),
        ranking_gt_actors=("scripted",),
        text_actors=_LLM_ACTORS_TEXT,
        vision_actors=(*_LLM_ACTORS_VISION, "clip", "siglip", "self"),
        text_points=100,
        vision_points=200,
        gt_points=200,
        context_name="alfworld_40ms_react_tags",
        vision_context_name="alfworld_40ms_vision",
        experiment_info_aliases=("alfworld:eval_out_of_distribution",),
        method_params=_vle_method_params(_ALFWORLD_VLE_NEGATIVES, _ALFWORLD_VLE_BASELINE),
    ),
    EnvironmentSpec(
        env="frozen_lake",
        display="FrozenLake",
        dataset_id="8x8",
        dataset_path="shared/data/datasets/frozen_lake/8x8_scripted_20260420_152752.pkl",
        supports_vision=True,
        gt_actors=("scripted",),
        ranking_gt_actors=("scripted",),
        text_actors=_LLM_ACTORS_TEXT,
        vision_actors=(*_LLM_ACTORS_VISION, "clip", "siglip", "self"),
        text_points=100,
        vision_points=100,
        vision_nonllm_points=239,
        gt_points=200,
        gt_policy_name="frozen_lake_random_8x8",
        context_name="frozen_lake_8x8_react_tags",
        vision_context_name="frozen_lake_8x8_vision",
        experiment_info_aliases=("frozen_lake:eval", "frozen_lake/8x8"),
        method_params=_vle_method_params(_FROZEN_LAKE_VLE_NEGATIVES, _FROZEN_LAKE_VLE_BASELINE),
    ),
    EnvironmentSpec(
        env="open_apps",
        display="OpenApps",
        dataset_id="eps025-ms45-recovery-som-filtered-zeros-ranking",
        dataset_path=(
            "shared/data/datasets/open_apps/"
            "scripted_40traj_eps025_ms45_recovery_som_100pt_filtered_zeros_ranking_20260504_193708.pkl"
        ),
        supports_vision=True,
        gt_actors=("scripted",),
        ranking_gt_actors=("scripted",),
        text_actors=("g4-26-or", "g4-31-or", "q35-9-or", "q35-27-or", "q35-35-or", "q35-122-or"),
        vision_actors=(
            "g4-26-or", "g4-31-or", "q35-9-or", "q35-27-or", "q35-35-or", "q35-122-or",
            "clip", "siglip", "self",
        ),
        text_points=94,
        vision_points=94,
        gt_points=94,
        context_name="open_apps_recovery_som_text",
        # The surviving vle configs name a context "open_apps_vision" that now
        # lives in legacy/configs/environments/_old/; the real vision runs use the
        # set-of-marks recovery context (same family as the text context).
        vision_context_name="open_apps_vision_recovery_som",
        experiment_info_aliases=("openapps:eval", "add_meeting_with_dennis"),
        method_params=_vle_method_params(_OPEN_APPS_VLE_NEGATIVES, _OPEN_APPS_VLE_BASELINE),
    ),
    EnvironmentSpec(
        env="terminal_bench",
        display="TerminalBench",
        dataset_id="tblite-easy-40ms",
        dataset_path="shared/data/datasets/terminal_bench/200pt_lite-easy_40ms_ds32_20260415_014717.pkl",
        supports_vision=False,
        supports_sv=False,
        gt_actors=("codex-55", "ds32", "opus-47"),
        ranking_gt_actors=("codex-55", "ds32"),
        text_actors=_LLM_ACTORS_TEXT,
        text_points=100,
        gt_points=100,
        gt_aggregations=("max", "mean"),
        context_name="tblite_easy_40ms_react_tags",
        experiment_info_aliases=("terminal_bench:lite-easy", "tblite"),
    ),
)


# Method-name base -> spec. Seeds display/family/signals/modalities from the
# paper-figure registry + the discovery method matrix. Per-method
# extra EvalMethodConfig fields are refined against real configs when config
# generation is wired up.
_DIRECT_MODS = (TEXT, VISION)

METHODS: tuple[MethodSpec, ...] = (
    # ---- Direct family (llm_direct + gvl) ----
    MethodSpec("direct", "llm_direct", "Direct", "direct-single", (QV, SV), _DIRECT_MODS),
    MethodSpec(
        "direct-verif16", "llm_direct", "Direct", "direct-16", (QV, SV), _DIRECT_MODS,
        is_ablation=True,
        extra_fields=(("num_verifications", 16),),
    ),
    MethodSpec(
        "direct-batch4-packed", "llm_direct", "Direct", "direct-batched", (QV, SV), _DIRECT_MODS,
        is_ablation=True,
        extra_fields=(("prompt_batch_size", 4), ("prompt_batch_mode", "packed")),
    ),
    MethodSpec(
        "direct-batch8-seq", "llm_direct", "Direct", "direct-sequential", (QV, SV), _DIRECT_MODS,
        is_ablation=True,
        extra_fields=(
            ("prompt_batch_size", 8),
            ("prompt_batch_mode", "sequential"),
            ("disable_prompt_truncation", True),
        ),
    ),
    MethodSpec("gvl", "llm_direct_gvl", "Direct", "gvl", (QV, SV), _DIRECT_MODS),
    # ---- Code family (codegen/eureka use backend_codegen). Sample counts and
    #      eureka search params pre-filled from the surviving canonical config. ----
    MethodSpec(
        "eureka", "llm_eureka", "Code", "eureka", (QV, SV), (TEXT,),
        num_samples=8, extra_fields=(("search_iterations", 16), ("judge_num_points", 8)),
    ),
    MethodSpec("codegen", "llm_codegen", "Code", "codegen", (QV, SV), (TEXT,), num_samples=16),
    # ---- Intrinsic ----
    MethodSpec("verifier", "llm_verifier", "Intrinsic", "verifier", (QV,), (TEXT,)),
    MethodSpec("delta-belief", "delta_belief_ranking", "Intrinsic", r"$\Delta$belief", (QV,), (TEXT,), is_ranking=True),
    # ---- Ranking ----
    MethodSpec("ranking", "llm_ranking", "Ranking", "ranking", (QV,), (TEXT, VISION), is_ranking=True),
    # ---- Self-Distillation ----
    MethodSpec("sdpo", "sdpo_ranking", "Self-Distillation", "sdpo", (QV,), (TEXT,), is_ranking=True),
    MethodSpec("sdpo-gt", "sdpo_ranking", "Self-Distillation", "sdpo-gt", (QV,), (TEXT,), is_ranking=True),
    # ---- Pre-trained (pinned to the "self" actor; backend is method-supplied).
    #      goal_source / similarity are env-agnostic (trajectory_end uses each
    #      trajectory's own last frame); pre-filled from the surviving configs. ----
    MethodSpec("vip", "vip", "Pre-trained", "vip", (QV, SV), (VISION,), prompt_preset=None, pinned_actor="self", backend_override="vip_resnet50",
               extra_fields=(("vip_goal_source", "trajectory_end"),)),
    MethodSpec("liv-cos", "liv_img", "Pre-trained", "liv-cos", (QV, SV), (VISION,), prompt_preset=None, pinned_actor="self", backend_override="liv_clip_rn50",
               extra_fields=(("liv_goal_source", "trajectory_end"), ("liv_similarity", "cosine"))),
    MethodSpec("liv-l2", "liv_img", "Pre-trained", "liv-l2", (QV, SV), (VISION,), prompt_preset=None, pinned_actor="self", backend_override="liv_clip_rn50",
               extra_fields=(("liv_goal_source", "trajectory_end"), ("liv_similarity", "l2"))),
    MethodSpec("liv-txt", "liv_txt", "Pre-trained", "liv-txt", (QV, SV), (VISION,), prompt_preset=None, pinned_actor="self", backend_override="liv_clip_rn50",
               extra_fields=(("liv_goal_per_point", True),)),
    # ---- Embedding (vle-*, pinned to a clip/siglip backbone; QV + SV).
    #      These image-based methods score *states*, not actions, so the Q-value
    #      is proxied as the state value of the next state, V(s'). QV is the
    #      primary signal for the paper's main results. The method picks the
    #      frame to score from signal_type alone (vle/method.py: next_state for
    #      q_value, state otherwise), so the QV/SV split needs no extra config
    #      field — the generator's per-signal signal_type is sufficient.
    #      goal_per_point + alpha/temperature/beta are method-level; the
    #      negative goals and baseline prompt are environment-specific and live
    #      in EnvironmentSpec.method_params (see _*_VLE_NEGATIVES below). ----
    MethodSpec("vle-cosine", "vlm_rm", "Embedding", "vlm-rm-cos", (QV, SV), (VISION,), prompt_preset=None, requires_embedding_backbone=True,
               extra_fields=(("vle_goal_per_point", True),)),
    MethodSpec("vle-gbr", "vlm_rm", "Embedding", "vlm-rm", (QV, SV), (VISION,), prompt_preset=None, requires_embedding_backbone=True,
               extra_fields=(("vle_goal_per_point", True), ("vle_alpha", 0.5))),
    MethodSpec("vle-thresh", "vlm_sor", "Embedding", "vlm-sor", (QV, SV), (VISION,), prompt_preset=None, requires_embedding_backbone=True,
               extra_fields=(("vle_goal_per_point", True), ("vle_temperature", 0.07), ("vle_beta", 0.5))),
    MethodSpec("vle-softmax", "vlm_sor", "Embedding", "vlm-sor-softmax", (QV, SV), (VISION,), prompt_preset=None, requires_embedding_backbone=True,
               extra_fields=(("vle_goal_per_point", True), ("vle_temperature", 0.07))),
)


REGISTRY = Registry(MODELS, ENVIRONMENTS, METHODS)
