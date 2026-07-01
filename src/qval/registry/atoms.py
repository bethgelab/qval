"""Typed atoms of an experiment: models, environments, and methods.

These dataclasses are the vocabulary the rest of the registry is built from.
They are deliberately pure data + tiny derivations (no I/O, no config loading)
so they can be imported cheaply from ``src/qval``, ``scripts/pipeline``
and ``scripts/paper_figures`` alike.

The concrete instances live in :mod:`qval.registry.catalog`; naming
derivations live in :mod:`qval.registry.naming`; applicability rules in
:mod:`qval.registry.applicability`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Modality(str, Enum):
    """Input modality an experiment is run under."""

    TEXT = "text"
    VISION = "vision"


class Signal(str, Enum):
    """Dense-signal token as it appears in file/dir names (``_qv`` / ``_sv``)."""

    QV = "qv"
    SV = "sv"

    @property
    def config_signal_type(self) -> str:
        """The ``EvalMethodConfig.signal_type`` / ``EstimationConfig.assumption``
        string this token maps to."""
        return {Signal.QV: "q_value", Signal.SV: "state_value"}[self]


@dataclass(frozen=True)
class ModelSpec:
    """An evaluation actor or a ground-truth-producing actor.

    ``actor`` is the canonical token used in on-disk directory names
    (``q35-27-or``, ``clip``, ``scripted``); it is the primary key. It is
    distinct from ``backend_base`` (the ``backends.yaml`` entry name, e.g.
    ``qwen35_27_thinking_or``) and from ``display`` (the figure label, e.g.
    ``Qwen3.5 27B``). Unifying that three-way identity is a core reason this
    registry exists.
    """

    actor: str
    display: str
    # backends.yaml entry names — references only; the definitions stay
    # hand-maintained in ``shared/configs/backends.yaml``. ``None`` where the actor
    # has no such backend (gt-only / pinned non-LLM actors, or a usage the
    # actor doesn't have a distinct backend for).
    backend_base: str | None = None
    backend_codegen: str | None = None
    backend_vision: str | None = None
    # capability flags
    supports_thinking: bool = True
    supports_vision: bool = False
    supports_multi_image: bool = False
    is_embedding_backbone: bool = False  # clip, siglip (used only by vle-* methods)
    is_pretrained: bool = False  # the "self" pseudo-actor (vip, liv-*)
    is_gt_only: bool = False  # scripted / codex-55 / ds32 / opus-47 / random
    include_in_results: bool = True  # False excludes a model from results figures/tables

    @property
    def is_non_llm(self) -> bool:
        """Embedding backbones and pinned pre-trained actors don't share the
        LLM model axis (they have their own ``num_points`` slice in some envs).
        """
        return self.is_embedding_backbone or self.is_pretrained

    def backend_for(self, *, codegen: bool = False, vision: bool = False) -> str | None:
        """Resolve the concrete backend name for a usage, falling back to base."""
        if codegen and self.backend_codegen is not None:
            return self.backend_codegen
        if vision and self.backend_vision is not None:
            return self.backend_vision
        return self.backend_base


@dataclass(frozen=True)
class EnvironmentSpec:
    """One benchmark environment and its canonical dataset slice.

    ``env`` is the on-disk directory token (``alfworld``); ``dataset_id`` is the
    canonical, hyphenated dataset identifier that appears in dir names
    (``all-types-40ms``).

    Point counts are intentionally explicit fields rather than a mapping
    because they vary along three axes that don't reduce to ``{modality: int}``:
    text, vision (LLM), and vision (non-LLM embedding/pretrained) can all
    differ, and the GT slice is independent again. See :meth:`points_for`.
    """

    env: str
    display: str
    dataset_id: str
    dataset_path: str
    supports_vision: bool
    gt_actors: tuple[str, ...]
    text_actors: tuple[str, ...]
    vision_actors: tuple[str, ...] = ()
    ranking_gt_actors: tuple[str, ...] = ()
    supports_sv: bool = True
    # point counts (the "<N>pt" token)
    text_points: int | None = None
    vision_points: int | None = None
    vision_nonllm_points: int | None = None  # falls back to vision_points when None
    gt_points: int | None = None
    # context selection (refined when configs are generated)
    context_name: str | None = None
    vision_context_name: str | None = None
    contexts_dir: str = "shared/configs/environments/"
    # experiment_info.env_name values that map to this env in the viz layer
    experiment_info_aliases: tuple[str, ...] = ()
    # Generation defaults emitted into generated prediction configs. Global
    # defaults here; override per environment in the catalog when an env needs
    # a different value. ``gt_aggregations`` is the set of MC aggregations the
    # GT estimations are produced under (``{sig}_mc_{agg}``).
    batch_size: int = 32
    discount_factor: float = 0.95
    max_history_turns: int = 40
    max_aborted_points: float = 0.3
    gt_aggregations: tuple[str, ...] = ("max",)
    # GT rollout source for the generated GT configs. Scripted-policy envs use
    # ``gt_policy_name``; LLM ground-truth actors (e.g. TerminalBench's
    # codex-55 / ds32 / opus-47) instead roll out with the gt actor's backend
    # (``ModelSpec.backend_base``). Both are fill-in slots — leave ``None`` and
    # the generator emits the estimation without a source for you to complete.
    gt_policy_name: str | None = None
    gt_num_rollouts: int = 1
    # Per-environment, per-method extra config fields, for hyperparameters that
    # are genuinely environment-specific (e.g. the vle-* methods' negative
    # goals / baseline prompt). Keyed by method base; merged on top of
    # ``MethodSpec.extra_fields`` when the config is generated. Values are
    # hashable tuples-of-pairs; list-valued fields use a tuple of items.
    method_params: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = ()

    def params_for(self, method_base: str) -> tuple[tuple[str, object], ...]:
        """The environment-specific extra fields for ``method_base`` (or ())."""
        for base, fields in self.method_params:
            if base == method_base:
                return fields
        return ()

    def points_for(self, modality: Modality, *, non_llm: bool = False) -> int:
        """Number of evaluation points for a (modality, actor-class) slice."""
        if modality == Modality.TEXT:
            if self.text_points is None:
                raise ValueError(f"{self.env} has no text point count")
            return self.text_points
        if non_llm and self.vision_nonllm_points is not None:
            return self.vision_nonllm_points
        if self.vision_points is None:
            raise ValueError(f"{self.env} has no vision point count")
        return self.vision_points

    def actors_for(self, modality: Modality) -> tuple[str, ...]:
        return self.text_actors if modality == Modality.TEXT else self.vision_actors


@dataclass(frozen=True)
class MethodSpec:
    """A dense-signal method evaluated against the ground truth.

    ``base`` is the method-name base (sans signal token), e.g. ``direct`` or
    ``vle-gbr``; the on-disk ``method_name`` is ``f"{base}_{signal}"``.
    """

    base: str
    type: str  # EvalMethodConfig.type
    family: str
    display: str
    signals: tuple[Signal, ...]
    modalities: tuple[Modality, ...]
    prompt_preset: str | None = "v2"
    pinned_actor: str | None = None  # "self" for pre-trained methods
    # Backend that is determined by the *method*, not the actor (vip / liv-*):
    # these run their own pre-trained backbone regardless of the `self` actor.
    # ``None`` means the backend comes from the actor (see resolve_backend).
    backend_override: str | None = None
    requires_embedding_backbone: bool = False  # vle-* -> actor in {clip, siglip}
    is_ranking: bool = False
    # Ablation methods are declarable but excluded from generation by default
    # (the generator emits the "main" pipeline unless --include-ablations).
    is_ablation: bool = False
    num_samples: int | None = None
    # extra EvalMethodConfig fields rendered verbatim into generated YAML
    extra_fields: tuple[tuple[str, object], ...] = ()

    @property
    def is_pinned(self) -> bool:
        return self.pinned_actor is not None or self.requires_embedding_backbone


@dataclass(frozen=True)
class Registry:
    """Indexed view over the catalogs with the common lookups.

    Lives here (with the atoms it indexes) rather than in any one catalog module
    so that a catalog loaded from an arbitrary path — inside or outside the
    package — produces a ``Registry`` of *this* class. That keeps ``isinstance``
    checks and re-exports stable across catalogs.
    """

    models: tuple[ModelSpec, ...]
    environments: tuple[EnvironmentSpec, ...]
    methods: tuple[MethodSpec, ...]

    def model_by_actor(self, actor: str) -> ModelSpec | None:
        return next((m for m in self.models if m.actor == actor), None)

    def model_by_display(self, display: str) -> ModelSpec | None:
        return next((m for m in self.models if m.display == display), None)

    def model_by_backend(self, backend_name: str) -> ModelSpec | None:
        for m in self.models:
            if backend_name in (m.backend_base, m.backend_codegen, m.backend_vision):
                return m
        return None

    def env_by_name(self, env: str) -> EnvironmentSpec | None:
        return next((e for e in self.environments if e.env == env), None)

    def env_by_display(self, display: str) -> EnvironmentSpec | None:
        return next((e for e in self.environments if e.display == display), None)

    def env_by_alias(self, alias: str) -> EnvironmentSpec | None:
        return next(
            (e for e in self.environments if alias in e.experiment_info_aliases),
            None,
        )

    def method_by_base(self, base: str) -> MethodSpec | None:
        return next((m for m in self.methods if m.base == base), None)

    @property
    def eval_actors(self) -> tuple[str, ...]:
        return tuple(m.actor for m in self.models if not m.is_gt_only)

    @property
    def gt_actors(self) -> tuple[str, ...]:
        return tuple(m.actor for m in self.models if m.is_gt_only)
