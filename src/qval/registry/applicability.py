"""Applicability rules: which (method, model, env, modality, signal) tuples are valid.

This single-sources the cross-cutting constraints that were previously implicit
across the hand-written experiment YAMLs and the discovery layer's
``EXPECTED_*`` tables. The rules are pinned against the actual on-disk prediction matrix in
``tests/test_applicability.py``.

Three rules cover everything observed on disk:

* **vision is QV-only for plain-LLM methods** — every ``EVAL_*_<llm>_vision``
  dir contains only ``*_qv`` files. State-value signals in vision come solely
  from the pinned pre-trained (vip / liv-*) and embedding (vle-*) methods.
* **the ``supports_sv`` gate** — TerminalBench produces no ``*_sv`` artifacts.
* **the actor-class partition** — plain-LLM methods run on LLM actors; ``self``
  runs only vip / liv-*; the clip/siglip backbones run only vle-*.
"""

from __future__ import annotations

from qval.registry.active import active_registry
from qval.registry.atoms import (
    EnvironmentSpec,
    MethodSpec,
    Modality,
    ModelSpec,
    Registry,
    Signal,
)


def _is_plain_llm_method(method: MethodSpec) -> bool:
    """A method that runs on the LLM actor axis (direct / gvl / codegen /
    eureka / verifier / ranking / sdpo / …), as opposed to the pinned
    pre-trained (vip / liv) and embedding (vle) methods."""
    return method.pinned_actor is None and not method.requires_embedding_backbone


def _is_plain_llm_actor(model: ModelSpec) -> bool:
    return not model.is_non_llm and not model.is_gt_only


def signals_for(
    method: MethodSpec, env: EnvironmentSpec, modality: Modality
) -> tuple[Signal, ...]:
    """The signals ``method`` produces in this (env, modality), after the
    ``supports_sv`` gate and the vision-QV-only rule."""
    out: list[Signal] = []
    for sig in method.signals:
        if sig is Signal.SV:
            if not env.supports_sv:
                continue
            if modality is Modality.VISION and _is_plain_llm_method(method):
                continue
        out.append(sig)
    return tuple(out)


def is_applicable(
    method: MethodSpec,
    model: ModelSpec,
    env: EnvironmentSpec,
    modality: Modality,
) -> bool:
    """Whether ``method`` run by ``model`` is a valid experiment in this
    (env, modality) for at least one signal."""
    if modality not in method.modalities:
        return False
    if modality is Modality.VISION and not (env.supports_vision and model.supports_vision):
        return False

    # actor-class partition
    if method.requires_embedding_backbone:
        if not model.is_embedding_backbone:
            return False
    elif method.pinned_actor is not None:
        if model.actor != method.pinned_actor:
            return False
    elif not _is_plain_llm_actor(model):
        return False

    return bool(signals_for(method, env, modality))


def build_method_matrix(
    env: EnvironmentSpec,
    modality: Modality,
    *,
    include_ablations: bool = False,
    registry: Registry | None = None,
) -> dict[Signal, tuple[str, ...]]:
    """``{Signal: (method_base, …)}`` of methods applicable in this
    (env, modality), unioned over the actor classes the env lists for that
    modality. Method bases follow catalog order; signals are QV then SV.

    This is the shape the discovery layer's expected-method matrix is pinned to.
    """
    registry = registry or active_registry()
    actors = [m for m in (registry.model_by_actor(a) for a in env.actors_for(modality)) if m]
    buckets: dict[Signal, list[str]] = {}
    for method in registry.methods:
        if method.is_ablation and not include_ablations:
            continue
        if not any(is_applicable(method, model, env, modality) for model in actors):
            continue
        for sig in signals_for(method, env, modality):
            bucket = buckets.setdefault(sig, [])
            if method.base not in bucket:
                bucket.append(method.base)
    return {sig: tuple(buckets[sig]) for sig in (Signal.QV, Signal.SV) if sig in buckets}


# Discovery vocabulary: ranking methods that produce a QV signal
# are tracked under a dedicated key (they live in ranking-format prediction
# files, not the pointwise qv/sv buckets).
RANKING_QV_KEY = f"ranking_{Signal.QV.value}"

# A reference environment with full vision + state-value support. It makes the
# per-actor expected-method matrix purely actor-class driven; environment
# gates (``supports_sv``, ranking ground-truth actors) are applied by callers.
_REF_ENV = EnvironmentSpec(
    env="_ref",
    display="",
    dataset_id="",
    dataset_path="",
    supports_vision=True,
    supports_sv=True,
    gt_actors=(),
    text_actors=(),
    vision_actors=(),
)


def expected_methods_for_actor(
    actor: str,
    modality: Modality | str,
    *,
    registry: Registry | None = None,
) -> dict[str, tuple[str, ...]]:
    """Env-agnostic ``{"qv", "sv", "ranking_qv"}`` method-base matrix for one
    evaluation actor, in the shape the ``evaluate.py`` discovery layer consumes.

    Unlike :func:`build_method_matrix` (which unions over an environment's
    actors, excludes ablations, and lumps ranking into QV), this is **per
    actor** and mirrors the historical ``default_expected_methods``: ablation
    methods are included (they have on-disk prediction files) and ranking
    methods producing a QV signal are split into the ``ranking_qv`` bucket.
    Codegen is **not** sample-expanded here — that file-level detail belongs to
    the discovery layer. Returns ``{"qv": (), "sv": ()}`` for unknown or
    ground-truth-only actors. Method bases follow catalog order.
    """
    registry = registry or active_registry()
    model = registry.model_by_actor(actor)
    if model is None or model.is_gt_only:
        return {"qv": (), "sv": ()}
    mod = Modality(modality)
    qv: list[str] = []
    sv: list[str] = []
    ranking_qv: list[str] = []
    for method in registry.methods:
        if not is_applicable(method, model, _REF_ENV, mod):
            continue
        for sig in signals_for(method, _REF_ENV, mod):
            if sig is Signal.QV and method.is_ranking:
                ranking_qv.append(method.base)
            elif sig is Signal.QV:
                qv.append(method.base)
            else:
                sv.append(method.base)
    result: dict[str, tuple[str, ...]] = {"qv": tuple(qv), "sv": tuple(sv)}
    if ranking_qv:
        result[RANKING_QV_KEY] = tuple(ranking_qv)
    return result
