"""Canonical naming: derive every path/name from registry atoms.

This module is the single implementation of the conventions documented in
``docs/naming_conventions.md``. It absorbed the former
``scripts/evaluation_naming.py`` and the prediction-directory parsing that is
also implemented in the discovery layer of ``scripts/pipeline/evaluate.py``.

Two naming schemes live here:

* **Evaluation-output dirs** — ``<dataset_id>_<gt_actor>_<eval_actor>_<modality>``
  under ``data/evaluations/<env>/``, with ``summary_<ts>.json`` files. The
  format/parse functions here are byte-identical to the historical
  implementation; the viz layer depends on that.
* **Prediction dirs** — ``GT_<N>pt_<dataset_id>_<actor>`` and
  ``EVAL_<N>pt_<dataset_id>_<actor>_<modality>`` under
  ``data/predictions/<env>/``, with ``<method>_<ts>.json`` files.

The actor / GT-actor / modality vocabularies are *derived* from the active
catalog's registry rather than hand-maintained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from qval.registry.active import active_registry
from qval.registry.atoms import Modality, Signal

# ---------------------------------------------------------------------------
# Derived vocabularies. ``MODALITIES`` is catalog-independent. The actor
# vocabularies (``EVAL_ACTORS`` / ``GT_ACTORS`` / ``ALL_ACTORS``) follow the
# active catalog and are resolved lazily through module ``__getattr__`` (PEP
# 562), so a ``set_active(...)`` before first access re-points the parsers.
# Functions below fetch the vocabulary into locals for the same reason (a bare
# module-global reference would not trigger ``__getattr__``).
# ---------------------------------------------------------------------------
MODALITIES: tuple[str, ...] = tuple(m.value for m in Modality)


def __getattr__(name: str) -> tuple[str, ...]:
    if name in ("EVAL_ACTORS", "GT_ACTORS", "ALL_ACTORS"):
        registry = active_registry()
        eval_actors = registry.eval_actors
        gt_actors = registry.gt_actors
        return {
            "EVAL_ACTORS": eval_actors,
            "GT_ACTORS": gt_actors,
            "ALL_ACTORS": eval_actors + gt_actors,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

SUMMARY_RE = re.compile(r"^summary_(\d{8}_\d{6})\.json$")


# ---------------------------------------------------------------------------
# Evaluation-output dirs (data/evaluations/<env>/...).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationOutputDirInfo:
    dataset_id: str
    gt_actor: str
    eval_actor: str
    modality: str


@dataclass(frozen=True)
class CanonicalSummaryPath:
    env: str
    info: EvaluationOutputDirInfo
    path: Path
    timestamp: str


def format_evaluation_output_dir_name(
    dataset_id: str,
    *,
    gt_actor: str,
    eval_actor: str,
    modality: str,
) -> str:
    """Return ``<dataset>_<gt_actor>_<eval_actor>_<modality>``."""
    return f"{dataset_id}_{gt_actor}_{eval_actor}_{modality}"


def parse_evaluation_output_dir_name(
    path_or_name: str | Path,
) -> EvaluationOutputDirInfo | None:
    """Parse canonical evaluation output directory names.

    Dataset IDs may contain underscores, so parsing works from the right using
    the finite actor/modality vocabularies.
    """
    registry = active_registry()
    eval_actors = registry.eval_actors
    gt_actors = registry.gt_actors
    name = Path(path_or_name).name
    modality = None
    rest = name
    for candidate in MODALITIES:
        suffix = f"_{candidate}"
        if rest.endswith(suffix):
            modality = candidate
            rest = rest[: -len(suffix)]
            break
    if modality is None:
        return None

    eval_split = _split_suffix_actor(rest, eval_actors)
    if eval_split is None:
        return None
    rest, eval_actor = eval_split

    gt_split = _split_suffix_actor(rest, gt_actors)
    if gt_split is None:
        return None
    dataset_id, gt_actor = gt_split
    if not dataset_id:
        return None

    return EvaluationOutputDirInfo(
        dataset_id=dataset_id,
        gt_actor=gt_actor,
        eval_actor=eval_actor,
        modality=modality,
    )


def parse_summary_filename(path_or_name: str | Path) -> str | None:
    match = SUMMARY_RE.fullmatch(Path(path_or_name).name)
    return match.group(1) if match else None


def latest_canonical_summary_paths(root: Path) -> tuple[CanonicalSummaryPath, ...]:
    """Return the latest summary per canonical evaluation output directory."""
    selected: dict[tuple[str, str, str, str, str], CanonicalSummaryPath] = {}
    for path in Path(root).rglob("summary_*.json"):
        timestamp = parse_summary_filename(path)
        if timestamp is None:
            continue
        info = parse_evaluation_output_dir_name(path.parent.name)
        if info is None:
            continue
        env = path.parent.parent.name
        key = (env, info.dataset_id, info.gt_actor, info.eval_actor, info.modality)
        summary = CanonicalSummaryPath(
            env=env,
            info=info,
            path=path,
            timestamp=timestamp,
        )
        current = selected.get(key)
        if current is None or summary.timestamp > current.timestamp:
            selected[key] = summary
    return tuple(sorted(selected.values(), key=lambda item: str(item.path)))


def _split_suffix_actor(
    value: str,
    actors: tuple[str, ...],
) -> tuple[str, str] | None:
    for actor in sorted(actors, key=len, reverse=True):
        suffix = f"_{actor}"
        if value.endswith(suffix):
            prefix = value[: -len(suffix)]
            if prefix:
                return prefix, actor
    return None


# ---------------------------------------------------------------------------
# Prediction dirs (data/predictions/<env>/...).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionDirInfo:
    kind: str  # "GT" or "EVAL"
    num_points: int | None
    dataset_id: str
    actor: str
    modality: str | None  # None for GT dirs


def points_token(num_points: int) -> str:
    """``100`` -> ``"100pt"``."""
    return f"{num_points}pt"


def gt_dir_name(*, num_points: int, dataset_id: str, gt_actor: str) -> str:
    """``GT_<N>pt_<dataset_id>_<gt_actor>``."""
    return f"GT_{points_token(num_points)}_{dataset_id}_{gt_actor}"


def eval_dir_name(
    *, num_points: int, dataset_id: str, actor: str, modality: str | Modality
) -> str:
    """``EVAL_<N>pt_<dataset_id>_<actor>_<modality>``."""
    modality = modality.value if isinstance(modality, Modality) else modality
    return f"EVAL_{points_token(num_points)}_{dataset_id}_{actor}_{modality}"


def parse_prediction_dir_name(path_or_name: str | Path) -> PredictionDirInfo | None:
    """Parse ``GT_*`` / ``EVAL_*`` prediction directory names.

    Mirrors the discovery logic in the ``scripts/pipeline/evaluate.py``
    discovery layer:
    EVAL actors are matched against the eval vocabulary, GT actors against the
    full vocabulary (eval + gt), and the leading ``<N>pt`` token is split off
    the dataset id.
    """
    registry = active_registry()
    eval_actors = registry.eval_actors
    all_actors = eval_actors + registry.gt_actors
    name = Path(path_or_name).name
    if name.startswith("EVAL_"):
        rest = name[len("EVAL_") :]
        modality: str | None = None
        for candidate in MODALITIES:
            suffix = f"_{candidate}"
            if rest.endswith(suffix):
                modality = candidate
                rest = rest[: -len(suffix)]
                break
        if modality is None:
            return None
        split = _split_suffix_actor(rest, eval_actors)
        if split is None:
            return None
        dataset_key, actor = split
        num_points, dataset_id = _split_points_prefix(dataset_key)
        if not dataset_id:
            return None
        return PredictionDirInfo("EVAL", num_points, dataset_id, actor, modality)
    if name.startswith("GT_"):
        rest = name[len("GT_") :]
        split = _split_suffix_actor(rest, all_actors)
        if split is None:
            return None
        dataset_key, actor = split
        num_points, dataset_id = _split_points_prefix(dataset_key)
        if not dataset_id:
            return None
        return PredictionDirInfo("GT", num_points, dataset_id, actor, None)
    return None


def _split_points_prefix(dataset_key: str) -> tuple[int | None, str]:
    """``"100pt_all-types-40ms"`` -> ``(100, "all-types-40ms")``.

    Mirrors the discovery layer's ``_normalize_dataset_id`` (handles ``_`` and ``-``
    separators after the ``pt`` token) and additionally recovers the integer.
    """
    for separator in ("_", "-"):
        marker = f"pt{separator}"
        if marker in dataset_key:
            prefix, rest = dataset_key.split(separator, 1)
            if prefix.endswith("pt") and prefix[:-2].isdigit():
                return int(prefix[:-2]), rest
    if dataset_key.endswith("pt") and dataset_key[:-2].isdigit():
        return int(dataset_key[:-2]), ""
    return None, dataset_key


# ---------------------------------------------------------------------------
# Prediction / GT / log file names.
# ---------------------------------------------------------------------------
def _signal_token(signal: Signal | str) -> str:
    return signal.value if isinstance(signal, Signal) else signal


def eval_method_name(method_base: str, signal: Signal | str) -> str:
    """``("direct", QV)`` -> ``"direct_qv"`` (the JSON ``method_name``)."""
    return f"{method_base}_{_signal_token(signal)}"


def prediction_filename(method_base: str, signal: Signal | str, timestamp: str) -> str:
    """``"direct_qv_<ts>.json"`` (eval prediction)."""
    return f"{eval_method_name(method_base, signal)}_{timestamp}.json"


def codegen_sidecar_filename(
    method_base: str, signal: Signal | str, timestamp: str
) -> str:
    """``"codegen-s0_qv_<ts>.py"`` (generated-code sidecar)."""
    return f"{eval_method_name(method_base, signal)}_{timestamp}.py"


def gt_method_filename(gt_method: str, timestamp: str) -> str:
    """``"qv_mc_max_<ts>.json"`` (GT prediction)."""
    return f"{gt_method}_{timestamp}.json"


def log_filename(prefix: str, timestamp: str) -> str:
    """``("predict.logs", ts)`` -> ``"predict.logs_<ts>.jsonl"``."""
    return f"{prefix}_{timestamp}.jsonl"


def predictions_dir_path(predictions_root: str | Path, env: str, dir_name: str) -> Path:
    """``data/predictions/<env>/<dir_name>``."""
    return Path(predictions_root) / env / dir_name
