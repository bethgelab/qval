"""Prediction storage: save/load per-method prediction JSON files."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qval.config import PredictionSourceConfig
from qval.experiment_logging import sanitize_for_json
from qval.path_resolution import extract_timestamp, timestamped_path_glob


@dataclass
class Prediction:
    """A single method's predictions for a set of evaluation points.

    Attributes:
        method_name: Human-readable method name.
        method_type: ``"gt"`` for ground truth methods, ``"eval"`` for
            evaluated methods.
        values: Predicted signal values (one per evaluation point).
        config: Method configuration (estimation config dict for GT, etc.).
        metadata: Additional metadata (timestamps, model info, etc.).
    """

    method_name: str
    method_type: str  # "gt" or "eval"
    values: list[float]
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

def save_prediction(pred: Prediction, path: str | Path) -> None:
    """Save a Prediction to a JSON file.

    NaN and Inf values are replaced with ``null`` for JSON compatibility.
    Creates parent directories if they don't exist.  Writes are atomic
    (temp file + ``os.replace``) so a crash never leaves a half-written
    artifact.

    If ``metadata`` contains ``"generated_code"``, the code is written to a
    sibling ``.py`` file (same stem) and removed from the JSON to avoid
    escaping issues with embedded multi-line code.
    """
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = sanitize_for_json(asdict(pred))
    metadata = data.setdefault("metadata", {})

    # Extract generated_code to a sibling .py file.
    generated_code = metadata.pop("generated_code", None)
    code_path = path.with_suffix(".py")
    tmp_py: str | None = None
    if generated_code is not None:
        metadata["generated_code_present"] = True
        metadata["generated_code_sha256"] = hashlib.sha256(
            generated_code.encode("utf-8")
        ).hexdigest()
        fd, tmp_py = tempfile.mkstemp(dir=path.parent, suffix=".py.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(generated_code)
        except BaseException:
            try:
                os.unlink(tmp_py)
            except OSError:
                pass
            raise
    else:
        metadata["generated_code_present"] = False

    fd, tmp_json = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_json, path)
        if tmp_py is not None:
            os.replace(tmp_py, code_path)
            tmp_py = None
        elif code_path.exists():
            try:
                code_path.unlink()
            except FileNotFoundError:
                pass
    except BaseException:
        try:
            os.unlink(tmp_json)
        except OSError:
            pass
        if tmp_py is not None:
            try:
                os.unlink(tmp_py)
            except OSError:
                pass
        raise


def _restore_nan(value: Any) -> float:
    """Convert None back to NaN for loaded values."""
    if value is None:
        return float("nan")
    return float(value)


def load_prediction(path: str | Path) -> Prediction:
    """Load a Prediction from a JSON file.

    ``null`` values in the ``values`` list are converted back to ``NaN``.
    If a sibling ``.py`` file exists, its content is loaded into
    ``metadata["generated_code"]``.
    Raises ``json.JSONDecodeError`` with the file path included in the
    message if the file contains invalid JSON.
    """
    path = Path(path)
    with path.open() as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                f"Corrupted prediction file {path}: {exc.msg}",
                exc.doc,
                exc.pos,
            ) from exc
    data["values"] = [_restore_nan(v) for v in data["values"]]

    # Restore generated_code from sibling .py file if it exists.
    code_path = path.with_suffix(".py")
    metadata = data.setdefault("metadata", {})
    generated_code_present = metadata.get("generated_code_present")
    if code_path.exists():
        if generated_code_present is False:
            warnings.warn(
                f"Ignoring stale generated-code sibling for {path}",
                stacklevel=2,
            )
        else:
            code = code_path.read_text()
            expected_hash = metadata.get("generated_code_sha256")
            if expected_hash is not None:
                code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
                if code_hash != expected_hash:
                    warnings.warn(
                        f"Ignoring mismatched generated-code sibling for {path}",
                        stacklevel=2,
                    )
                else:
                    metadata["generated_code"] = code
            else:
                metadata["generated_code"] = code

    metadata.pop("generated_code_present", None)
    metadata.pop("generated_code_sha256", None)

    return Prediction(**data)


def load_predictions_dir(dir_path: str | Path) -> dict[str, Prediction]:
    """Load all prediction JSON files from a directory.

    Returns a dict mapping method_name to Prediction. Files must have
    a ``.json`` extension. When duplicate method names are found, the
    last file in sorted order wins (latest timestamp).
    """
    dir_path = Path(dir_path)
    paths = sorted(dir_path.glob("*.json"))
    if not paths:
        return {}
    return load_predictions_paths(paths)


def _prediction_family_key(path: Path) -> tuple[Path, str, str]:
    timestamp = extract_timestamp(path)
    stem = path.stem
    if timestamp is not None:
        stem = stem[: -len(timestamp.strftime("_%Y%m%d_%H%M%S"))]
    return (path.parent, stem, path.suffix)


def _candidate_sort_key(path: Path) -> tuple[int, Any, str]:
    timestamp = extract_timestamp(path)
    if timestamp is None:
        return (0, "", path.name)
    return (1, timestamp, path.name)


def _load_selected_prediction(path: Path) -> Prediction:
    return load_prediction(path)


def _load_prediction_family(paths: list[Path]) -> tuple[Path, Prediction]:
    ordered = sorted(paths, key=_candidate_sort_key, reverse=True)
    newest = ordered[0]

    try:
        pred = _load_selected_prediction(newest)
    except json.JSONDecodeError:
        warnings.warn(
            f"Corrupted newest prediction file {newest}; failing load",
            stacklevel=3,
        )
        raise

    for older in ordered[1:]:
        try:
            _load_selected_prediction(older)
        except json.JSONDecodeError:
            warnings.warn(
                f"Corrupted older prediction file {older}; ignoring because newer "
                f"artifact {newest} was selected",
                stacklevel=3,
            )

    return newest, pred


def load_predictions_paths(paths: list[Path]) -> dict[str, Prediction]:
    """Load prediction JSON files from an explicit list of paths.

    Timestamped sibling files are grouped by filename family and only the
    newest artifact in each family is selected as the canonical prediction.
    Older siblings are inspected only to warn about corruption; they do not
    override the newest selection. If the newest artifact is corrupt, loading
    fails after emitting a warning. When duplicate method names are still found
    across different selected files, the last selected file wins and a warning
    is emitted.
    """
    if not paths:
        return {}

    indexed_paths = list(enumerate(paths))
    grouped: dict[tuple[Path, str, str], list[tuple[int, Path]]] = {}
    for idx, path in indexed_paths:
        grouped.setdefault(_prediction_family_key(path), []).append((idx, path))

    selected: list[tuple[int, Path, Prediction]] = []
    for family_paths in grouped.values():
        family_only_paths = [path for _, path in family_paths]
        selected_path, pred = _load_prediction_family(family_only_paths)
        selected_index = max(
            idx for idx, family_path in family_paths if family_path == selected_path
        )
        selected.append((selected_index, selected_path, pred))

    predictions: dict[str, Prediction] = {}
    seen_paths: dict[str, Path] = {}
    for _, path, pred in sorted(selected, key=lambda item: item[0]):
        if pred.method_name in predictions:
            prev_path = seen_paths[pred.method_name]
            warnings.warn(
                f"Duplicate prediction for {pred.method_name!r}: "
                f"{path} overrides {prev_path}",
                stacklevel=2,
            )
        predictions[pred.method_name] = pred
        seen_paths[pred.method_name] = path
    return predictions


def load_predictions_from_path(path: str | Path) -> dict[str, Prediction]:
    """Load predictions from a file, directory, or timestamped sibling family."""
    path = Path(path)
    if path.is_dir():
        return load_predictions_dir(path)
    if path.exists():
        pred = load_prediction(path)
        return {pred.method_name: pred}

    matches = sorted(path.parent.glob(timestamped_path_glob(path, default_suffix=".json")))
    if matches:
        return load_predictions_paths(matches)
    raise FileNotFoundError(f"Prediction path not found: {path}")


def load_predictions_from_sources(
    sources: tuple[PredictionSourceConfig, ...] | list[PredictionSourceConfig],
    *,
    base: Path | None = None,
) -> dict[str, Prediction]:
    """Load and filter predictions from evaluation-config sources.

    Evaluation configs store ``prediction_sources`` paths catalog-relative (e.g.
    ``data/predictions/<env>/<dir>``). Pass the owning catalog root as ``base``
    to resolve them against it; relative paths are joined to ``base`` while
    absolute paths are left untouched. With ``base=None`` paths resolve
    CWD-relative.
    """
    predictions: dict[str, Prediction] = {}
    used_methods: set[str] = set()
    for source in sources:
        paths: list[Path] = []
        if source.path:
            paths.append(Path(source.path))
        if source.paths:
            paths.extend(Path(p) for p in source.paths)
        if not paths:
            raise ValueError("prediction_sources entries must include path or paths")
        if base is not None:
            paths = [base / p for p in paths]

        include = set(source.include_methods or [])
        exclude = set(source.exclude_methods or [])
        source_type = source.type
        prefix = source.method_name_prefix or ""

        for path in paths:
            candidates = load_predictions_from_path(path)
            for pred in candidates.values():
                if source_type and pred.method_type != source_type:
                    continue
                if include and pred.method_name not in include:
                    continue
                if pred.method_name in exclude:
                    continue
                effective_name = f"{prefix}{pred.method_name}"
                if effective_name in used_methods:
                    raise ValueError(
                        f"Duplicate method_name across sources: {effective_name}"
                    )
                if prefix:
                    pred.method_name = effective_name
                predictions[effective_name] = pred
                used_methods.add(effective_name)

    if not predictions:
        raise ValueError("No predictions matched prediction_sources filters")
    return predictions
