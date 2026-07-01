"""Shared helpers for resolving timestamped artifact paths."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


def extract_timestamp(path: Path) -> datetime | None:
    """Extract a trailing ``_YYYYmmdd_HHMMSS`` timestamp from a path stem."""
    match = _TIMESTAMP_RE.search(path.stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def timestamped_path_glob(path: Path, *, default_suffix: str) -> str:
    """Return the glob pattern matching timestamped siblings for ``path``."""
    suffix = path.suffix or default_suffix
    stem = path.stem if path.suffix else path.name
    return f"{stem}_*{suffix}"


def resolve_latest_artifact_path(
    path: str | Path,
    *,
    suffix: str,
    artifact_label: str,
) -> Path:
    """Resolve an artifact path to the newest timestamped file when needed.

    Supports three modes:
    - If ``path`` is a directory, pick the latest timestamped ``*{suffix}`` inside.
    - If ``path`` exists as a file, use it directly.
    - If ``path`` does not exist, look for timestamped siblings and pick the latest.
    """
    candidate = Path(path)
    if candidate.is_dir():
        matches = sorted(candidate.glob(f"*{suffix}"))
        if not matches:
            raise FileNotFoundError(f"No {artifact_label} files in {candidate}")
        return _select_latest_timestamped_path(matches, artifact_label=artifact_label)
    if candidate.exists():
        return candidate

    matches = sorted(candidate.parent.glob(timestamped_path_glob(candidate, default_suffix=suffix)))
    if matches:
        return _select_latest_timestamped_path(matches, artifact_label=artifact_label)
    raise FileNotFoundError(f"{artifact_label.capitalize()} path not found: {candidate}")


def _select_latest_timestamped_path(paths: list[Path], *, artifact_label: str) -> Path:
    with_ts: list[tuple[datetime, Path]] = []
    for path in paths:
        timestamp = extract_timestamp(path)
        if timestamp is not None:
            with_ts.append((timestamp, path))
    if not with_ts:
        raise ValueError(f"No timestamped {artifact_label} files found")
    return max(with_ts, key=lambda pair: pair[0])[1]
