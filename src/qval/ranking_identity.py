"""Stable identity helpers for ranking prediction coverage."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalize_descriptor(value: Any) -> Any:
    """Normalize nested descriptors into stable JSON-compatible structures."""
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return _normalize_descriptor(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_descriptor(inner)
            for key, inner in value.items()
        }
    if isinstance(value, tuple | list):
        return [_normalize_descriptor(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            "__type__": type(value).__name__,
            "fields": _normalize_descriptor(vars(value)),
        }
    return repr(value)


def stable_hash(value: Any) -> str:
    """Return a short stable hash for a JSON-serializable payload."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ranking_point_identity_payload(point: Any) -> dict[str, Any]:
    """Build the canonical identity payload for one ranking point."""
    return {
        "trajectory_index": point.trajectory_index,
        "step_index": point.step_index,
        "candidates": [
            {
                "source": (
                    candidate.source.name
                    if hasattr(candidate.source, "name")
                    else str(candidate.source)
                ),
                "action": _normalize_descriptor(candidate.action),
                "extracted_action": candidate.extracted_action,
                "resolved_action": candidate.resolved_action,
            }
            for candidate in point.candidates
        ],
    }


def ranking_point_hash(point: Any) -> str:
    """Return the stable identity hash for one ranking point."""
    return stable_hash(ranking_point_identity_payload(point))


def build_ranking_coverage_metadata(
    dataset_fingerprint: Any,
    ranking_points: list[Any],
) -> dict[str, Any]:
    """Build stable coverage metadata for an ordered ranking-point list."""
    fp_str = str(getattr(dataset_fingerprint, "sha256", dataset_fingerprint))
    ordered_points = [ranking_point_identity_payload(point) for point in ranking_points]
    point_hashes = [stable_hash(payload) for payload in ordered_points]
    return {
        "dataset_fingerprint": fp_str,
        "ranking_point_hashes": point_hashes,
        "ranking_points_hash": stable_hash(ordered_points),
        "num_ranking_points": len(ranking_points),
        "total_candidates": sum(len(point.candidates) for point in ranking_points),
    }
