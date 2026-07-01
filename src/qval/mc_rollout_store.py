"""Persistent append-only storage for MC rollout trajectories."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_log = logging.getLogger(__name__)

PointKey = tuple[int, ...]


_DIFF_REPR_MAX = 200


def _summarize_for_diff(value: Any) -> str:
    """Render a spec value for the diff log, summarizing long content.

    Long ``repr()`` outputs (e.g. multi-KB system prompts) would dominate
    the warning line and make small-but-meaningful diffs hard to spot.
    Values whose ``repr()`` exceeds ``_DIFF_REPR_MAX`` are replaced with
    ``<str length=N, sha256=...>`` (or the type-named equivalent for
    non-strings) so the user can still verify equality across runs while
    the actual diff line stays readable.
    """
    rendered = repr(value)
    if len(rendered) <= _DIFF_REPR_MAX:
        return rendered
    if isinstance(value, str):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return f"<str length={len(value)}, sha256={digest}>"
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
    return f"<{type(value).__name__} repr_length={len(rendered)}, sha256={digest}>"


def _spec_field_diff(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Render a flat list of field-level differences between two spec dicts.

    Used by ``MCRolloutStore.open(trust_existing_hash=True)`` to surface
    in the warning exactly which BackendConfig/prompt/environment fields
    changed between the cached and the requested generation spec — the
    diagnostic users need to decide whether reuse is actually safe. Large
    string/list values are summarized via ``_summarize_for_diff`` so the
    rendered line stays readable.
    """
    diffs: list[str] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            sub_prefix = f"{prefix}.{key}" if prefix else key
            if key not in old:
                diffs.append(f"{sub_prefix}: <missing> -> {_summarize_for_diff(new[key])}")
            elif key not in new:
                diffs.append(f"{sub_prefix}: {_summarize_for_diff(old[key])} -> <missing>")
            else:
                diffs.extend(_spec_field_diff(old[key], new[key], sub_prefix))
    elif isinstance(old, list) and isinstance(new, list):
        if old != new:
            diffs.append(
                f"{prefix or '<root>'}: "
                f"{_summarize_for_diff(old)} -> {_summarize_for_diff(new)}"
            )
    elif old != new:
        diffs.append(
            f"{prefix or '<root>'}: "
            f"{_summarize_for_diff(old)} -> {_summarize_for_diff(new)}"
        )
    return diffs


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)


def _atomic_write_pickle(path: Path, value: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(value, f)
    tmp_path.replace(path)


def _point_key_to_str(point_key: PointKey) -> str:
    return ":".join(str(part) for part in point_key)


def _point_key_from_str(value: str) -> PointKey:
    return tuple(int(part) for part in value.split(":"))


@dataclass(frozen=True)
class DatasetFingerprint:
    """Stable identity for a dataset artifact."""

    sha256: str
    path_hint: str | None = field(default=None, compare=False)


def dataset_fingerprint_for_file(path: str | Path) -> DatasetFingerprint:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return DatasetFingerprint(
        sha256=digest.hexdigest(),
        path_hint=str(resolved),
    )


def dataset_fingerprint_for_object(
    value: Any,
    *,
    path_hint: str | None = None,
) -> DatasetFingerprint:
    """Hash an already-loaded dataset object when no backing file is available."""
    payload = pickle.dumps(value)
    return DatasetFingerprint(
        sha256=hashlib.sha256(payload).hexdigest(),
        path_hint=path_hint,
    )


@dataclass(frozen=True)
class ReturnSpec:
    """How to derive scalar returns from stored trajectories."""

    discount_factor: float = 1.0
    reward_signal_name: str | None = "correctness"
    step_penalty: float | None = None


@dataclass(frozen=True)
class StoredTransition:
    """Persisted logical transition."""

    state: Any
    action: Any
    next_state: Any
    rewards: Any
    extracted_action: str | None = None
    resolved_action: str | None = None


@dataclass(frozen=True)
class StoredTrajectory:
    """Persisted logical trajectory for GT recomputation."""

    initial_state: Any
    transitions: tuple[StoredTransition, ...]

    @classmethod
    def from_mapping(
        cls,
        *,
        initial_state: Any,
        transitions: list[dict[str, Any]],
    ) -> StoredTrajectory:
        return cls(
            initial_state=initial_state,
            transitions=tuple(StoredTransition(**transition) for transition in transitions),
        )


def stored_trajectory_from_live(trajectory: Any) -> StoredTrajectory:
    """Convert a live llenvs trajectory to the persisted logical format."""
    return StoredTrajectory(
        initial_state=trajectory.initial_state,
        transitions=tuple(
            StoredTransition(
                state=transition.state,
                action=transition.action,
                next_state=transition.next_state,
                rewards=transition.rewards,
                extracted_action=transition.extracted_action,
                resolved_action=transition.resolved_action,
            )
            for transition in trajectory.transitions
        ),
    )


def stored_trajectory_return(
    trajectory: StoredTrajectory,
    return_spec: ReturnSpec,
    *,
    start_index: int = 0,
) -> float:
    """Compute a discounted return from a stored trajectory."""
    total = 0.0
    discount = 1.0
    for transition in trajectory.transitions[start_index:]:
        if return_spec.reward_signal_name is None:
            reward = sum(
                signal.reward or 0.0
                for signal in transition.rewards.signals
                if signal.name != "step_penalty"
            )
        else:
            signal = transition.rewards.by_name(
                return_spec.reward_signal_name,
                required=True,
            )
            reward = 0.0 if signal.reward is None else signal.reward
        if return_spec.step_penalty is not None:
            reward -= return_spec.step_penalty
        total += discount * reward
        discount *= return_spec.discount_factor
    return total


@dataclass(frozen=True)
class RolloutGenerationSpec:
    """Normalized runtime description of a rollout family."""

    method: str
    namespace: str
    primitive: str
    dataset_fingerprint: DatasetFingerprint
    environment: dict[str, Any]
    policy: dict[str, Any]
    prompt: dict[str, Any]

    def normalized_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "namespace": self.namespace,
            "primitive": self.primitive,
            "dataset_fingerprint": asdict(self.dataset_fingerprint),
            "environment": self.environment,
            "policy": self.policy,
            "prompt": self.prompt,
        }

    @property
    def generation_hash(self) -> str:
        payload = json.dumps(
            self.normalized_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShardManifest:
    filename: str
    point_counts: dict[str, int]
    num_trajectories: int


@dataclass(frozen=True)
class RolloutStoreManifest:
    generation_spec: RolloutGenerationSpec
    shards: tuple[ShardManifest, ...] = ()
    point_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "generation_spec": self.generation_spec.normalized_dict(),
                "generation_hash": self.generation_spec.generation_hash,
                "shards": [
                    {
                        "filename": shard.filename,
                        "point_counts": shard.point_counts,
                        "num_trajectories": shard.num_trajectories,
                    }
                    for shard in self.shards
                ],
                "point_counts": self.point_counts,
            },
            indent=2,
            sort_keys=True,
        )


class MCRolloutStore:
    """Append-only shard store for persisted MC trajectories."""

    def __init__(self, root: Path, manifest: RolloutStoreManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.manifest_path = root / "manifest.json"
        self._shards_dir = root / "shards"

    @classmethod
    def open(
        cls,
        root: str | Path,
        generation_spec: RolloutGenerationSpec,
        *,
        trust_existing_hash: bool = False,
    ) -> MCRolloutStore:
        """Open or create a rollout store at ``root``.

        When ``trust_existing_hash`` is False (default), an existing
        manifest or orphan shard with a different ``generation_hash``
        than ``generation_spec`` raises. With it True, the mismatch is
        logged as a WARNING that includes the field-level diff between
        the cached and requested specs, the existing shards are adopted
        as-is, and the manifest is rewritten with the new hash. Callers
        opting in must have separately verified that the rollouts are
        semantically valid for the new spec.
        """
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        shards_dir = root_path / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = root_path / "manifest.json"

        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text())
            existing_hash = manifest_data.get("generation_hash")
            if existing_hash != generation_spec.generation_hash:
                if not trust_existing_hash:
                    raise ValueError(
                        "Existing rollout store is incompatible with requested generation spec"
                    )
                diff = _spec_field_diff(
                    manifest_data.get("generation_spec", {}),
                    generation_spec.normalized_dict(),
                )
                _log.warning(
                    "Adopting rollout store at %s with mismatched generation_hash "
                    "(trust_existing_hash=True). Cached hash %s -> requested hash %s. "
                    "Spec differences (cached -> requested):\n  %s",
                    root_path,
                    existing_hash,
                    generation_spec.generation_hash,
                    "\n  ".join(diff) if diff else "(no field-level diff captured)",
                )
            known = {
                shard_data["filename"]: ShardManifest(
                    filename=shard_data["filename"],
                    point_counts=dict(shard_data["point_counts"]),
                    num_trajectories=int(shard_data["num_trajectories"]),
                )
                for shard_data in manifest_data.get("shards", [])
            }
        else:
            known = {}

        actual_files = sorted(path.name for path in shards_dir.glob("shard_*.pkl"))
        manifests: list[ShardManifest] = []
        point_counts: dict[str, int] = {}

        for filename in actual_files:
            shard_manifest = known.get(filename)
            if shard_manifest is None:
                with (shards_dir / filename).open("rb") as f:
                    shard_payload = pickle.load(f)
                if shard_payload["generation_hash"] != generation_spec.generation_hash:
                    if not trust_existing_hash:
                        raise ValueError(f"Incompatible orphan shard: {filename}")
                    _log.warning(
                        "Adopting orphan shard %s with mismatched generation_hash "
                        "(trust_existing_hash=True). Shard hash %s -> requested hash %s.",
                        filename,
                        shard_payload["generation_hash"],
                        generation_spec.generation_hash,
                    )
                shard_manifest = ShardManifest(
                    filename=filename,
                    point_counts=dict(shard_payload["point_counts"]),
                    num_trajectories=int(shard_payload["num_trajectories"]),
                )
            manifests.append(shard_manifest)
            for point_key, count in shard_manifest.point_counts.items():
                point_counts[point_key] = point_counts.get(point_key, 0) + count

        # ``open()`` only reconciles and rewrites the manifest; it does not
        # rename ``root`` if the path encodes an old hash. The higher-level
        # sibling-adoption path in ``estimator.py`` renames first so normal
        # rollout stores keep dirname == generation_hash. Direct callers of
        # ``trust_existing_hash`` are responsible for any path maintenance.
        manifest = RolloutStoreManifest(
            generation_spec=generation_spec,
            shards=tuple(manifests),
            point_counts=point_counts,
        )
        store = cls(root_path, manifest)
        store._write_manifest()
        return store

    def count_for(self, point_key: PointKey) -> int:
        return self.manifest.point_counts.get(_point_key_to_str(point_key), 0)

    def append(self, point_trajectories: dict[PointKey, tuple[StoredTrajectory, ...]]) -> None:
        if not point_trajectories:
            return

        next_index = len(self.manifest.shards) + 1
        filename = f"shard_{next_index:06d}.pkl"
        encoded = {
            _point_key_to_str(point_key): trajectories
            for point_key, trajectories in point_trajectories.items()
        }
        point_counts = {
            key: len(value)
            for key, value in encoded.items()
        }
        payload = {
            "generation_hash": self.manifest.generation_spec.generation_hash,
            "point_counts": point_counts,
            "num_trajectories": sum(point_counts.values()),
            "trajectories": encoded,
        }
        _atomic_write_pickle(self._shards_dir / filename, payload)

        new_manifest = RolloutStoreManifest(
            generation_spec=self.manifest.generation_spec,
            shards=self.manifest.shards + (
                ShardManifest(
                    filename=filename,
                    point_counts=point_counts,
                    num_trajectories=payload["num_trajectories"],
                ),
            ),
            point_counts=dict(self.manifest.point_counts),
        )
        for point_key, count in point_counts.items():
            new_manifest.point_counts[point_key] = new_manifest.point_counts.get(point_key, 0) + count
        self.manifest = new_manifest
        self._write_manifest()

    def load_point_trajectories(
        self,
        requested_counts: dict[PointKey, int],
    ) -> dict[PointKey, tuple[StoredTrajectory, ...]]:
        remaining = {
            point_key: requested_counts[point_key]
            for point_key in requested_counts
            if requested_counts[point_key] > 0
        }
        loaded: dict[PointKey, list[StoredTrajectory]] = {
            point_key: []
            for point_key in remaining
        }
        if not remaining:
            return {}

        for shard in self.manifest.shards:
            relevant = False
            for point_key in remaining:
                if shard.point_counts.get(_point_key_to_str(point_key), 0) > 0:
                    relevant = True
                    break
            if not relevant:
                continue
            with (self._shards_dir / shard.filename).open("rb") as f:
                payload = pickle.load(f)
            for point_key, needed in list(remaining.items()):
                key_str = _point_key_to_str(point_key)
                if needed <= 0 or key_str not in payload["trajectories"]:
                    continue
                trajectories = payload["trajectories"][key_str]
                loaded[point_key].extend(trajectories[:needed])
                remaining[point_key] -= min(needed, len(trajectories))
                if remaining[point_key] <= 0:
                    del remaining[point_key]
            if not remaining:
                break

        return {
            point_key: tuple(trajectories)
            for point_key, trajectories in loaded.items()
        }

    def _write_manifest(self) -> None:
        _atomic_write_text(self.manifest_path, self.manifest.to_json())
