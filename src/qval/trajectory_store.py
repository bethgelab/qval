"""Observable trajectory store: save/load trajectories without hidden state.

Provides ``ObservableTrajectory`` and ``ObservableTransition`` — frozen
dataclasses that store only the observable parts of a trajectory (observation
text, action text, reward, termination). These are suitable for:

- Persisting trajectories to disk (JSON format)
- Loading example trajectories into ``MethodContext`` for prompts
- Post-hoc analysis of experiment runs

The ``from_trajectory_result()`` function converts a live llenvs
``TrajectoryResult`` to observable format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llenvs.core.state import ImageContent, ObservationImages

from qval.methods.serialization import (
    extract_images,
    serialize_action,
    serialize_observation,
)


@dataclass(frozen=True)
class ObservableTransition:
    """Observable-only transition (no hidden state).

    Attributes:
        observation: Text representation of the state's observation.
        raw_action: Full model generation text (including thinking tokens).
        extracted_action: Clean action extracted by the environment adapter
            (tier 2), or ``None`` when extraction failed.
        resolved_action: Formatted native action from the adapter (tier 3),
            e.g. ``"right"`` for gym action 2. ``None`` when the action was
            invalid or the adapter doesn't provide this.
        reward: Reward received for this transition.
        terminated: Whether this transition ended the episode.
        observation_images: Images from the observation, separated by source
            (task vs state).  Empty for text-only environments.
    """

    observation: str
    raw_action: str
    extracted_action: str | None
    resolved_action: str | None
    reward: float
    terminated: bool
    observation_images: ObservationImages = ObservationImages()


@dataclass(frozen=True)
class ObservableTrajectory:
    """Observable-only trajectory for storage and prompts.

    Attributes:
        transitions: Tuple of observable transitions in chronological order.
        total_reward: Cumulative reward across all transitions.
        success: Whether the trajectory was successful.
    """

    transitions: tuple[ObservableTransition, ...]
    total_reward: float
    success: bool


def from_trajectory_result(
    result: Any,
    reward_signal_name: str | None = "correctness",
) -> ObservableTrajectory:
    """Convert a live TrajectoryResult to observable-only format.

    Uses ``resolved_action`` from each transition when available (clean
    extracted text from the llenvs adapter), falling back to raw action text.

    Args:
        result: A llenvs ``TrajectoryResult`` instance.
        reward_signal_name: Which reward signal to extract (None for total).

    Returns:
        An ObservableTrajectory with text observations and actions.
    """
    error = getattr(result, "metadata", {}).get("error")
    if error:
        raise ValueError(
            "Cannot serialize a failed trajectory result to observable format: "
            f"{error}"
        )

    transitions: list[ObservableTransition] = []

    for t in result.trajectory.transitions:
        obs_text = serialize_observation(t.state)
        raw_action = serialize_action(t.action)
        extracted_action = t.extracted_action
        obs_images = extract_images(t.state)

        if reward_signal_name is not None:
            signal = t.rewards.by_name(reward_signal_name, required=True)
            reward = signal.reward if signal.reward is not None else 0.0
        else:
            reward = t.rewards.total

        terminated = t.next_state.metadata.is_terminal

        transitions.append(ObservableTransition(
            observation=obs_text,
            raw_action=raw_action,
            extracted_action=extracted_action,
            resolved_action=t.resolved_action,
            reward=reward,
            terminated=terminated,
            observation_images=obs_images,
        ))

    return ObservableTrajectory(
        transitions=tuple(transitions),
        total_reward=result.total_reward,
        success=result.success,
    )


def _serialize_images(images: ObservationImages) -> dict[str, Any]:
    """Serialize ObservationImages to a JSON-compatible dict."""
    def _img_list(imgs: tuple[ImageContent, ...]) -> list[dict[str, str]]:
        return [{"data": img.data, "media_type": img.media_type} for img in imgs]

    return {
        "task": _img_list(images.task),
        "state": _img_list(images.state),
    }


def _deserialize_images(data: dict[str, Any] | None) -> ObservationImages:
    """Deserialize ObservationImages from a JSON dict.

    Handles missing keys for backward compatibility with old JSON files.
    """
    if not data:
        return ObservationImages()
    return ObservationImages(
        task=tuple(
            ImageContent(data=img["data"], media_type=img.get("media_type", "image/png"))
            for img in data.get("task", [])
        ),
        state=tuple(
            ImageContent(data=img["data"], media_type=img.get("media_type", "image/png"))
            for img in data.get("state", [])
        ),
    )


def save_trajectories(
    trajectories: list[ObservableTrajectory],
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    include_images: bool = True,
) -> None:
    """Save trajectories to JSON.

    Args:
        trajectories: List of ObservableTrajectory instances.
        path: Output file path.
        metadata: Optional metadata dict included in the JSON.
        include_images: Whether to include images in the JSON output.
            Set to ``False`` for lightweight saves (e.g., example trajectories).
    """
    data = {
        "version": 1,
        "metadata": metadata or {},
        "trajectories": [
            {
                "total_reward": traj.total_reward,
                "success": traj.success,
                "transitions": [
                    {
                        "observation": t.observation,
                        "raw_action": t.raw_action,
                        "extracted_action": t.extracted_action,
                        "resolved_action": t.resolved_action,
                        "reward": t.reward,
                        "terminated": t.terminated,
                        **(
                            {"observation_images": _serialize_images(t.observation_images)}
                            if include_images and t.observation_images
                            else {}
                        ),
                    }
                    for t in traj.transitions
                ],
            }
            for traj in trajectories
        ],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def load_trajectories(path: str | Path) -> list[ObservableTrajectory]:
    """Load trajectories from JSON.

    Handles both old JSON files (no ``observation_images`` key) and new
    files with images.

    Args:
        path: Path to the JSON file.

    Returns:
        List of ObservableTrajectory instances.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    trajectories: list[ObservableTrajectory] = []
    for traj_data in data["trajectories"]:
        transitions = tuple(
            ObservableTransition(
                observation=t["observation"],
                raw_action=t["raw_action"],
                extracted_action=t["extracted_action"],
                resolved_action=t["resolved_action"],
                reward=t["reward"],
                terminated=t["terminated"],
                observation_images=_deserialize_images(t.get("observation_images")),
            )
            for t in traj_data["transitions"]
        )
        trajectories.append(ObservableTrajectory(
            transitions=transitions,
            total_reward=traj_data["total_reward"],
            success=traj_data["success"],
        ))

    return trajectories
