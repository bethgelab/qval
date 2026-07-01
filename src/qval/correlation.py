"""Correlation computation between ground truth and predicted signals."""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats

from qval.types import CorrelationMethod, CorrelationResult

_log = logging.getLogger(__name__)


def compute_correlation(
    ground_truth: list[float],
    predicted: list[float],
    method: CorrelationMethod,
) -> CorrelationResult:
    """Compute correlation between ground truth and predicted signal values.

    Pairs where either value is NaN are filtered out before computation.

    Args:
        ground_truth: Ground truth signal values.
        predicted: Predicted signal values from the method being evaluated.
        method: Which correlation metric to use.

    Returns:
        CorrelationResult with the correlation coefficient and p-value.
        ``num_points`` reflects the count after NaN filtering.

    Raises:
        ValueError: If inputs have different lengths or fewer than 2
            non-NaN points remain after filtering.
    """
    if len(ground_truth) != len(predicted):
        msg = (
            f"ground_truth and predicted must have the same length, "
            f"got {len(ground_truth)} and {len(predicted)}"
        )
        raise ValueError(msg)

    # Filter out pairs where either value is NaN
    pairs = [
        (g, p) for g, p in zip(ground_truth, predicted)
        if not (math.isnan(g) or math.isnan(p))
    ]
    num_filtered = len(ground_truth) - len(pairs)
    if num_filtered > 0:
        _log.info(
            "Filtered %d NaN pairs from %d total (%d remaining)",
            num_filtered, len(ground_truth), len(pairs),
        )

    if len(pairs) < 2:
        msg = f"Need at least 2 non-NaN points for correlation, got {len(pairs)}"
        raise ValueError(msg)

    gt = np.array([g for g, _ in pairs])
    pred = np.array([p for _, p in pairs])

    if method == CorrelationMethod.PEARSON:
        corr, p_value = stats.pearsonr(gt, pred)
    elif method == CorrelationMethod.SPEARMAN:
        corr, p_value = stats.spearmanr(gt, pred)
    elif method == CorrelationMethod.KENDALL_TAU:
        corr, p_value = stats.kendalltau(gt, pred)
    elif method == CorrelationMethod.SIGN_AGREEMENT:
        gt_signs = np.sign(gt)
        pred_signs = np.sign(pred)
        n = len(gt)
        successes = int(np.sum(gt_signs == pred_signs))
        corr = float(successes / n)
        p_value = float(stats.binomtest(successes, n, p=0.5).pvalue)
    else:
        msg = f"Unknown correlation method: {method}"
        raise ValueError(msg)

    return CorrelationResult(
        method=method,
        correlation=float(corr),
        p_value=float(p_value),
        num_points=len(pairs),
    )


def compute_ranking_correlation(
    gt_values: list[float],
    eval_values: list[float],
    candidate_counts: list[int] | tuple[int, ...],
) -> dict[str, float]:
    """Compute per-point Spearman correlation for ranking predictions.

    Args:
        gt_values: Flat list of GT Q-values in candidate order.
        eval_values: Flat list of predicted ranks in candidate order.
            ``ranks[i]`` = rank assigned to candidate i (1=best). NaN values
            indicate extraction failures.
        candidate_counts: Number of candidates in each ranking point.

    Returns:
        Dict with ``mean_spearman``, ``std_spearman``, ``num_points``,
        ``num_skipped``, and ``per_point`` (list of per-point correlations,
        NaN for skipped points).

    Raises:
        ValueError: If inputs have incompatible lengths or candidate counts
            are invalid.
    """
    if len(gt_values) != len(eval_values):
        raise ValueError(
            f"gt_values and eval_values must have the same length, "
            f"got {len(gt_values)} and {len(eval_values)}"
        )
    if any(count <= 0 for count in candidate_counts):
        raise ValueError(
            "candidate_counts must all be positive integers"
        )
    total_candidates = sum(candidate_counts)
    if len(gt_values) != total_candidates:
        raise ValueError(
            f"candidate_counts sum to {total_candidates}, but got "
            f"{len(gt_values)} flattened values"
        )

    num_points = len(candidate_counts)
    per_point: list[float] = []
    valid_correlations: list[float] = []
    num_skipped = 0
    offset = 0

    for k in candidate_counts:
        gt_chunk = gt_values[offset : offset + k]
        eval_chunk = eval_values[offset : offset + k]
        offset += k

        if k < 2:
            per_point.append(float("nan"))
            num_skipped += 1
            continue

        # Skip if any eval value is NaN (extraction failure)
        if any(math.isnan(v) for v in eval_chunk):
            per_point.append(float("nan"))
            num_skipped += 1
            continue

        # Skip if all GT Q-values are equal (no meaningful ranking)
        gt_arr = np.array(gt_chunk)
        if np.all(gt_arr == gt_arr[0]):
            per_point.append(float("nan"))
            num_skipped += 1
            continue

        # GT: convert Q-values to ranks (descending, ties averaged)
        gt_ranks = stats.rankdata(-gt_arr, method="average")
        eval_arr = np.array(eval_chunk)

        corr, _ = stats.spearmanr(gt_ranks, eval_arr)
        per_point.append(float(corr))
        valid_correlations.append(float(corr))

    if not valid_correlations:
        return {
            "mean_spearman": float("nan"),
            "std_spearman": float("nan"),
            "num_points": num_points,
            "num_valid": 0,
            "num_skipped": num_skipped,
            "per_point": per_point,
        }

    mean_corr = sum(valid_correlations) / len(valid_correlations)
    if len(valid_correlations) > 1:
        variance = sum((c - mean_corr) ** 2 for c in valid_correlations) / (len(valid_correlations) - 1)
        std_corr = variance ** 0.5
    else:
        std_corr = 0.0

    return {
        "mean_spearman": mean_corr,
        "std_spearman": std_corr,
        "num_points": num_points,
        "num_valid": len(valid_correlations),
        "num_skipped": num_skipped,
        "per_point": per_point,
    }
