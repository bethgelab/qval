"""Inspect a dataset and print candidate-action statistics per ranking point."""

from __future__ import annotations

import argparse
from collections import Counter

from qval.data_cache import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print candidate-action counts for ranking points in a dataset.",
    )
    parser.add_argument("dataset", help="Path to the dataset pickle file.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    if not dataset.ranking_points:
        print("No ranking points in this dataset.")
        return

    counts: Counter[int] = Counter()
    for rp in dataset.ranking_points:
        counts[len(rp.candidates)] += 1

    for n_candidates, n_points in sorted(counts.items()):
        print(f"{n_points} points with {n_candidates} candidate actions")


if __name__ == "__main__":
    main()
