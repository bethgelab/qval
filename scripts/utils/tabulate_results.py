"""Render correlation summary.json files into a Markdown or LaTeX table.

Usage:
    python scripts/tabulate_results.py <results_dir> [options]

Scans ``<results_dir>`` recursively for ``summary*.json`` files, merges
them, and prints (or writes) one table per GT method. Environment and
model are assumed fixed across the scanned files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from qval.reporting import build_tables, render

logger = logging.getLogger("tabulate_results")


def _load_summaries(results_dir: Path) -> list[dict]:
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")
    paths = sorted(results_dir.rglob("summary*.json"))
    if not paths:
        raise SystemExit(f"No summary*.json files found under {results_dir}")
    summaries: list[dict] = []
    for p in paths:
        logger.info("Loading %s", p)
        with p.open() as f:
            summaries.append(json.load(f))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tabulate correlation results from summary.json files.",
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory to search recursively for summary*.json files.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "latex"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write to this file; default stdout.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for correlation values (default: 3).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Subset of correlation metrics to include, in this order "
            "(e.g. --metrics pearson spearman kendall_tau). "
            "Default: all present, in pearson/spearman/kendall_tau order."
        ),
    )
    parser.add_argument(
        "--no-pvalues",
        dest="show_p_values",
        action="store_false",
        default=True,
        help=(
            "Suppress per-cell p-values (default: show as '0.847 (p=0.012)' "
            "for single-sample cells)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    summaries = _load_summaries(args.results_dir)
    tables = build_tables(
        summaries,
        metrics=tuple(args.metrics) if args.metrics else None,
    )
    text = render(
        tables,
        fmt=args.format,
        decimals=args.decimals,
        show_p_values=args.show_p_values,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        logger.info("Wrote %s", args.output)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
