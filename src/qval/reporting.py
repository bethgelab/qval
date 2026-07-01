"""Tabulation of correlation results into Markdown or LaTeX tables."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

DEFAULT_METRICS: tuple[str, ...] = (
    "pearson",
    "spearman",
    "kendall_tau",
)

RANKING_METRICS: tuple[str, ...] = (
    "ranking_spearman_mean",
    "ranking_spearman_std",
    "ranking_num_valid",
    "ranking_num_skipped",
)

METRIC_DISPLAY: dict[str, str] = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "kendall_tau": "Kendall τ",
    "ranking_spearman_mean": "Mean Spearman",
    "ranking_spearman_std": "Std",
    "ranking_num_valid": "Valid",
    "ranking_num_skipped": "Skipped",
}

MISSING = "—"

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cell:
    """A single table cell.

    Single-sample cells carry ``value`` (and optionally ``p_value``).
    Multi-sample cells carry ``value`` (the mean), ``std``,
    ``min_value``, ``max_value``, and ``num_samples``. ``p_value`` is
    meaningful only for single-sample cells — the upstream aggregator
    in ``evaluate.py`` drops per-sample p-values.
    """

    value: float | int | None
    p_value: float | None = None
    std: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    num_samples: int | None = None

    def format_text(self, decimals: int = 3, *, show_p_values: bool = True) -> str:
        v = self.value
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return MISSING
        if (
            isinstance(v, int)
            and self.p_value is None
            and self.std is None
            and self.num_samples is None
        ):
            return str(v)
        s = f"{v:.{decimals}f}"
        if (
            self.std is not None
            and self.num_samples is not None
            and self.num_samples > 1
        ):
            out = f"{s} ± {self.std:.{decimals}f}"
            if self.min_value is not None and self.max_value is not None:
                out = (
                    f"{out} [{self.min_value:.{decimals}f}, "
                    f"{self.max_value:.{decimals}f}]"
                )
            return f"{out} (N={self.num_samples})"
        if show_p_values and self.p_value is not None:
            s = f"{s} ({_format_p(self.p_value, decimals)})"
        return s


def _format_p(p: float, decimals: int) -> str:
    """Format a p-value; use '<threshold' notation for values below precision."""
    threshold = 10 ** (-decimals)
    if p < threshold:
        return f"p<{threshold:.{decimals}f}"
    return f"p={p:.{decimals}f}"


@dataclass(frozen=True)
class Table:
    """A table of correlation results for a single GT method."""

    gt_name: str
    assumption: str
    env_name: str | None
    methods: tuple[str, ...]
    metrics: tuple[str, ...]
    cells: dict[tuple[str, str], Cell]


def _cell_from_entry(method_entry: dict, metric: str) -> Cell:
    """Build a Cell from a methods[method_name] dict for the given metric.

    Handles single-sample (``correlations``) and multi-sample
    (``aggregate_correlations``) forms. Only single-sample cells carry a
    p-value — the upstream aggregator drops per-sample p-values.
    """
    if "aggregate_correlations" in method_entry:
        agg = method_entry["aggregate_correlations"].get(metric)
        n = method_entry.get("num_samples")
        if agg is None:
            return Cell(value=None, num_samples=n)
        return Cell(
            value=agg.get("mean"),
            std=agg.get("std"),
            min_value=agg.get("min"),
            max_value=agg.get("max"),
            num_samples=n,
        )

    entry = method_entry.get("correlations", {}).get(metric)
    if entry is None:
        return Cell(value=None)
    return Cell(
        value=entry.get("correlation"),
        p_value=entry.get("p_value"),
    )


def _ranking_cells_from_entry(method_entry: dict) -> dict[str, Cell]:
    """Build ranking-table cells from a ranking method entry."""
    entry = method_entry.get("correlations", {}).get("ranking_spearman")
    if entry is None:
        return {metric: Cell(value=None) for metric in RANKING_METRICS}
    return {
        "ranking_spearman_mean": Cell(value=entry.get("mean")),
        "ranking_spearman_std": Cell(value=entry.get("std")),
        "ranking_num_valid": Cell(value=entry.get("num_valid")),
        "ranking_num_skipped": Cell(value=entry.get("num_skipped")),
    }


def build_tables(
    summaries: list[dict],
    *,
    metrics: tuple[str, ...] | None = None,
) -> list[Table]:
    """Merge N summary.json dicts into a list of Tables (one per GT method).

    Args:
        summaries: Parsed summary.json dicts.
        metrics: If provided, use exactly these columns (missing cells get
            an em-dash). Otherwise, use DEFAULT_METRICS order restricted to
            metrics present in the data.

    Raises:
        ValueError: if the same (gt_name, method_name) appears in multiple
            summaries. Experiments for a fixed (env, model) shouldn't
            silently overwrite rows.
    """
    accum: dict[str, dict] = {}
    seen_metrics: dict[str, list[str]] = {}

    for summary in summaries:
        env_name = summary.get("experiment_info", {}).get("env_name")
        for gt_name, gt_entry in summary.get("estimations", {}).items():
            granularity = gt_entry.get("granularity", "point")

            bucket = accum.setdefault(
                gt_name,
                {
                    "assumption": gt_entry.get("assumption", "unknown"),
                    "env_name": env_name,
                    "granularity": granularity,
                    "methods": [],
                    "cells": {},
                },
            )
            if bucket["granularity"] != granularity:
                raise ValueError(
                    f"GT {gt_name!r} mixes granularities "
                    f"{bucket['granularity']!r} and {granularity!r} across summaries."
                )
            metric_order = seen_metrics.setdefault(gt_name, [])

            for method_name, method_entry in gt_entry.get("methods", {}).items():
                if method_name in bucket["methods"]:
                    raise ValueError(
                        f"Duplicate method {method_name!r} for GT {gt_name!r} "
                        f"across summaries; refusing to overwrite silently."
                    )
                bucket["methods"].append(method_name)

                if granularity == "ranking":
                    ranking_cells = _ranking_cells_from_entry(method_entry)
                    for metric_name, cell in ranking_cells.items():
                        if metric_name not in metric_order:
                            metric_order.append(metric_name)
                        bucket["cells"][(method_name, metric_name)] = cell
                else:
                    source_metrics = (
                        method_entry.get("aggregate_correlations", {}).keys()
                        if "aggregate_correlations" in method_entry
                        else method_entry.get("correlations", {}).keys()
                    )
                    for m in source_metrics:
                        if m not in metric_order:
                            metric_order.append(m)
                        bucket["cells"][(method_name, m)] = _cell_from_entry(
                            method_entry, m,
                        )

    tables: list[Table] = []
    for gt_name, bucket in accum.items():
        seen = seen_metrics.get(gt_name, [])
        if metrics is not None:
            cols = tuple(metrics)
        elif bucket["granularity"] == "ranking":
            cols = tuple(m for m in RANKING_METRICS if m in seen)
            for m in seen:
                if m not in cols:
                    cols = (*cols, m)
        else:
            cols = tuple(m for m in DEFAULT_METRICS if m in seen)
            for m in seen:
                if m not in cols:
                    cols = (*cols, m)

        cells = dict(bucket["cells"])
        for method in bucket["methods"]:
            for m in cols:
                cells.setdefault((method, m), Cell(value=None))

        tables.append(
            Table(
                gt_name=gt_name,
                assumption=bucket["assumption"],
                env_name=bucket["env_name"],
                methods=tuple(bucket["methods"]),
                metrics=cols,
                cells=cells,
            )
        )

    return tables


def _pad(s: str, width: int) -> str:
    return s + " " * (width - len(s))


def render_markdown(
    table: Table,
    decimals: int = 3,
    *,
    show_p_values: bool = True,
) -> str:
    """Render one Table as a Markdown section (heading + pipe table)."""
    header_cells = ["Method"] + [METRIC_DISPLAY.get(m, m) for m in table.metrics]
    rows: list[list[str]] = []
    for method in table.methods:
        row = [method]
        for metric in table.metrics:
            cell = table.cells.get((method, metric), Cell(value=None))
            row.append(cell.format_text(decimals=decimals, show_p_values=show_p_values))
        rows.append(row)

    widths = [
        max(len(header_cells[i]), *(len(r[i]) for r in rows))
        if rows
        else len(header_cells[i])
        for i in range(len(header_cells))
    ]
    widths = [max(w, 3) for w in widths]

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(_pad(c, w) for c, w in zip(cells, widths)) + " |"

    divider = "| " + " | ".join("-" * w for w in widths) + " |"

    caption_env = f" — {table.env_name}" if table.env_name else ""
    lines = [
        f"### GT: `{table.gt_name}` (assumption: {table.assumption}){caption_env}",
        "",
        fmt_row(header_cells),
        divider,
        *[fmt_row(r) for r in rows],
    ]
    return "\n".join(lines) + "\n"


_LATEX_ESCAPE = str.maketrans({
    "_": r"\_",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "$": r"\$",
})


def _latex_escape(s: str) -> str:
    return s.translate(_LATEX_ESCAPE)


def _cell_to_latex(cell: Cell, decimals: int, *, show_p_values: bool) -> str:
    """Format a Cell for LaTeX: math-mode ± when multi-sample, em-dash for missing."""
    v = cell.value
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    if (
        isinstance(v, int)
        and cell.p_value is None
        and cell.std is None
        and cell.num_samples is None
    ):
        return str(v)
    base = f"{v:.{decimals}f}"
    if (
        cell.std is not None
        and cell.num_samples is not None
        and cell.num_samples > 1
    ):
        out = f"${base} \\pm {cell.std:.{decimals}f}$"
        if cell.min_value is not None and cell.max_value is not None:
            out = (
                f"{out} [{cell.min_value:.{decimals}f}, "
                f"{cell.max_value:.{decimals}f}]"
            )
        return f"{out} (N={cell.num_samples})"
    if show_p_values and cell.p_value is not None:
        return f"{base} ({_format_p(cell.p_value, decimals)})"
    return base


def render_latex(
    table: Table,
    decimals: int = 3,
    *,
    show_p_values: bool = True,
) -> str:
    """Render one Table as a LaTeX `table` with `tabular` + booktabs rules."""
    headers = ["Method"] + [METRIC_DISPLAY.get(m, m) for m in table.metrics]
    col_spec = "l" + "c" * len(table.metrics)

    body_rows: list[str] = []
    for method in table.methods:
        cells_fmt = [_latex_escape(method)]
        for metric in table.metrics:
            cell = table.cells.get((method, metric), Cell(value=None))
            cells_fmt.append(_cell_to_latex(cell, decimals, show_p_values=show_p_values))
        body_rows.append(" & ".join(cells_fmt) + r" \\")

    caption_env = f" ({_latex_escape(table.env_name)})" if table.env_name else ""
    caption = (
        f"Correlations vs.\\ GT {_latex_escape(table.gt_name)} "
        f"(assumption: {_latex_escape(table.assumption)}){caption_env}."
    )
    label = f"tab:corr_{table.gt_name}"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render(
    tables: list[Table],
    *,
    fmt: str,
    decimals: int = 3,
    show_p_values: bool = True,
) -> str:
    """Render a list of tables in the requested format, separated by blank lines."""
    if not tables:
        raise ValueError("No tables to render (empty input).")
    if fmt == "markdown":
        render_one = render_markdown
    elif fmt == "latex":
        render_one = render_latex
    else:
        raise ValueError(f"Unknown format: {fmt!r} (expected 'markdown' or 'latex').")
    return "\n".join(
        render_one(t, decimals=decimals, show_p_values=show_p_values)
        for t in tables
    )
