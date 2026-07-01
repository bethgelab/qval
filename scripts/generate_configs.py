#!/usr/bin/env python3
"""Generate experiment config YAMLs from the registry catalog.

The catalog (by default the bundled ``catalogs/qval_benchmark/catalog.py``) is
the single source of truth for the experiment matrix. This script expands it
into the per-(env, actor, modality) prediction configs and per-(env, gt_actor)
GT configs under ``<catalog>/configs/prediction/<env>/``, with every
``predictions_dir`` derived from the naming conventions so the pipeline output
lands in the canonical location automatically.

Usage::

    # write/refresh all generated configs (main pipeline, no ablations)
    python scripts/generate_configs.py

    # include ablation methods (direct-batch*/verif16)
    python scripts/generate_configs.py --include-ablations

    # CI drift gate: exit 1 if any generated file is missing or stale
    python scripts/generate_configs.py --check

Generated files carry a ``DO NOT EDIT`` header; edit the catalog and regenerate.
Shared backends (``shared/configs/backends.yaml``) and contexts
(``shared/configs/environments/``) are referenced by name only and are never
written by this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/generate_configs.py) as well
# as a module, without depending on the package being installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qval.registry import add_catalog_arg, set_active  # noqa: E402
from qval.registry import generation as gen  # noqa: E402

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Also emit ablation methods (direct-batch*/verif16).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if any generated config is missing or stale.",
    )
    add_catalog_arg(parser)
    args = parser.parse_args(argv)
    ctx = set_active(args.catalog)
    root = ctx.root

    if args.check:
        drift = gen.check_drift(
            root, include_ablations=args.include_ablations, registry=ctx.registry
        )
        if drift:
            print(f"config drift detected ({len(drift)} file(s)):")
            for item in drift:
                print(f"  {item}")
            print("run scripts/generate_configs.py to refresh.")
            return 1
        print("generated configs are up to date.")
        return 0

    written = gen.write_all(
        root, include_ablations=args.include_ablations, registry=ctx.registry
    )
    print(f"wrote {len(written)} generated config(s) under {root}/configs/prediction/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
