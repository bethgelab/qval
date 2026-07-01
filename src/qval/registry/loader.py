"""Load an experiment catalog from a path and wrap it in a :class:`CatalogContext`.

A *catalog* is a Python module that defines the experiment matrix — either a
ready :data:`Registry` exported as ``REGISTRY`` or the bare ``MODELS`` /
``ENVIRONMENTS`` / ``METHODS`` tuples the loader assembles into one. The bundled
benchmark catalog ships at :data:`DEFAULT_CATALOG_PATH`; users select their own
with ``--catalog`` (see :func:`add_catalog_arg`).

The catalog is loaded by file path (``importlib.util.spec_from_file_location``,
the same idiom ``tests/test_evaluate_discovery.py`` uses) rather than imported as
a package submodule, so a catalog may live anywhere on disk — including outside
the ``qval`` package. Because :class:`Registry` and the atom types are
imported absolutely from :mod:`qval.registry.atoms`, a path-loaded catalog
produces atoms of the same classes as the rest of the package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from qval.registry.atoms import Registry

# The bundled benchmark catalog, shipped under ``catalogs/qval_benchmark/`` at the
# repo root (outside the package, so it is loaded by path like any user catalog).
DEFAULT_CATALOG_PATH: Path = (
    Path(__file__).resolve().parents[3] / "catalogs" / "qval_benchmark" / "catalog.py"
)

# Repo-relative anchor for cross-catalog *shared inputs* (datasets, env contexts,
# backends, …). Defined here as the single source for that resolution; consumers
# join it for ``shared/``-prefixed literals.
SHARED_ROOT: Path = Path(__file__).resolve().parents[3] / "shared"


@dataclass(frozen=True)
class CatalogContext:
    """A loaded catalog: its :class:`Registry` plus on-disk identity.

    * ``registry`` — the indexed view over the catalog's atoms.
    * ``root`` — absolute directory the catalog lives in (the parent of
      ``catalog.py``). Per-catalog outputs are resolved against this.
    * ``name`` — the catalog's ``NAME`` export, else ``root.name``.
    * ``path`` — absolute path to the loaded catalog file.
    """

    registry: Registry
    root: Path
    name: str
    path: Path


def _module_name_for(path: Path) -> str:
    """A stable, collision-free module name for a path-loaded catalog."""
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:12]
    return f"qval.registry._loaded_catalog_{digest}"


def _load_module_from_path(catalog_file: Path) -> ModuleType:
    module_name = _module_name_for(catalog_file)
    spec = importlib.util.spec_from_file_location(module_name, catalog_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load catalog module from {catalog_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _extract_registry(module: ModuleType, catalog_file: Path) -> Registry:
    registry = getattr(module, "REGISTRY", None)
    if isinstance(registry, Registry):
        return registry
    try:
        models = module.MODELS
        environments = module.ENVIRONMENTS
        methods = module.METHODS
    except AttributeError as exc:
        raise ValueError(
            f"catalog {catalog_file} must define a Registry as REGISTRY, or the "
            "MODELS / ENVIRONMENTS / METHODS tuples."
        ) from exc
    return Registry(models, environments, methods)


def load_catalog(path: str | Path) -> CatalogContext:
    """Load the catalog at ``path``.

    ``path`` may be a ``.py`` file or a directory containing ``catalog.py``.
    Raises :class:`FileNotFoundError` if the file is absent and
    :class:`ValueError` if it defines neither a ``REGISTRY`` nor the
    ``MODELS``/``ENVIRONMENTS``/``METHODS`` tuples.
    """
    resolved = Path(path).resolve()
    if resolved.is_dir():
        root = resolved
        catalog_file = resolved / "catalog.py"
    else:
        catalog_file = resolved
        root = resolved.parent
    if not catalog_file.is_file():
        raise FileNotFoundError(f"catalog file not found: {catalog_file}")

    module = _load_module_from_path(catalog_file)
    registry = _extract_registry(module, catalog_file)
    name = getattr(module, "NAME", None) or root.name
    return CatalogContext(registry=registry, root=root, name=name, path=catalog_file)


def load_default() -> CatalogContext:
    """Load the bundled benchmark catalog (:data:`DEFAULT_CATALOG_PATH`)."""
    return load_catalog(DEFAULT_CATALOG_PATH)


def add_catalog_arg(
    parser: argparse.ArgumentParser, *, flag: str = "--catalog"
) -> argparse.ArgumentParser:
    """Add the standard ``--catalog`` selection flag to ``parser``."""
    parser.add_argument(
        flag,
        default=None,
        help=(
            "Path to a catalog .py file (or a directory containing catalog.py). "
            "Defaults to the bundled qval_benchmark catalog."
        ),
    )
    return parser
