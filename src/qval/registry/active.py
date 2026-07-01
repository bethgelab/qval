"""The process-global *active catalog*.

Most consumers reach the catalog by attribute or import access, not by a call
site we can thread a ``registry`` argument through (the ``naming`` vocabulary,
the paper-figure axes, the lazy package-level ``REGISTRY``). A single active
catalog — chosen once per process via ``--catalog`` and otherwise lazily the
bundled default — lets all of them agree without plumbing. This mirrors the
registration singletons already used for ``optimal_policies`` / ``action_samplers``.

The active catalog is set at most once per CLI run, before the registry-derived
state is read. Tests that re-point it must snapshot and restore :data:`_active`.
"""

from __future__ import annotations

from qval.registry.atoms import Registry
from qval.registry.loader import CatalogContext, load_catalog, load_default

_active: CatalogContext | None = None


def get_active() -> CatalogContext:
    """The active :class:`CatalogContext`, lazily loading the default."""
    global _active
    if _active is None:
        _active = load_default()
    return _active


def set_active(catalog: CatalogContext | str | None) -> CatalogContext:
    """Set the active catalog and return it.

    ``catalog`` may be an already-loaded :class:`CatalogContext`, a path to a
    catalog file/directory, or ``None`` to (re)select the bundled default.
    """
    global _active
    if isinstance(catalog, CatalogContext):
        _active = catalog
    elif catalog is None:
        _active = load_default()
    else:
        _active = load_catalog(catalog)
    return _active


def active_registry() -> Registry:
    """The :class:`Registry` of the active catalog."""
    return get_active().registry
