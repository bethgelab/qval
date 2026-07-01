"""Single source of truth for qval experiment atoms and naming.

Public API:

* atoms — :class:`ModelSpec`, :class:`EnvironmentSpec`, :class:`MethodSpec`,
  :class:`Registry`, :class:`Modality`, :class:`Signal`.
* loader / active — :func:`load_catalog`, :func:`load_default`,
  :class:`CatalogContext`, :func:`set_active`, :func:`get_active`,
  :func:`active_registry`, :func:`add_catalog_arg`. Multiple catalogs can live
  under ``catalogs/``; one is *active* per process (the bundled ``qval_benchmark``
  by default). See ``docs/registry.md`` for selecting and authoring catalogs.
* :data:`REGISTRY` / :data:`MODELS` / :data:`ENVIRONMENTS` / :data:`METHODS` —
  the active catalog's concrete view. These are resolved **lazily** through the
  active catalog (module ``__getattr__``), so a ``set_active(...)`` before they
  are first read takes effect.
* naming — path/name derivation (also re-exported here for convenience).
* legacy — artifact-compatibility tables for pre-convention on-disk names.
"""

from __future__ import annotations

from qval.registry import legacy, naming
from qval.registry.active import (
    active_registry,
    get_active,
    set_active,
)
from qval.registry.atoms import (
    EnvironmentSpec,
    MethodSpec,
    Modality,
    ModelSpec,
    Registry,
    Signal,
)
from qval.registry.loader import (
    DEFAULT_CATALOG_PATH,
    SHARED_ROOT,
    CatalogContext,
    add_catalog_arg,
    load_catalog,
    load_default,
)

__all__ = [
    "EnvironmentSpec",
    "MethodSpec",
    "Modality",
    "ModelSpec",
    "Signal",
    "Registry",
    "REGISTRY",
    "MODELS",
    "ENVIRONMENTS",
    "METHODS",
    "CatalogContext",
    "load_catalog",
    "load_default",
    "set_active",
    "get_active",
    "active_registry",
    "add_catalog_arg",
    "DEFAULT_CATALOG_PATH",
    "SHARED_ROOT",
    "naming",
    "legacy",
]


def __getattr__(name: str):
    """Resolve the active catalog's concrete view lazily (PEP 562).

    Deferring ``REGISTRY``/``MODELS``/``ENVIRONMENTS``/``METHODS`` to access time
    (rather than binding them at import) lets a ``set_active(...)`` performed
    before the first read re-point every consumer, including modules that bind
    these via ``from qval.registry import REGISTRY`` at their own import.
    """
    if name == "REGISTRY":
        return active_registry()
    if name in ("MODELS", "ENVIRONMENTS", "METHODS"):
        registry = active_registry()
        return {
            "MODELS": registry.models,
            "ENVIRONMENTS": registry.environments,
            "METHODS": registry.methods,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
