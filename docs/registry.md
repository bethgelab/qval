# The experiment registry

`src/qval/registry/` is the single source of truth for the atoms of an
experiment — the models, environments, and methods that combine into runs — and
for the naming conventions that tie their artifacts together on disk. The
pipeline scripts and the paper-figure code derive from it instead of keeping
their own copies of actor lists, dataset ids, method tables, and path formats.

## Modules

- `atoms.py` — the typed dataclasses and enums:
  - `ModelSpec`: an actor. `actor` is the on-disk token (`q35-27-or`, `clip`,
    `scripted`) and the primary key; `display` is the figure label
    (`Qwen3.5 27B`). `backend_base` / `backend_codegen` / `backend_vision` are
    **names referencing `shared/configs/backends.yaml`** (definitions stay there);
    `backend_for(codegen=…, vision=…)` resolves a usage, falling back to the
    base. Capability flags (`supports_vision`, `is_embedding_backbone`,
    `is_pretrained`, `is_gt_only`, `include_in_results`) drive applicability.
  - `EnvironmentSpec`: an environment and its canonical dataset slice — `env`
    token, `dataset_id`, `dataset_path`, vision support, GT actors, the eval
    actors available per modality, and the point counts. Point counts are
    explicit fields because they vary independently along text / vision-LLM /
    vision-non-LLM / GT axes (see `points_for`). It also carries the
    generation defaults emitted into configs (`batch_size`, `discount_factor`,
    `max_history_turns`, `max_aborted_points`, `gt_aggregations`, the GT
    rollout slots `gt_policy_name` / `gt_num_rollouts`) and `method_params` —
    per-method, environment-specific config fields (e.g. the vle-* negative
    goals / baseline prompt), looked up via `params_for(base)`.
  - `MethodSpec`: a dense-signal method — `base` name, `EvalMethodConfig.type`,
    `family`, `display`, the `signals` and `modalities` it applies to, pinned
    actor / embedding-backbone requirements, a `backend_override` (vip/liv run
    a method-supplied backbone regardless of actor), `is_ablation` (excluded
    from generation unless `--include-ablations`), `num_samples`, and
    `extra_fields` (environment-agnostic config fields rendered verbatim).
  - `Modality` (`text`/`vision`) and `Signal` (`qv`/`sv`, with
    `config_signal_type` mapping to `q_value`/`state_value`).
  - `Registry`: the indexed view over a catalog's atoms, with the common
    lookups (`model_by_actor`, `env_by_name`, `env_by_alias`, `method_by_base`,
    `eval_actors`, `gt_actors`). It lives with the atoms it indexes so a catalog
    loaded from any path produces a `Registry` of this class.
- `catalog.py` — the bundled benchmark catalog: the concrete `MODELS`,
  `ENVIRONMENTS`, `METHODS`, its `NAME` (`qval_benchmark`), and the `REGISTRY`
  instance built from them. **This is the file you edit to add or change an
  experiment dimension of the bundled benchmark** (or copy as the template for a
  new catalog — see *Multiple catalogs* below).
- `loader.py` / `active.py` — catalog selection. `load_catalog(path)` loads a
  catalog module from a file (or a directory containing `catalog.py`) and wraps
  it in a `CatalogContext(registry, root, name, path)`; `load_default()` loads
  the bundled catalog at `DEFAULT_CATALOG_PATH`. `active.py` holds the
  process-global *active* catalog (`set_active`, `get_active`, `active_registry`),
  lazily the bundled default. `add_catalog_arg(parser)` adds the shared
  `--catalog` flag.
- `applicability.py` — the validity rules, single-sourcing constraints that
  used to be implicit across the experiment YAMLs and the prediction-discovery
  code:
  - `signals_for(method, env, modality)` — the signals a method produces here,
    after the `supports_sv` gate and the *vision is QV-only for plain-LLM
    methods* rule.
  - `is_applicable(method, model, env, modality)` — the modality gate plus the
    actor-class partition (plain-LLM methods → LLM actors; `self` → vip/liv;
    clip/siglip → vle-*).
  - `build_method_matrix(env, modality)` — `{Signal: (method_base, …)}`,
    unioned over the actor classes the env lists (ablations excluded by
    default; ranking lumped into QV). Used by config generation.
  - `expected_methods_for_actor(actor, modality)` — the **per-actor**,
    env-agnostic `{"qv", "sv", "ranking_qv"}` matrix that `evaluate.py`
    discovery consumes. Ablations are included (they have on-disk files),
    ranking-QV methods are split into `ranking_qv`, and codegen is left
    un-expanded (its per-sample file components are added in the discovery
    layer). `evaluate.py`'s `default_expected_methods` is a thin adapter over it.
- `generation.py` — expands the matrix into config files. `iter_generated_configs`
  yields `GeneratedConfig(path, text, kind, env)` deterministically;
  `write_all(root)` writes them; `check_drift(root)` reports missing/stale
  generator-owned files (CI gate). `resolve_backend` implements the backend
  rule (override → embedding backbone → actor's `backend_for`).
  These rules used to be hand-maintained as the `EXPECTED_*` / actor / method
  constants inside the prediction-discovery code; the discovery layer of
  `scripts/pipeline/evaluate.py` now derives its vocabulary, dataset paths, and
  expected-method matrix from the registry, so there is one place to change them.
- `legacy.py` — artifact-compatibility tables that are **not** derived from the
  catalog: `LEGACY_NAME_REMAP` (pre-convention `method_name` → canonical),
  `LEGACY_PINNED_EMBEDDING` (the `vle_*_clip` / `vle_*_siglip` names whose
  backbone is baked into the method name), and `CODEGEN_SAMPLE_RE` (matches
  canonical *and* legacy codegen-sample names). The catalog describes the
  current matrix; these describe old files on disk that must keep resolving in
  the paper-figure loader. Delete entries as the files are renamed/removed.
- `naming.py` — pure functions deriving every path and name from the atoms:
  - Evaluation-output dirs: `format_evaluation_output_dir_name`,
    `parse_evaluation_output_dir_name`, `latest_canonical_summary_paths`,
    `parse_summary_filename` (consumed by the viz layer).
  - Prediction dirs: `gt_dir_name`, `eval_dir_name`, `parse_prediction_dir_name`.
  - File names: `prediction_filename`, `codegen_sidecar_filename`,
    `gt_method_filename`, `log_filename`, and `eval_method_name`.
  - The actor / GT-actor vocabularies (`EVAL_ACTORS`, `GT_ACTORS`,
    `ALL_ACTORS`) are *derived* from the active catalog (resolved lazily, so a
    `set_active(...)` before first access re-points parsing); `MODALITIES` is
    catalog-independent.

## Multiple catalogs

A *catalog* is a Python module defining the experiment matrix — either a
`Registry` exported as `REGISTRY`, or the bare `MODELS` / `ENVIRONMENTS` /
`METHODS` tuples the loader assembles into one (plus an optional `NAME`). The
bundled `qval_benchmark` catalog is one such module; you can author your own —
e.g. to evaluate different supervision methods on the same environments — and
select it per run.

The catalog-derived state is read through the **active catalog**, a process
global set once per run and otherwise lazily the bundled default:

- The package attributes `qval.registry.REGISTRY` / `MODELS` /
  `ENVIRONMENTS` / `METHODS` resolve through the active catalog at access time.
- `naming`, `applicability`, and `generation` all read the active catalog
  (their `registry=` arguments still let you pass one explicitly).
- The catalog-aware scripts take `--catalog <path>` (a `.py` file or a directory
  containing `catalog.py`); omitting it uses the bundled default:

  ```bash
  python scripts/generate_configs.py --catalog path/to/catalog.py
  python scripts/pipeline/evaluate.py --catalog path/to/catalog.py
  ```

  `predict.py` / `collect_dataset.py` take explicit config / dataset paths and
  do not need `--catalog`. The paper-figure scripts always render the bundled
  catalog.

## The paper-figure layer

`scripts/paper_figures/registry.py` and `scripts/paper_figures/domain.py` derive
their vocabularies from the catalog instead of keeping hand-synced copies:

- `registry.py` builds `METHOD_REGISTRY` (`method_name` → display/family/pinned),
  `MODEL_REGISTRY` (actor token → display), `ENV_REGISTRY`
  (`experiment_info.env_name` → display), and `V_VALUE_REAL_NAMES` from
  `MethodSpec` / `ModelSpec` / `EnvironmentSpec`. `resolve_method` consults
  `legacy.py` for pre-convention names; its signature is unchanged. The only
  figure-specific addition is codegen's `codegen-avg` mean-of-samples row.
- `domain.py` derives `MODELS`, `NON_LLM_MODELS`, `TEXT_AND_VISION_METHODS`,
  `VISION_ONLY_METHODS`, `PRIMARY_DATASET_IDS`, `PRIMARY_GT_ACTORS`, and
  `GT_ACTOR_DISPLAY_NAMES` from the catalog. What stays hand-written is
  *presentation* (figure-axis order in `MODELS` / `ENVIRONMENTS` / `FAMILIES`)
  and *paper-figure choices* that aren't experiment atoms (`MULTIMODAL_METHODS`,
  the `qv_mc_max` / `sv_mc_max` GT block names, the ablation path tables).

The parity tests in `tests/test_paper_figures_registry.py` assert the
presentation-ordered tables remain an exact cover of the catalog, so the order
can be a figure choice while membership can never silently drift.

## Relationship to `docs/naming_conventions.md`

That document is the human-readable description of the on-disk layout.
`naming.py` is its executable implementation, and reproduces it exactly — the
parity tests in `tests/test_registry_naming_parity.py` round-trip the catalog
and naming functions against the real directories under the bundled catalog's
`catalogs/qval_benchmark/data/predictions/` and `data/evaluations/`.

## Generating experiment configs

`scripts/generate_configs.py` expands the catalog into the experiment YAMLs
under `<catalog>/configs/prediction/<env>/` (the bundled catalog:
`catalogs/qval_benchmark/configs/prediction/<env>/`):

- one prediction config per valid `(env, actor, modality)` at
  `<points>pt_<dataset_id>_<actor>_<modality>.yaml`, and
- one GT config per `(env, gt_actor)` at `<gt_points>pt_<dataset_id>_<gt_actor>_gt.yaml`.

Every `predictions_dir` is derived from `naming.py`, so `predict.py` writes its
output (eval/gt artifacts, rollouts, logs) into the canonical location with no
hand-typed path. Generated files carry a `# GENERATED … DO NOT EDIT` header.

```bash
python scripts/generate_configs.py                    # main pipeline (no ablations)
python scripts/generate_configs.py --include-ablations # also direct-batch*/verif16
python scripts/generate_configs.py --check             # CI: exit 1 on drift
python scripts/generate_configs.py --catalog path/to/catalog.py  # a non-default catalog
```

The generator only *references* backends (`shared/configs/backends.yaml`) and contexts
(`shared/configs/environments/`) by name — it never creates or edits either. Backend
names and `context_name` references must resolve there; the tests guard the
context references, and a missing backend is a runtime error you'll catch on
first run.

## Editing workflow

1. Edit `catalog.py` — add/adjust a `ModelSpec` / `EnvironmentSpec` /
   `MethodSpec`, a capability flag, a backend reference, or an
   environment-specific `method_params` entry.
2. Run the tests:

   ```bash
   uv run --no-sync python -m pytest \
     tests/test_registry_naming_parity.py \
     tests/test_applicability.py tests/test_generate_configs.py -q
   ```

3. Regenerate and commit the config diff:

   ```bash
   python scripts/generate_configs.py
   ```
