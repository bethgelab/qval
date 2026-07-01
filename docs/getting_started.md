# Getting started with QVal

QVal is a **training-free testbed for dense signal functions**. A *dense signal
function* takes a `(state, action, next_state)` step from an agent's trajectory
and returns a scalar — a reward, a state-value, a Q-value, or an advantage.
Lots of methods produce such signals (LLMs prompted for a number, LLMs that
write a reward function as code, embedding-similarity scorers, intrinsic
confidence, self-distillation, …), but they are usually compared only by
training an agent on the signal and looking at downstream performance — which is
expensive and confounds the signal's quality with the training recipe.

QVal compares them directly instead: it scores a method by **how well its signal
correlates with reference-policy ground-truth values**, estimated by Monte-Carlo
rollouts from a fixed set of evaluation points. No agent training, no downstream
runs. See [`signal_types.md`](signal_types.md) for the precise definitions of
each signal type and the correlation metrics.

This guide takes you from a fresh checkout to a table of correlations. To bring
your own method, model, or environment, read [`extending.md`](extending.md)
afterwards. To re-run the exact benchmark from the paper, see
[`reproducing_the_paper.md`](reproducing_the_paper.md).

> Throughout, the project is called **QVal** in prose, but the repository,
> the Python package (`qval`), and all paths/commands keep the
> `qval` / `qval` names.

## 1. Install

**Prerequisites:** Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

QVal drives third-party RL environments through the companion library
[`llenvs`](https://github.com/serhez/llenvs). Clone it next to this repo so the
path dependency resolves:

```
parent/
  llenvs/        # git clone …
  qval/   # this repo
```

```bash
# Core only (no environment adapters / model backends)
uv pip install -e ".[dev]"

# Everything: all llenvs adapters and backends (vLLM requires Linux)
uv pip install -e ".[all]"
```

Run the test suite to confirm the install:

```bash
uv run python -m pytest tests/ -q
```

> **macOS:** use `uv run --no-sync …` for every `uv run` command — a plain
> `uv run` tries to sync a Linux-only CUDA wheel and fails.

## 2. The mental model

Everything starts from one Python file — the **registry catalog**
(`catalogs/qval_benchmark/catalog.py`) — which lists the models, environments,
and methods you want to run. A generator expands that catalog into config files;
three pipeline scripts then turn configs into results, with clean data
boundaries between them:

```
  catalogs/qval_benchmark/catalog.py        ← the single source of truth
                 │   scripts/generate_configs.py
                 ▼
  catalogs/qval_benchmark/configs/prediction/<env>/*.yaml   (generated: eval + GT configs)

  [1] collect_dataset.py  +  a collection config (hand-written)
                 └──────────────►  shared/data/datasets/<env>/<dataset>.pkl   (a Dataset)
                                              │
  [2] predict.py  +  a generated prediction/GT config  ──┘
                 └──────────────►  catalogs/qval_benchmark/data/predictions/<env>/{EVAL_*,GT_*}/   (per-method JSONs)
                                              │
  [3] evaluate.py  ───────────────────────────┘
                 └──────────────►  catalogs/qval_benchmark/data/evaluations/<env>/…/summary_*.json   (correlations)
```

The three data boundaries are: a **`Dataset` pickle** (evaluation points +
trajectory returns + metadata) → **per-method prediction JSONs** → a **summary
JSON** of correlations. Each script reads one boundary and writes the next, so
you can re-run any stage independently.

## 3. Step 0 — Define what you want to run (`catalog.py`)

`catalogs/qval_benchmark/catalog.py` holds three tuples:

- `MODELS` — the actors. Each `ModelSpec` ties together an `actor` token (used in
  output paths), a `display` label, and the `backends.yaml` entries that serve
  it.
- `ENVIRONMENTS` — each `EnvironmentSpec` names an environment, its dataset
  slice, point counts, ground-truth actors, and which environment-context YAML
  to use.
- `METHODS` — each `MethodSpec` is a dense-signal method: which signal types and
  modalities it applies to, which implementation `type` builds it, and any extra
  config fields.

The catalog ships **pre-populated** with a complete, working set of models,
environments, and methods — treat it as a worked example. To run a smaller
experiment, comment out the entries you don't need; to add your own, append new
specs alongside the existing ones (that's what [`extending.md`](extending.md)
covers). The full field-by-field reference is in [`registry.md`](registry.md).

You don't run anything from here directly — editing the catalog changes *what
gets generated* in the next step.

## 4. Step 1 — Generate configs

```bash
python scripts/generate_configs.py            # write/refresh all generated configs
python scripts/generate_configs.py --check    # don't write; exit 1 if anything is stale
python scripts/generate_configs.py --include-ablations   # also emit ablation methods
```

This expands the catalog into one **prediction config** per valid
`(env, actor, modality)` and one **GT config** per `(env, gt_actor)`, written to:

```
catalogs/qval_benchmark/configs/prediction/<env>/<points>pt_<dataset_id>_<actor>_<modality>.yaml
catalogs/qval_benchmark/configs/prediction/<env>/<points>pt_<dataset_id>_<gt_actor>_gt.yaml
```

Every generated file carries a `# GENERATED … DO NOT EDIT` header and a
`predictions_dir` derived from the naming conventions, so the pipeline output
lands in the canonical place with no hand-typed path. **Don't edit generated
files** — change `catalog.py` and regenerate. `--check` is the drift gate: it
fails if a generated file is missing or out of date.

The generator only **references** `shared/configs/backends.yaml` and
`shared/configs/environments/` by name; it never writes them. (Collection, evaluation,
and test-time-scaling configs are also hand-written, not generated.)

## 5. Step 2 — Make sure your backends exist (`backends.yaml`)

`shared/configs/backends.yaml` is the hand-maintained registry of model backends. Each
`ModelSpec` in the catalog refers to backend entries by **name only**
(`backend_base`, `backend_codegen`, `backend_vision`), and `backend_for()`
resolves which one a given run uses. The three-way identity is:

```
actor token (catalog)  →  backend name (backends.yaml)  →  display label (catalog)
   q35-27-or                qwen35_27_thinking_text_or        Qwen3.5 27B
```

An entry looks like this (an OpenRouter-hosted model and a locally-served vLLM
model):

```yaml
backends:
    # OpenRouter-hosted. Name pattern: {model}_{reasoning}[_codegen]_{modality}_{type}
    qwen35_9_thinking_text_or:
        type: openrouter
        model: qwen/qwen3.5-9b
        max_model_len: 262144
        enable_thinking: true
        sampling:
            temperature: 0.1
            top_p: 1.0
            max_tokens: 4096

    # Locally served with vLLM (under Singularity).
    qwen35_9_react_vllm:
        type: vllm_singularity
        model: Qwen/Qwen3.5-9B
        tensor_parallel_size: 1
        gpu_memory_utilization: 0.8
        max_model_len: 32768
        enable_thinking: false
        sampling:
            temperature: 1.0
            top_p: 0.95
            max_tokens: 8192
```

Backend `type`s in use: `openrouter` (hosted API), `vllm_singularity` (locally
served vLLM), `huggingface_vle` (embedding backbones for the `vle-*` methods),
`vip` / `liv` (pre-trained vision scorers), and `codex` (the Codex ground-truth
actor). Scripted ground-truth policies are not backends — they live in
`src/qval/optimal_policies` and are selected by `gt_policy_name`. You edit
this file when you add a model or provider, retune sampling, or change vLLM
tensor-parallelism / memory. A backend name referenced by the catalog but
missing here is a runtime error on the first run.

To serve a local model with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port 8000 \
    --max-model-len 4096 --gpu-memory-utilization 0.9
```

## 6. Step 3 — Collect a dataset

`predict.py` needs a `Dataset`: a fixed set of evaluation points (state-action
pairs sampled from agent trajectories) plus per-trajectory returns and metadata.
For the shipped environments the catalog's `dataset_path` already points at a
collected slice. To build your own, run the collection script against a
**collection config** (hand-written, under
`shared/configs/collection/<env>/`):

```bash
python scripts/pipeline/collect_dataset.py --config <collection-config.yaml>
```

| Flag | Meaning |
|------|---------|
| `--config` | collection config (required) |
| `--output-dir` | override the output directory |
| `--tensor-parallel-size` | tensor-parallel size for vLLM backends |
| `--fresh` | ignore and remove any resume checkpoint |

The output is a `Dataset` pickle at the config's `dataset_path` (e.g.
`shared/data/datasets/<env>/<dataset>.pkl`). Collection configs are not generated —
see [`extending.md`](extending.md) and [`configuration.md`](configuration.md)
for how to write one.

## 7. Step 4 — Predict

`predict.py` consumes a **generated** prediction or GT config, runs the methods
(LLM calls, code execution, embedding scoring) and/or the Monte-Carlo ground
truth, and writes one JSON per method:

```bash
python scripts/pipeline/predict.py \
    --config catalogs/qval_benchmark/configs/prediction/frozen_lake/100pt_8x8_q35-27-or_text.yaml
```

| Flag | Meaning |
|------|---------|
| `--config` | prediction/GT config (required) |
| `--dataset` | override the dataset pickle path |
| `--output-dir` | override `predictions_dir` |
| `--tensor-parallel-size` | tensor-parallel size for vLLM backends |
| `--max-eval-points` | use at most the first N points |
| `--max-aborted-points` | abort if the backend fails more than this fraction of points |

Which methods run is set by the config's `eval_methods` (or `estimations`, for
GT configs) — there is no `--methods` flag. Outputs land under the
`predictions_dir` from the config:

```
catalogs/qval_benchmark/data/predictions/<env>/EVAL_<points>pt_<dataset_id>_<actor>_<modality>/eval/<method>_<qv|sv>_<ts>.json
catalogs/qval_benchmark/data/predictions/<env>/GT_<points>pt_<dataset_id>_<gt_actor>/gt/<gt-method>_<ts>.json
```

Run the GT config(s) for an environment and the eval configs you care about; an
evaluation step pairs them up next. The output layout is documented in
[`naming_conventions.md`](naming_conventions.md).

## 8. Step 5 — Evaluate

Evaluation computes correlations between a GT prediction and the eval
predictions paired with it. `evaluate.py` has two modes.

**Discovery (default).** Run it with no single-evaluation arguments and it walks
`catalogs/qval_benchmark/data/predictions/`, pairs every GT/eval combination using the registry's
applicability rules, and evaluates each one in-process — one summary per
combination, no hand-written config:

```bash
python scripts/pipeline/evaluate.py
# preview the discovered jobs without running them:
python scripts/pipeline/evaluate.py --dry-run
```

It discovers **all** environments under `--predictions-root` (default
`catalogs/qval_benchmark/data/predictions`) and writes the canonical evaluation
tree under `--output-root` (default `catalogs/qval_benchmark/data/evaluations`):

```
catalogs/qval_benchmark/data/evaluations/<env>/<dataset_id>_<gt_actor>_<eval_actor>_<modality>/summary_<ts>.json
```

**Single config.** Pass `--config` (or the `--dataset` / `--predictions-dir`
triple) to run one explicit evaluation that names the prediction sources and
`(gt, eval)` comparisons:

```bash
python scripts/pipeline/evaluate.py --config <evaluation-config.yaml>
```

| Flag | Meaning |
|------|---------|
| `--config` | run a single explicit evaluation from this config |
| `--dataset` / `--predictions-dir` / `--output-dir` | single-evaluation inputs without a config |
| `--predictions-root` / `--output-root` | discovery roots (default `catalogs/qval_benchmark/data/predictions` → `…/data/evaluations`) |
| `--dry-run` | print discovered jobs without evaluating |
| `--missing-report` | write a JSON report of missing/uncovered combinations |

The summary JSON holds the correlation of each method's signal against the
ground truth, per metric (Pearson, Spearman, sign agreement).

## 9. Running on a cluster (SLURM)

The pipeline scripts are plain Python — run them locally as above for small jobs.
For full runs there are SLURM wrappers under `scripts/slurm/` that submit one
config per job:

```bash
sbatch scripts/slurm/run_collection.sh  <collection-config.yaml>
sbatch scripts/slurm/run_prediction.sh  <prediction-config.yaml>
sbatch scripts/slurm/run_evaluation.sh  <evaluation-config.yaml>
```

Each wrapper sources `setup_base.sh` (environment, caches, credentials) and —
for GPU jobs — `setup_gpu.sh`, which exports `NUM_GPUS` from
`$SLURM_GPUS_ON_NODE` and passes it through as `--tensor-parallel-size`.

> These scripts are **examples tied to one cluster**: SBATCH partitions, the
> Python virtualenv, Hugging Face cache locations, and model paths are all
> site-specific. Copy and adapt them to your scheduler before use.

## 10. Where to go next

- [`extending.md`](extending.md) — add your own method, model, or environment.
- [`reproducing_the_paper.md`](reproducing_the_paper.md) — re-run the paper's
  benchmark.
- [`registry.md`](registry.md) — the catalog/atoms/applicability reference.
- [`configuration.md`](configuration.md) — every config field in detail.
- [`signal_types.md`](signal_types.md) — signal types and correlation metrics.
- [`naming_conventions.md`](naming_conventions.md) — the on-disk artifact layout.
