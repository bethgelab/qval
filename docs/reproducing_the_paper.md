# Reproducing the paper's benchmark

The QVal paper instantiates the testbed with a specific set of environments,
models, and dense-signal methods. That exact configuration is what the shipped
registry catalog (`catalogs/qval_benchmark/catalog.py`) encodes — so
**reproducing the paper is mostly running the pipeline against the default
catalog**, with no edits.

This document names the specific lineup and gives the end-to-end steps. For the
mechanics of each command, see [`getting_started.md`](getting_started.md); for
the catalog fields, [`registry.md`](registry.md).

## What's in the benchmark

The default catalog covers:

- **Environments** — ALFWorld (embodied household tasks), FrozenLake (grid
  navigation), OpenApps (computer use), and TerminalBench (command-line agent
  tasks). All four run in text; the first three also run in vision. TerminalBench
  is text-only and Q-value-only.
- **Models** — the Gemma4 (26B-A4B, 31B) and Qwen3.5 (9B, 27B, 35B-A3B,
  122B-A10B) families as evaluation actors, plus the embedding backbones (CLIP,
  SigLIP) and pinned pre-trained backbones used by the vision methods.
- **Methods** — the Direct and Code families (prompted-for-a-number and
  code-generating LLMs), Intrinsic signals, Ranking, Self-Distillation, the
  pre-trained vision scorers (`vip`, `liv-*`), and the embedding scorers
  (`vle-*`). The Q-value signal is the primary one; state-value is reported where
  available.
- **Ground truth** — scripted optimal policies for ALFWorld, FrozenLake, and
  OpenApps; strong-agent rollouts (Codex 5.5, DeepSeek v3.2, Opus 4.7) for
  TerminalBench.

The applicability rules in the catalog encode which method × actor × modality ×
signal combinations are valid, so the generator emits exactly the benchmark
matrix.

## Steps

### 0. Install with adapters

Follow [`getting_started.md`](getting_started.md) §1, using the full extra so all
environment adapters and backends are present:

```bash
uv pip install -e ".[all]"
```

### 1. Obtain the datasets

The benchmark evaluates against fixed `Dataset` slices — the pickles the
catalog's `dataset_path` fields point at, under `shared/data/datasets/<env>/`.

Download them from the release bundle on cloud storage (see the link in the
release notes / repository landing page) and place each pickle at the
`dataset_path` its `EnvironmentSpec` expects. With the files in place, the rest
of the pipeline runs unchanged.

Alternatively, regenerate them from scratch by running the per-environment
collection configs under `shared/configs/collection/<env>/` (see
[`extending.md`](extending.md) §C); they write to the same `dataset_path`.

### 2. Generate configs

```bash
python scripts/generate_configs.py
```

This writes the full benchmark's prediction and GT configs under
`catalogs/qval_benchmark/configs/prediction/<env>/`. Verify there is no drift:

```bash
python scripts/generate_configs.py --check
```

### 3. Run predictions

Submit each generated config. On a cluster (adapt the SLURM wrappers to your
site first — see [`getting_started.md`](getting_started.md) §9):

```bash
for cfg in catalogs/qval_benchmark/configs/prediction/*/*.yaml; do
  sbatch scripts/slurm/run_prediction.sh "$cfg"
done
```

Run the GT configs (`*_gt.yaml`) as well as the eval configs — both are under the
same `prediction/` directory. Outputs land in the canonical
`catalogs/qval_benchmark/data/predictions/<env>/{GT_*,EVAL_*}/` tree.

### 4. Evaluate

Let the discovery evaluator pair every GT/eval combination and write the
canonical evaluation tree:

```bash
python scripts/pipeline/evaluate.py
```

Summaries appear at
`catalogs/qval_benchmark/data/evaluations/<env>/<dataset_id>_<gt_actor>_<eval_actor>_<modality>/summary_<ts>.json`.

### 5. Build the figures

The paper figures and tables are produced from the evaluation summaries by the
scripts under `scripts/paper_figures/`, which derive their model/method/
environment vocabularies from the same catalog.

## Running a subset

To reproduce only part of the benchmark (one environment, a few models, one
method family), trim the catalog to the entries you want and regenerate — the
generator emits only what the catalog lists. This is the same edit-and-regenerate
loop described in [`extending.md`](extending.md); the difference is you are
removing entries rather than adding them.
