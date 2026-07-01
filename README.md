# QVal

QVal is a **training-free testbed for dense signal functions** (i.e., the
reward, state-value, Q-value, and advantage signals that methods produce for
post-training). Instead of training an agent on a candidate
signal and measuring downstream performance, QVal scores a method directly by
how well its signal correlates with reference Q-values estimated
from a fixed set of evaluation points.

**New here? Start with [docs/getting_started.md](docs/getting_started.md).**

## Setup

**Prerequisites:** Python >= 3.12, [uv](https://docs.astral.sh/uv/)

Clone [llenvs](https://github.com/serhez/llenvs) alongside this repo so the path
dependency resolves:

```
parent/
  llenvs/        # git clone ...
  qval/          # this repo
```

```bash
uv pip install -e ".[dev]" # core only (no adapters/backends)
uv pip install -e ".[all]" # all llenvs adapters and backends (vLLM requires Linux)
```

On macOS, prefix `uv run` commands with `--no-sync` to skip a Linux-only CUDA
wheel.

Model backends are declared in `shared/configs/backends.yaml` (OpenRouter, locally
served vLLM, Hugging Face, embedding backbones, and scripted policies). To serve
a local model with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port 8000 \
    --max-model-len 4096 --gpu-memory-utilization 0.9
```

## Running QVal

Experiments are driven from a single source of truth (the registry catalog at
`catalogs/qval_benchmark/catalog.py`) and flow through three pipeline scripts.

1. **Define** the models / environments / methods in `catalog.py`.
2. **Generate** configs: `python scripts/generate_configs.py` (writes every prediction + GT config).
3. **Collect** a dataset: `scripts/pipeline/collect_dataset.py` → a `Dataset` pickle.
4. **Predict**: `scripts/pipeline/predict.py --config <generated-config>` → per-method prediction JSONs.
5. **Evaluate**: `scripts/pipeline/evaluate.py` with **no arguments** discovers every
   prediction and pairs each GT/eval combination automatically → correlation summaries.
   (Hand-written evaluation configs are an option for custom comparisons — see the docs.)

The full walkthrough (including running on a SLURM cluster) is in
[docs/getting_started.md](docs/getting_started.md).

## Documentation

- [Getting started](docs/getting_started.md) — run the pipeline end-to-end.
- [Extending](docs/extending.md) — add your own method, model, or environment.
- [Reproducing the paper](docs/reproducing_the_paper.md) — the paper's benchmark.
- [The experiment registry](docs/registry.md) — the catalog / atoms / applicability rules.
- [Naming conventions](docs/naming_conventions.md) — the on-disk artifact layout.
- [Configuration](docs/configuration.md) — every config field in detail.
- [Signal types](docs/signal_types.md) — signal types and correlation metrics.
- [Architecture](docs/architecture.md) — module and pipeline overview.
- Environment notes: [TerminalBench](docs/terminal_bench.md).
