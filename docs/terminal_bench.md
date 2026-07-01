# Running TerminalBench Experiments

Step-by-step guide for running TerminalBench experiments with qval. Covers two deployment scenarios: HPC clusters with Singularity/Apptainer, and local machines with Docker.

## Prerequisites (both scenarios)

1. **qval installed** with dev dependencies:
   ```bash
   uv pip install -e ".[all]"
   ```

2. **llenvs installed** as a local path dependency (already configured in `pyproject.toml`).

3. **Harbor Python package** installed:
   ```bash
   pip install harbor
   ```
   Harbor provides the task registry, task definitions, and verifier scripts. The container runtime is separate.

4. **LLM backends** for the agent and evaluation, declared in `shared/configs/backends.yaml`. TerminalBench trajectories are collected with a strong actor (e.g. `deepseek_v32_react`); predictions are produced by the eval backends the catalog references (e.g. `qwen35_9_thinking_text_or`), and ground truth by the GT actors (`codex`, `opus-47`, `ds32`). OpenRouter-hosted `*_or` entries need no local GPU; configure whatever matches your available hardware.

## Scenario A: HPC cluster with Singularity/Apptainer

For clusters where GPU nodes only have `singularity` or `apptainer` (no Docker or Podman).

### A1. Identify your runtime

Check which binary is available on your GPU nodes:

```bash
singularity --version   # or: apptainer --version
```

Check if fakeroot works (recommended for the overlay fast path):

```bash
singularity instance start --fakeroot --writable-tmpfs \
    library://alpine:latest test-fakeroot
singularity instance stop test-fakeroot
```

If fakeroot works, keep `fakeroot: true` in the config. On clusters where the overlay path is unreliable, set `rootfs_mode: sandbox` so Apptainer skips the overlay probe and goes straight to writable sandbox copies.

The apptainer runtime now has two internal execution paths:
- Overlay mode keeps `/app` and `/tests` writable via host-backed binds and probes whether the container root itself is writable.
- Sandbox mode uses a writable per-trial sandbox copy of the SIF when the overlay path is insufficient.

Agent and verifier commands default to the image workdir, which is typically `/app` for TerminalBench tasks.

### A2. Configure the environment context

Create an environment context with apptainer-hpc settings to match your cluster, for example:

```yaml
make_kwargs:
  max_steps: 10
  exec_timeout: 180
  command_soft_timeout: 60
  trajectory_timeout: 900
  verify_on_truncation: true
  text_exec_mode: tmux_session
  tmux_bootstrap_if_missing: true
  environment_type: apptainer-hpc
  apptainer_command: singularity    # or "apptainer", whichever your cluster has
  fakeroot: true                    # set to false if fakeroot is unavailable
  sif_cache_dir: .sif_cache         # path to your SIF cache directory
  rootfs_mode: sandbox              # recommended when overlay is known not to work
```

An apptainer-hpc collection can also keep `runtime_probing: true`, discard Harbor trajectories flagged as replay-risky, and enable verbose collection debug logs:

```yaml
collection:
  debug: true
  runtime_probing: true
  discard_runtime_probe_risky_trajectories: true
```

With `debug: true`, the collector raises `qval` and `llenvs` package
loggers to DEBUG so the logs show round boundaries, batch generation,
environment step waits, Harbor reset/step timing, runtime probes, and verifier
execution.

The provided TerminalBench collection configs keep replay validation disabled.
For now, normal replay-safety filtering comes from `runtime_probing: true`
combined with `discard_runtime_probe_risky_trajectories: true`. A stronger
live-vs-replay audit can be added later if needed, but it is not part of the
default collection path, and collection does not run any post-collection replay
probe capture while replay validation stays disabled.

If `fakeroot: false`, keep `rootfs_mode: sandbox`. `writable_tmpfs` is only relevant if you later switch back to overlay mode on a cluster where that path works:

```yaml
  fakeroot: false
  rootfs_mode: sandbox
  writable_tmpfs: true
```

### A3. Stage SIF images (login node, once)

TerminalBench tasks ship as Docker images. GPU nodes can't pull or build Docker images, so you must pre-convert them to SIF format on a node that has network access and `singularity`/`apptainer`.

```bash
# Preview what will be built (no changes)
uv run python scripts/stage_sif_images.py \
    --env-name "terminal-bench@2.0" \
    --sif-cache-dir .sif_cache \
    --apptainer-command singularity \
    --dry-run

# Build all images
uv run python scripts/stage_sif_images.py \
    --env-name "terminal-bench@2.0" \
    --sif-cache-dir .sif_cache \
    --apptainer-command singularity

# With curated package preinstalls (optional):
uv run python scripts/stage_sif_images.py \
    --env-name "terminal-bench@2.0" \
    --sif-cache-dir .sif_cache \
    --apptainer-command singularity \
    --preinstall-config shared/configs/image_preinstalls/terminal_bench.yaml
```

This:
- Loads the TerminalBench task registry via Harbor
- Skips compose-backed (multi-container) tasks automatically
- Converts each task's Docker image to a `.sif` file in the cache directory
- When `--preinstall-config` is provided, applies curated package preinstalls via a two-stage build (`Bootstrap: localimage` + `%post`) for matching images
- The TerminalBench preinstall config supports a global `all:` rule, which is useful for baking shared prerequisites such as `tmux` and `r-base` into every image while keeping task-specific package overrides separate
- Writes a `manifest.json` mapping image references to SIF filenames
- Is idempotent — rerunning skips already-cached images (use `--force` to rebuild). Preinstall changes are detected automatically via `preinstall_hashes.json`

If `sif_cache_dir` is on a shared filesystem (e.g., NFS, Lustre), you only need to run this once. GPU nodes will read from the same path.

### A4. Preflight check (GPU node, optional but recommended)

Before committing to a full experiment, verify the runtime works on a GPU node:

```bash
SIF=$(ls .sif_cache/*.sif | head -1)

# Basic instance lifecycle
singularity instance start --fakeroot --cleanenv --contain --no-home "$SIF" test-inst
singularity exec --cleanenv instance://test-inst bash -lc "echo hello"
singularity exec --cleanenv --pwd /tmp instance://test-inst bash -lc "pwd"
singularity instance stop test-inst

# If using disk overlays (overlay fast path):
singularity overlay create --size 256 /tmp/test-overlay.img
singularity instance start --fakeroot --overlay /tmp/test-overlay.img \
    --cleanenv --contain --no-home "$SIF" test-inst2
singularity exec --cleanenv instance://test-inst2 bash -lc "touch /testfile && ls /testfile"
singularity instance stop test-inst2
rm /tmp/test-overlay.img
```

In `qval`, `apptainer-hpc` first tries the overlay path. If the container root still behaves as read-only on the compute node, it automatically rebuilds that trial as a writable sandbox copy instead.

### A5. Run an experiment

The repo ships example configs for the TBLite-easy dataset: collection under
`shared/configs/collection/terminal_bench/`, prediction/evaluation under
`catalogs/qval_benchmark/configs/{prediction,evaluation}/terminal_bench/`. The
three pipeline steps:

```bash
uv run python scripts/pipeline/collect_dataset.py \
    --config shared/configs/collection/terminal_bench/200pt_40ms_ds32.yaml

uv run python scripts/pipeline/predict.py \
    --config catalogs/qval_benchmark/configs/prediction/terminal_bench/100pt_tblite-easy-40ms_q35-9-or_text.yaml

uv run python scripts/pipeline/evaluate.py \
    --config catalogs/qval_benchmark/configs/evaluation/terminal_bench/tblite-easy-40ms_q35-9-or_text.yaml
```

The shipped collection config targets the local TBLite-easy dataset (see the
TBLite section below for image staging); the environment is set by the config's
`env_name`/`context_name`, so `collect_dataset.py` takes no `--env-name` flag. To
collect from the `terminal-bench@2.0` registry instead, write a collection config
whose `env_name` is `terminal-bench@2.0` and create analogous prediction/evaluation
configs for that dataset.

For full experiments, you will want an environment context that matches your cluster with production settings (higher `max_steps`, longer `exec_timeout`). The shipped `tblite_easy_40ms_react_tags.yaml` already sets `environment_type: apptainer-hpc` — copy it and adjust those values rather than starting from scratch.

### Apptainer-HPC limitations

- **Single-container tasks only** — compose-backed tasks (multi-service) are filtered out automatically during collection
- **No network isolation** — tasks with `allow_internet: false` are filtered out (Apptainer uses host networking)
- **Filesystem checkpoint/restore** — in sandbox mode, GT estimation can cache replayed rootfs as tar archives via `harbor_replay_cache: true` in prediction config, reducing N-fold replay to 1 replay + (N-1) tar restores per eval point. Requires `rootfs_mode: sandbox` (or `auto` falling back to sandbox). Process-level runtime probing requires `pid_namespace: true`. This cache only restores filesystem state; it does not restore process state or persistent tmux shell-session state, so keep it disabled for TerminalBench contexts that use `text_exec_mode: tmux_session`. If exporting one cached rootfs fails, Harbor falls back to normal replay for that state instead of aborting the full prediction run
- **No on-node image builds** — SIF images must be pre-staged; Dockerfile-only tasks without a pre-built SIF are filtered out
- **Replay timeout hardening** — replay uses the same `command_soft_timeout` as normal collection (60s per command). If a replayed command times out and recovery succeeds, replay continues (the timeout observation is preserved in the state's message history). Timeout recovery escalates from `Ctrl-C` to `Ctrl-\` to tmux TUI escape attempts (`Esc Esc`, `:qa!`, `q`). Syntactically incomplete commands are cancelled immediately when bash enters the continuation prompt. Replay and restore work do not consume the later live `trajectory_timeout` budget used for the continued rollout
- **Point-local replay aborts** — if replay fails with a classified recoverable environment error for one evaluation point, prediction retains any already completed rollouts for that point, stops scheduling more rollouts for it, emits a `rollout_abort` log event, and continues with the rest of the job. Current recoverable cases include replay/restore timeouts, shell-continuation prompts, Harbor tmux-session death during command execution, and classified Jericho native faults. Ranking collection similarly skips candidates whose restore fails instead of aborting

## TBLite (OpenThoughts-TBLite)

TBLite tasks use the same Harbor adapter and Apptainer runtime as
TerminalBench, but come from a local dataset directory instead of the
Harbor registry. The key difference: TBLite tasks provide Dockerfiles
(not pre-built Docker Hub images), so image staging requires building
Docker images locally first.

### Task distribution

100 tasks across five difficulty levels:

| Difficulty | Count |
|---|---|
| easy | 24 |
| medium | 43 |
| hard | 30 |
| very-hard | 1 |
| expert | 2 |

### Prerequisites

- The TBLite dataset cloned locally on both your laptop and the cluster
  (e.g., `~/dev/OpenThoughts-TBLite`)
- Docker installed on your laptop (for building images)

### T1. Build Docker images (laptop, once)

TBLite tasks ship as Dockerfiles, not pre-built images. Since HPC
clusters don't have Docker, build the images on your laptop and export
them as tar archives.

**Important**: the script builds for `linux/amd64` regardless of host
architecture (e.g., Apple Silicon). This uses QEMU emulation and is
slower than native builds, but produces images that run on x86_64
clusters. Each image is verified after build.

No script ships for this laptop-side step; build the archives directly:

```bash
mkdir -p docker_archives
for task_dir in ~/dev/OpenThoughts-TBLite/*/; do
    task=$(basename "$task_dir")
    [ -f "docker_archives/$task.tar" ] && continue          # idempotent skip
    docker build --platform linux/amd64 -t "tblite/$task" "$task_dir/environment"
    docker save "tblite/$task" -o "docker_archives/$task.tar"
done
```

This iterates over all task directories, runs
`docker build --platform linux/amd64` with each task's `environment/`
directory as build context, and saves each image as
`docker_archives/<task_name>.tar`, skipping tasks that already have a tar.
Expect 30-60 minutes (QEMU cross-compilation) and 20-50 GB depending on
image sizes. (On the cluster, `stage_sif_images.py` can also build directly
from the Dockerfiles when an archive is missing — see T3 — so this pre-build
is an optimization, not a hard requirement.)

If you need to rebuild specific tasks, delete their tar files and
rerun.

### T2. Upload archives to cluster

```bash
scp -r docker_archives/ <user>@<cluster>:~/dev/qval/docker_archives/
```

These are only needed during SIF staging and can be deleted afterwards.

### T3. Stage SIF images (cluster, once)

Convert the tar archives to SIF files and apply preinstalls:

```bash
python scripts/stage_sif_images.py \
    --dataset-path ~/dev/OpenThoughts-TBLite \
    --sif-cache-dir .sif_cache_lite \
    --apptainer-command singularity \
    --preinstall-config shared/configs/image_preinstalls/terminal_bench.yaml \
    --docker-archive-dir docker_archives/ \
    --fakeroot
```

For each task, this reads the tar from `docker_archives/<task_name>.tar`,
builds a base SIF via `singularity build ... docker-archive:<tar>`, then
applies the preinstall layer on top (build-essential, p7zip-full, xxd,
nodejs, etc.). Final SIFs go into `.sif_cache_lite/`.

Add `--dry-run` first to preview what will be built. Use `--force` to
rebuild existing SIFs.

If a tar is missing for a task, the script falls back to building from
the Dockerfile directly (converting it to a Singularity `.def` file).
This works for simple Dockerfiles but may fail for complex ones.

### T4. Configure the environment context

Use a TBLite-specific environment context. Example for easy tasks with
react tags (`shared/configs/environments/tblite_easy_40ms_react_tags.yaml`):

```yaml
adapter: harbor
env_name: "tblite"
prompting_scheme: react_tags

make_kwargs:
    dataset_path: ~/dev/OpenThoughts-TBLite
    difficulties: ["easy"]
    max_steps: 40
    exec_timeout: 180
    command_soft_timeout: 60
    trajectory_timeout: 900
    verify_on_truncation: true
    text_exec_mode: tmux_session
    tmux_bootstrap_if_missing: true
    environment_type: apptainer-hpc
    apptainer_command: singularity
    fakeroot: true
    sif_cache_dir: .sif_cache_lite
    rootfs_mode: sandbox
    pid_namespace: true
```

Key differences from TerminalBench:
- `dataset_path` points to the local TBLite clone (instead of using the
  Harbor registry)
- `difficulties` filters tasks by difficulty level
- `sif_cache_dir` points to `.sif_cache_lite` (where T3 staged the
  images)

### T5. Run collection

```bash
python scripts/pipeline/collect_dataset.py \
    --config shared/configs/collection/terminal_bench/200pt_40ms_ds32.yaml
```

The rest of the pipeline (predict, evaluate) is identical to
TerminalBench — see Scenario A above.

## Scenario B: Local machine with Docker

For development, testing, or machines where Docker is available.

### B1. Verify Docker is running

```bash
docker info
```

### B2. No image staging needed

Docker-mode Harbor pulls images automatically. No SIF conversion or pre-staging required.

### B3. Configure the environment context

No Docker environment context ships by default — the shipped TerminalBench context (`tblite_easy_40ms_react_tags.yaml`) targets apptainer-hpc. For local Docker, create a context that omits `environment_type` (no `environment_type` = Docker via Harbor's built-in provider), for example:

```yaml
make_kwargs:
  max_steps: 20
  exec_timeout: 120
  command_soft_timeout: 60
  verify_on_truncation: true
  text_exec_mode: tmux_session
  tmux_bootstrap_if_missing: true
```

No changes needed for local Docker use.

TerminalBench uses strict action parsing. The standard configs require actor
commands in `<answer>...</answer>`, while the ReAct configs use either
`<thought>...</thought>` plus `<action>...</action>` (`react_tags`) or
`Thought:`/`Action:` lines (`react_classic`). These TerminalBench configs do
not fall back to raw execution: malformed output produces an explicit
invalid-format observation and Harbor does not execute any shell command for
that turn. For TerminalBench, `react_tags` is the preferred ReAct mode because
it keeps multiline shell commands unambiguous.

The Harbor prompt used for TerminalBench also reminds models to avoid
interactive or incomplete shell commands when possible, for example bare
`cat`, Python prompts, `ssh`, editors, or unfinished quotes and heredocs.
The prompt states the maximum number of turns and that failing to
complete the task within that budget is a failure; the turn count is
taken from `make_kwargs.max_steps` and substituted into the prompt
template. When `max_steps` is not provided, the prompt falls back to a
generic "limited number of steps" wording.

### B4. Using podman-hpc locally

If you have Podman installed (but not Docker), use the podman-hpc environment context:

```yaml
# environment context make_kwargs (podman-hpc)
make_kwargs:
  max_steps: 10
  exec_timeout: 45
  verify_on_truncation: true
  text_exec_mode: tmux_session
  tmux_bootstrap_if_missing: true
  environment_type: podman-hpc
  podman_command: podman   # use "podman-hpc" if that's your binary
```

podman-hpc supports everything Docker does (single-container, compose, snapshots) without requiring Docker.

## Troubleshooting

### "SIF image not found"

```
FileNotFoundError: SIF image not found at .sif_cache/abc123.sif.
Pre-build it on a login node: singularity build .sif_cache/abc123.sif docker://ubuntu:22.04
```

Run the staging script (step A3). If you see this for specific tasks, the error message includes the exact `singularity build` command to run manually.

### "Compose-backed tasks are not supported"

The Apptainer runtime does not support multi-container tasks. These are filtered automatically during collection. If you're hitting this error directly, ensure you're using an apptainer-hpc environment context (e.g. `tblite_easy_40ms_react_tags.yaml`), not a Docker one.

### "apptainer command failed" / "singularity command failed"

Check that:
1. The binary name in `apptainer_command` matches what's installed (`singularity` vs `apptainer`)
2. fakeroot is available if `fakeroot: true` is set
3. The SIF file exists and is not corrupted (try `singularity inspect <file>.sif`)

### "network isolation" filtered tasks

Tasks requiring `allow_internet: false` cannot run on Apptainer (no network isolation). These are filtered automatically. This is expected and logged.

### Replay validation failures

```
WARNING: Replay validation discarded N points (max_discards=3)
```

Some tasks produce non-deterministic state on replay (network-dependent commands, timestamps, etc.). Replay validation detects this and discards those evaluation points. Increase `max_discards` if you're losing too many points, or investigate which tasks are failing.

## Config reference

### Environment contexts

| File | Runtime | Use case |
|---|---|---|
| `tblite_easy_40ms_react_tags.yaml` | apptainer-hpc | TBLite easy tasks (local dataset, react tags) — the shipped context |
| *(create your own; omit `environment_type`)* | Docker | Local development, Docker-equipped machines |

### Key `make_kwargs` parameters

| Parameter | Values | Description |
|---|---|---|
| `environment_type` | `apptainer-hpc`, `podman-hpc`, or omit for Docker | Container runtime |
| `apptainer_command` | `apptainer` or `singularity` | Binary name (apptainer-hpc only) |
| `podman_command` | `podman` or `podman-hpc` | Binary name (podman-hpc only) |
| `sif_cache_dir` | path | Pre-built SIF image directory (apptainer-hpc only) |
| `fakeroot` | `true`/`false` | Enable fakeroot for the overlay fast path and writable sandbox startup (apptainer-hpc only) |
| `rootfs_mode` | `auto`, `overlay`, `sandbox` | Overlay fast path with sandbox fallback, overlay-only, or sandbox-only (apptainer-hpc only) |
| `writable_tmpfs` | `true`/`false` | Use in-memory tmpfs instead of a disk overlay for overlay mode (apptainer-hpc only) |
| `overlay_size_mb` | integer | Disk overlay size in MB for overlay mode (apptainer-hpc, default 512) |
| `pid_namespace` | `true`/`false` | Enable `--pid` namespace for container (apptainer-hpc only, needed for process-level runtime probing) |
| `max_steps` | integer | Max commands per episode |
| `exec_timeout` | integer | Per-command timeout in seconds |
| `command_soft_timeout` | integer or omit | Recoverable timeout (seconds) for live model-issued Harbor text commands. On timeout, Harbor escalates through `Ctrl-C`, `Ctrl-\`, and tmux TUI escape attempts (`Esc Esc`, `:qa!`, `q`). If the shell returns to a prompt, Harbor appends a standard timeout observation and keeps the session alive |
| `trajectory_timeout` | integer, `null`, or omit | Live per-trajectory wall-clock budget (seconds) for Harbor text mode. Replay, snapshot restore, replay validation, and replay-cache filesystem restores do not consume this budget; a restored continuation starts with a fresh live budget |
| `verify_on_truncation` | `true`/`false` | Run verifier at max_steps |
| `text_exec_mode` | `tmux_session` or `independent_exec` | Text-mode execution model. TerminalBench contexts now default to `tmux_session` so shell state and background jobs persist across commands |
| `dataset_path` | path or omit | Local dataset directory for Dockerfile-only datasets (e.g., TBLite). When set, tasks are loaded from this directory instead of the Harbor registry |
| `difficulties` | list of strings or omit | Filter tasks by difficulty (e.g., `["easy"]`). Only applies to datasets with difficulty metadata |
| `tmux_bootstrap_if_missing` | `true`/`false` | When using `tmux_session`, attempt a bounded package-manager install of `tmux` inside the task image if it is missing |

When `text_exec_mode: tmux_session` is enabled, `collect_dataset.py` also runs
a one-time Harbor tmux preflight over the selected task indices before model
sampling starts. This catches missing `tmux` or session-startup failures early,
which is especially helpful on Apptainer clusters where task startup is
expensive.

When `command_soft_timeout`, `trajectory_timeout`, and `exec_timeout` are set
together, Harbor treats live TerminalBench command timeouts as recoverable text
observations instead of immediate step errors while still enforcing a separate
live per-trajectory budget. Replay validation probe commands still use the hard
`exec_timeout` path, and replay/restore work does not consume the later live
`trajectory_timeout` budget.
