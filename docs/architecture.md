# Architecture

qval evaluates methods that produce dense signal functions for RL environments. It computes ground truth signal values via Monte Carlo estimation, then correlates them with predictions from the methods being evaluated.

## Module Overview

```
src/qval/
    types.py                # Core data types (enums, dataclasses, MethodContext)
    config.py               # Configuration dataclasses + YAML loading
    dense_signal.py         # Protocols (DenseSignalFunction) + DenseSignalMethod ABC
    rollout.py              # Single and lockstep batched trajectory rollouts + trajectory_return()
    estimator.py            # aggregate_returns() + create_gt_method() factory
    mc_method.py            # MCMethod: ground truth estimation via batched MC rollouts
    mcts_method.py          # MCTSMethod: ground truth estimation via MCTS with batch expansion
    data_cache.py           # Dataset + pickle save/load for data caching
    prediction_store.py     # Prediction JSON save/load per method
    transforms.py           # Signal transform utilities (sum_per_trajectory, extract_potential_values)
    evaluation_points.py    # Collecting (s, a, s') points from trajectories
    correlation.py          # Pearson/Spearman/Kendall/sign-agreement correlation computation
    benchmark.py            # Utilities: collect_trajectories, prepare_display_points, resolve_backend
    test_time_scaling.py    # GuidedActor + online candidate scoring for test-time scaling
    script_utils.py         # Shared helpers for CLI scripts (adapter/backend factories)
    experiment_logging.py   # ExperimentLogger + LoggingBackend for LLM call capture
    trajectory_store.py     # Observable trajectory types + JSON save/load
    ranking_collection.py   # Ranking point construction (manual + LLM sampling)
    action_samplers.py      # Manual action sampler registry + built-ins (FrozenLake)
    methods/                # Built-in method implementations
        code_parsing.py     # Code extraction, validation, and signature checking
        serialization.py    # State/action → text conversion
        baseline_random.py  # RandomBaselineMethod (uniform [0, 1) correlation floor)
        llm_direct.py       # LLMDirectMethod (per-point numeric LLM output)
        llm_verifier.py     # LLMVerifierMethod (score-bin verifier output)
        llm_codegen.py      # LLMCodeGenMethod (LLM generates Python code once)
        llm_eureka.py       # LLMEurekaMethod (iterative code search with LLM judging)
        llm_ranking.py      # LLMRankingMethod (ranks candidate actions by Q-value)
        vle/                # VLEMethod: frozen CLIP/SigLIP text-image encoders
            encoders.py     #   HFCLIPEncoder via transformers.AutoModel
            heads.py        #   cosine / goal-baseline / softmax / thresholded heads
            method.py       #   VLEMethod + build_vle_method
        vip.py              # VIPMethod: frozen ResNet-50 (Ego4D) image-to-goal-image L2
        liv.py              # LIVMethod: frozen CLIP-RN50 (EpicKitchens) image OR text goal
```

## Pipeline

The experiment workflow is split into independent scripts. An optional pre-step evaluates model competency, followed by three core steps:

### Step 0 (optional): `compute_pass_at_k.py` — Model Competency Evaluation

Evaluates whether a model is competent enough at an environment to serve as an "optimal policy" for ground truth estimation. Runs N independent trajectories per task, groups results by task, and computes Pass@k using the standard combinatorial formula (1 - C(n-c, k) / C(n, k)).

Success semantics: a trajectory counts as a Pass@k success when its undiscounted cumulative return under the environment's `reward_signal_name` meets `success_threshold` (default `1.0`). `step_penalty` is stripped from the sum, so control cost never flips a success into a failure. For environments with dense or score-based rewards, raise `success_threshold` to the per-environment meaningful cutoff.

1. **Collect trajectories** — run the environment with the same prompts/setup as collection, but with each task repeated N times.
2. **Group by task** — associate each trajectory result with its task index.
3. **Compute Pass@k** — per-task Pass@k for each requested k, plus macro-averaged aggregate.
4. **Save report** — JSON output with per-task and aggregate statistics, plus console summary.

Two companion scripts reuse the same analysis module against already-collected artifacts:

- **`compute_pass_at_k_from_dataset.py`** — reads `trajectory_results` from a `Dataset` pickle and emits the same Pass@k report without running any new rollouts. Use this to retroactively assess Pass@k for datasets produced by `collect_dataset.py`.
- **`compute_pass_at_k_from_predictions.py`** — reads the MC rollouts persisted by `predict.py` under `<prediction-dir>/rollouts` and reports Pass@k with two orthogonal groupings: each eval point / ranking candidate as its own task (`by_point`) and each underlying environment task (`by_env_task`). MCTS is out of scope (its rollouts are in-memory only).

All three scripts delegate to `qval.pass_at_k`, which owns the success-threshold logic, attempt adapters, grouping math (macro-averaged aggregate with sample stdev), and console formatting.

### Optional: `test_time_scaling.py` — Guided Online Action Selection

Runs an online experiment that uses QVal methods as action scorers at test time. At each environment step, `GuidedActor` samples `k` candidate actions from the actor backend, builds temporary action-conditioned `EvaluationPoint`s for those candidates, and selects the candidate with the highest finite score from the configured scorer. The actor backend and scorer backend are separate config choices, so the experiment can test a small actor guided by a larger evaluator, the same model in both roles, or the ordinary `k=1` actor baseline.

If the scorer includes next-state information, each candidate is stepped before scoring. Pure-step environments can do this directly from the current state; non-pure adapters require restore support so candidate probes do not advance the live trajectory. If the scorer omits next state, only the selected action is stepped in the environment.

Any benchmarked method can guide the actor, through one of two scorer families. **Individual-prediction** methods (every `DenseSignalMethod` — `llm_direct`, `llm_verifier`, `llm_codegen`, `llm_eureka`, the vision methods, and `baseline_random`) are wrapped in `DenseSignalCandidateScorer`: each candidate gets a predicted value and `GuidedActor` argmaxes over them. **Ranking-based** methods (`llm_ranking`, `sdpo_ranking`, `sdpo_short_expert_continuation`, `delta_belief_ranking`) are wrapped in `RankingCandidateScorer`: a step's candidates are assembled into one `RankingPoint` (candidate 0 marked `ACTOR_PRIMARY`, the rest `RANKING_LLM`), ranked or scored in a single batched call, and the resulting per-candidate score vector drives the same argmax. A permutation output (`rank_batch`, used for `llm_ranking`) is converted so the candidate at rank position `p` scores `n - p` and the top-ranked candidate is the argmax — a `None` permutation degrades to all-`NaN` and the actor falls back to ... (very long line, trimmed to 1000 chars)

A compute-matched **`self_consistency`** baseline is available as a third strategy. It is an ordinary `GuidedActor` whose scorer selects an action by consensus rather than a learned value, sampling the same `guided_actor.k` candidates per step:

- **`DiscreteVotingScorer`** (`discrete` mode, `include_next_state=True`) steps each candidate to obtain its `resolved_action` and scores each candidate by its vote count (the number of candidates resolving to the same action). `GuidedActor` then picks the `argmax`, breaking ties at random — so the executed action comes from a most-voted group. The vote key falls back to `extracted_action` then to the cleaned display text when an action cannot be resolved.
- **`OpenEndedSelfConsistencyScorer`** (`open_ended` mode, `include_next_state=False`) implements Universal Self-Consistency (USC): it does not step the environment, computes each candidate's extracted action with the environment's answer extractor, and prompts a selector LLM to pick the most consistent action. The prompt forces a single choice even when there is no clear majority (asking for the most representative candidate in that case) and lets the selector reason for a sentence or two before ending its reply with the anchor phrase the parser keys on — robust to a selector that quotes candidates while reasoning, since the anchored phrase wins over bare mentions. It returns a one-hot score for the chosen candidate, or all-`NaN` so `GuidedActor` falls back to a random candidate — both on a parse failure and when the selector call cannot be completed (a transient API failure, e.g. a dropped connection, that survives the per-slot retry policy). When all candidates render to the same action text the selector call is skipped (outcome-invariant) and the first candidate is chosen. Because it never steps candidates, it works on restore-based environments (e.g. ALFWorld, Harbor).

Within a strategy, trajectories are rolled out in concurrent waves of `max_concurrent_trajectories`, advancing in lockstep so each decision step batches the actor generation across all in-flight trajectories' candidates (`active × k`) into one `generate_chat_batch`, and dense-signal scoring into one `evaluate_batch` — mirroring the batched MC rollouts in `predict.py`. Pure-step environments share one environment instance; restore-based environments use one persistent instance per concurrent trajectory (reset in parallel, closed at wave end), with transient per-candidate environments bounded by `restore_concurrency`. Per-trajectory RNG is derived from `seed + trajectory_index`, making results independent of wave size, and a trajectory that errors is isolated as a skipped point rather than aborting the wave. `max_concurrent_trajectories: 1` reproduces serial rollout.

The test-time scaling pipeline always reports `baseline_k1`. A `guided` strategy is added when `guided_actor.k > 1` or a scorer is configured, and a `self_consistency` strategy is added when `self_consistency` is configured (which requires `guided_actor.k > 1`). Transition metadata records all sampled candidates, their scores, the selected index, and any fallback used when every score is `NaN`. For `self_consistency` the recorded `scores` hold the vote counts (`discrete`) or a one-hot vector (`open_ended`).

### Step 1: `collect_dataset.py` — Data Collection

1. **Collect trajectories** — run the environment using `collect_trajectories()` with one or more backends (multi-policy support).
2. **Extract evaluation points** — extract `(state, action, next_state)` triples from trajectories via `collect_evaluation_points()`.
3. **Collect ranking points** (optional) — when `ranking_actions` is set, generate alternative candidate actions per evaluation point and build `RankingPoint` objects with shuffled candidates. Supports manual sampling (registered per-environment sampler) and LLM sampling (batched generation with deduplication and resampling). Collection keeps points with 2 or more unique candidates, up to the configured maximum `k`. Each candidate now carries explicit provenance: exactly one candidate per point is the actor-sampled primary action from the original trajectory, and any remaining candidates are ranking-only alternatives. Ranking-only resume runs may intentionally build `ranking_points` for only a sampled subset of the stored evaluation points while preserving the source dataset's point config and returns, but they first validate any saved `env_name`, `adapter`, `make_kwargs`, and effective step limit (`make_kwargs.max_steps` when present, otherwise top-level `max_steps`) against the current collection setup so mismatched resume datasets fail before ranking begins. If both step-limit representations are present, they must agree.
4. **Compute trajectory returns** — sum per-trajectory rewards via `trajectory_return()`.
5. **Save Dataset** — persist `evaluation_points`, `ranking_points`, `trajectory_returns`, raw `trajectory_results`, collection config, and metadata to a pickle file. The dataset is signal-type-agnostic: ordinary prediction targets remain the actor-sampled `evaluation_points`, while `ranking_points` are auxiliary structures for ranking evaluation only. Raw `TrajectoryResult` objects are stored so that evaluation points can be re-extracted with different sampling parameters via `resample_points.py` without re-collecting trajectories.

### Resampling: `resample_points.py`

When a dataset contains stored `trajectory_results`, evaluation points can be re-extracted with different `EvaluationPointsConfig` parameters (sampling strategy, max points per trajectory, max trajectories to keep) without re-running trajectory collection. The script loads the dataset, drops any stored failed trajectories, can optionally sample a random subset of stored trajectories first, applies the new sampling parameters via `resample_dataset()`, recomputes trajectory returns from the kept set, drops ranking points, and saves a new timestamped dataset pickle. With `--unique-tasks`, that pre-filtering step becomes balanced across all unique tasks instead of sampling raw trajectories directly, preferring stored `task_name` when available and falling back to `task_index`; the total requested subset must divide evenly across the retained task groups, and the per-task quota cannot exceed the smallest retained group. The output metadata is rewritten to describe the kept trajectories, and `trajectory_selection.selected_trajectory_indices` is reported against the source pickle's trajectory order. Pass@k summary fields such as `total_trajectories` and `num_tasks` are updated to the kept set, while `samples_per_task` is removed when the kept trajectories no longer have a uniform per-task count. If the source metadata carries a step limit only inside `make_kwargs.max_steps`, resampling also backfills the top-level `max_steps` metadata field with that effective value. A `--strip-trajectories` flag produces a smaller output without stored trajectories. This is the intended bridge from the trajectory-only `.pkl` artifacts produced by `compute_pass_at_k.py` to a point dataset that can later be passed to `collect_dataset.py --resume_ranking_from ...` for ranking collection.

### Ranking-Candidate Pruning: `scripts/utils/prune_ranking_dataset.py`

This maintenance utility keeps only points whose ranking data is wide enough for a requested threshold. It loads a dataset pickle, keeps only `ranking_points` with at least `N` candidates, drops any `evaluation_points` that do not have a matching surviving ranking point, compacts surviving trajectories to keep trajectory-level evaluation consistent, and writes a new timestamped pickle in the same filename family (or to an explicitly chosen path). The output metadata records the source artifact and the before/after pruning counts.

### Task-Based Dataset Pruning: `scripts/utils/prune_dataset_tasks.py`

This maintenance utility removes every trajectory, evaluation point, and ranking point belonging to one or more named tasks. It resolves task membership from stored task names in trajectory results or point replay metadata, compacts surviving trajectory indices so the output dataset stays self-consistent, preserves the source pickle, and writes a fresh timestamped sibling artifact by default. The output metadata records the removed task names together with before/after counts for evaluation points, ranking points, and trajectories.

### Step 2: `predict.py` — GT Estimation + Eval Methods

1. **Load Dataset** — read evaluation points and metadata from pickle.
2. **GT estimation** (if estimation configs exist) — reconstruct the environment from metadata, build GT methods via `create_gt_method()`, collect rollouts via `collect_grouped_gt()`, check abort threshold, and save per-method prediction JSONs (`gt_{name}_<timestamp>.json`).
3. **Eval method predictions** (if `eval_methods` are configured) — build display points via `prepare_display_points()`, run the configured methods (LLM-based: `LLMDirectMethod`, `LLMGVLMethod`, `LLMVerifierMethod`, `LLMCodeGenMethod`, `LLMEurekaMethod`; frozen-encoder-based: `VLEMethod`, `VIPMethod`, `LIVMethod`), check abort threshold after each method, and save per-method prediction JSONs (`eval_{name}_<timestamp>.json`).
4. **Ranking predictions** (for any non-SHAPED_REWARD estimation config, if the dataset has ranking points, and `skip_ranking_predictions` is not set) — for GT: create pseudo-`EvaluationPoint`s per candidate and run MC Q-value estimation, saving flat Q-values with `prediction_format: "ranking_q_values"`. When a matching Q_VALUE eval-point GT exists, ACTOR_PRIMARY candidates reuse those pre-computed Q-values and only alternative candidates are rolled out; when no matching Q_VALUE GT exists, all candidates (including ACTOR_PRIMARY) are rolled out from scratch. Each non-SHAPED_REWARD estimation config produces its own ranking GT prediction. For eval: run `LLMRankingMethod.rank_batch()` and convert orderings to per-candidate ranks, saving with `prediction_format: "ranking_predicted"`. Shared datasets may carry `ranking_points` even when reused for other signal types; those non-Q-value prediction runs ignore ranking points entirely. Setting `skip_ranking_predictions: true` on the prediction config bypasses this branch entirely (logging a single INFO line) even when the dataset has `ranking_points` — useful for reusing a ranking-enabled dataset for a regular-points-only run. Ordinary `q_value`, `state_value`, `advantage`, `potential`, and `shaped_reward` methods still predict only the actor-sampled actions from `evaluation_points`. Both ranking artifact types flatten values in prediction order, keep `config.candidate_counts` for chunking, and store stable per-point hash metadata so evaluation can recover the exact ordered ranking-point subset later.

### Step 3: `evaluate.py` — Correlations

1. **Load Dataset** — read metadata, `trajectory_returns`, and evaluation points.
2. **Load predictions** — resolve prediction sources centrally from files, directories, or timestamped sibling families; select the newest artifact per family; warn only for corrupt artifacts; then filter by method names.
3. **Compute correlations** — for each GT × eval pair: read `assumption` from GT config, resolve the eval `signal_type` from evaluation config or eval prediction artifacts, apply `transform_predictions()` on eval values (handles SHAPED_REWARD→POTENTIAL via cumsum), then compute correlations. NaN/null pairs are excluded; if fewer than 50 usable pairs remain, the summary keeps the `num_points` count but records the correlation and p-value as null.
4. **Trajectory-level comparison** — for SHAPED_REWARD assumption: sum eval predictions per trajectory, correlate with GT trajectory returns.
5. **Ranking evaluation** — ranking predictions (detected via `prediction_format` metadata) use `compute_ranking_correlation()`: GT Q-values are converted to ranks per point using each point's actual candidate count, then per-point Spearman correlation is averaged across all ranking points.
6. **Save summary** — write `summary.json` with experiment info, correlations, and values.

**Prefix predictions.** Predictions may cover only the first N evaluation points instead of the whole dataset. In the pointwise branch, GT and eval may have different lengths: `evaluate.py` truncates both to `min(len(gt), len(eval))` and runs correlations on that prefix. Both predictions are assumed to cover the first N evaluation points (`sampling_strategy="first"` at prediction time). If both prediction artifacts are longer than the current dataset's `evaluation_points`, the comparison is capped at the dataset prefix because transforms cannot address points not present in the dataset. In the ranking branch, modern artifacts resolve coverage from their own ordered `metadata.ranking_point_hashes` plus `config.candidate_counts`, so GT and eval can safely compare any ordered subset of the current dataset's `ranking_points`. When GT and eval cover ordered subsets of different lengths but agree exactly on their shared `min(len(gt), len(eval))` prefix (same `dataset_indices` and `candidate_counts` for those leading points), the longer side is truncated to that prefix and correlation runs on it; the truncation is hash-verified when both sides carry `ranking_point_hashes`, otherwise it is a load-bearing prefix assumption logged at WARNING. Non-prefix subset mismatches still raise. Correlation partitioning uses the truncated artifact's `candidate_counts`, not a dataset slice. Older ranking artifacts without point hashes fall back to legacy prefix validation (predicted coverage is always `[0..N-1]`) with a warning, which keeps them usable against unchanged datasets but cannot verify arbitrary subset identity. SHAPED_REWARD does not yet support prefix predictions — it raises a targeted error.

**NaN-tolerant sample aggregation.** `_aggregate_sample_correlations()` filters `None` and NaN correlations before computing `mean`/`std`/`min`/`max`/`median` (required — `statistics.stdev` rejects NaN under Python 3.12+). The aggregate carries `num_samples` (valid) and `num_nan` (dropped); when every sample is NaN, all stats are NaN and `num_samples` is `0`. Multi-sample codegen also emits a sibling `eval_{method}_mean` prediction (see `LLMCodeGenMethod` below); since that artifact has no `sample_group`, it is correlated independently and appears in the summary as its own method — distinct from the per-sample aggregate statistics.

## Reporting

`scripts/tabulate_results.py` renders one or more `summary_*.json` files (produced by Step 3) into a Markdown or LaTeX table. Rows are eval methods, columns are correlation metrics (Pearson, Spearman, Kendall τ, Sign Agreement). One table is emitted per GT method.

Cell formatting:

- Single-sample: `0.847 (p=0.012)` — p-value inline (suppress with `--no-pvalues`).
- Multi-sample codegen: `mean ± std [min, max] (N=k)` — per-sample correlation statistics across the N sampled functions. P-values are not aggregated upstream, so none are shown. The sibling `{method}_mean` row (correlation of the per-point mean prediction) appears as a separate single-sample row and uses the standard `0.847 (p=0.012)` format.
- Missing / NaN: `—` (em-dash).

Environment and model are assumed fixed across the scanned results.

Usage:

```
python scripts/tabulate_results.py <results_dir> [--format markdown|latex]
    [--output FILE] [--decimals N] [--metrics pearson spearman ...]
    [--no-pvalues]
```

The pure-logic module is `src/qval/reporting.py` (testable without I/O).

### Paper-figures table emission

`scripts.paper_figures.main` (invoked with `--source real --results-root catalogs/qval_benchmark/data/evaluations --out figures/paper`) writes `figures/paper/tables.tex` alongside the figure PDFs. The file contains one NeurIPS-formatted LaTeX table per applicable `(env, signal_type, modality)` combination, covering Spearman ($\rho$) and Kendall ($\tau$) correlations with significance stars (`*` p<.05, `**` p<.01, `***` p<.001).

Figures 1-4, 7, and 8 are modality-agnostic. Methods that support both text
and image inputs (`direct-*`, `gvl`, `ranking`) use their text result in these
figures, while their image result is reserved for the modality-ablation figures
5 and 6. Vision-only methods (`vlm-*`, `vip`, `liv-*`) remain in the
modality-agnostic figures; if a method has no Q-value result but has a
state-value result, the main-result figures use that state-value result as the
available method score.

The TerminalBench table set includes separate Q-value text tables for the
Codex 5.5 and Opus 4.7 GT actors. The GT-choice ablation plots additionally
include the DeepSeek v3.2 GT actor. The GT-ablation plots use Kendall tau
values and render GT actors in the configured order. Figure 8 reads both standard
Q-value correlations and ranking-style `ranking_gt_qv_*` blocks when the plot
metric supports them, but only ranks methods that have finite scores under
every GT actor shown in the panel. The companion GT-ablation bar plot uses the
same method filter and shows the mean correlation value for each GT actor,
averaged across applicable models.

Schema is fixed by `METHOD_REGISTRY` + `V_VALUE_REAL_NAMES` + `FAMILIES`, not by what's on disk: every conceptually-applicable `(method, model)` cell appears in the schema, with em-dash (`---`) signalling experiments not yet run. Conceptually-inapplicable rows/columns (e.g. `$\Delta$belief` or `verifier` in V-value tables) are omitted from the schema entirely. Slice applicability per env follows `MULTIMODAL_ENVS` (envs with vision experiments) and `SIGNAL_TYPE_ENVS` (envs with V-side experiments) in `domain.py` — envs in neither set (currently TerminalBench) get only Q-value text tables. Every table emits as `sidewaystable[p]` (rotated landscape page) because codegen `\makecell` cells push widths past the upright text-rectangle even for the 7-LLM text tables.

Cell variants:

- Standard correlation: `$0.412^{**}$` — value with significance stars.
- Ranking-style methods (`ranking`, `sdpo`, `sdpo-gt`, `$\Delta$belief`): `$0.412 \pm 0.083$` (mean ± std over per-trajectory Spearman) on the $\rho$ row; `$\tau$` row is `---` (Kendall N/A).
- Codegen aggregate row (`codegen`): `\makecell{$0.412 \pm 0.083$ \\ $[0.310, 0.514]$}` — mean ± std on top, [min, max] below; no p-value (per-sample p's are dropped during aggregation). Sourced from either an inline `aggregate_correlations` payload or 16 `codegen-s<i>` sample correlations (per-sample wins when both are present).
- `codegen-avg` row is a *separate* method (`codegen-mean_qv`) that averages predictions before correlating once — rendered as a normal `StandardCell`.

Required preamble additions in `paper/main.tex` (NeurIPS 2026 single-column template):

```latex
\usepackage{multirow}
\usepackage{makecell}
\usepackage{rotating}
```

`booktabs` is already loaded by the NeurIPS template. The shared `\section*{Notation}` block at the top of `tables.tex` documents all cell semantics so per-table captions stay terse.

The pure-logic modules are `scripts/paper_figures/tables_data.py` (slice schema + JSON ingestion) and `scripts/paper_figures/tables_render.py` (LaTeX rendering). Both are testable without I/O against synthetic summary fixtures.

## Key Design Decisions

### Stateless environments

qval relies on the stateless design of llenvs environments (`env.step(state, action)` is a pure function). This enables Monte Carlo rollouts from arbitrary mid-trajectory states — the same state always produces the same result, so we can branch and explore freely.

### DenseSignalFunction protocol

The simplest interface for a method being evaluated:

```python
def __call__(state, action, next_state) -> float
```

The method receives the full `(s, a, s')` triple and returns a scalar. Which components it uses depends on the signal type — a state-value method may only use `state`, while a Q-value method uses `state` and `action`.

### DenseSignalMethod ABC

For methods that need richer context (task description, reward description, example trajectories) or a setup phase, subclass `DenseSignalMethod`:

```python
class MyMethod(DenseSignalMethod):
    def __init__(self, context: MethodContext, ...) -> None:
        super().__init__(context)
        # method-specific setup

    def evaluate(self, point: EvaluationPoint) -> float:
        # self.context.signal_type, task_description, etc. available
        ...
```

`DenseSignalMethod` provides `__call__` and `evaluate_batch` automatically, so subclasses satisfy both `DenseSignalFunction` and `BatchDenseSignalFunction` protocols.

### MethodContext

`MethodContext` bundles the information sources available to a method:

- `signal_type` (required) — what type of signal to produce (V, Q, A, potential, shaped reward).
- `policy_assumption` — which policy the value functions assume: `PolicyAssumption.OPTIMAL` (default) or `PolicyAssumption.SELF`. Controls policy-dependent language in prompts (e.g., "assuming optimal play" vs "assuming you are the agent").
- `discount_factor` — discount factor γ (0.0–1.0, default 1.0). Used in prompt descriptions and GT computation. When 1.0, prompts say "undiscounted cumulative reward"; otherwise "discounted cumulative reward (discount factor γ=X)".
- `task_description` — natural language description of the environment/task.
- `reward_description` — description or code of the sparse reward function.
- `example_trajectories` — list of `ObservableTrajectory` instances with observable states, actions, and rewards. Can be loaded from JSON files via `example_trajectories_file` in the environment config YAML.
- `evaluator_extractor` — extractor for parsing evaluator LLM output to numeric signal values.
- `environment_extractor` — extractor for parsing actor output into environment actions.
- `include_actor_thinking` — when `True`, pass raw actor output (with `<think>` traces) to methods instead of the clean `resolved_action`.
- `include_current_thoughts` — when `True`, include the actor's ReAct-style reasoning for the current evaluation point in a dedicated `**Actor's Reasoning:**` section in the user prompt (between current state and current action). The `<thought>` tags are stripped from the displayed action text to avoid duplication.
- `include_history_thoughts` — when `True`, include `[Turn i - Reasoning]` sections for history turns that have an extracted thought (between state and action per turn).
- `enable_thinking` — whether the method's backend has thinking/reasoning tokens enabled. When `False`, closing prompt blocks strip the "After any reasoning, ..." preamble to avoid encouraging visible reasoning that wastes output tokens. Populated from `BackendConfig.enable_thinking` by `predict.py`.
- `environment_notes` — runtime-determined notes about the environment (e.g., step limit). Rendered as a `## Episode Configuration` section in system prompts between Task Description and Reward Function. Not loaded from YAML — populated programmatically from the dataset's effective step limit in `predict.py` (`make_kwargs.max_steps` when present, otherwise top-level `max_steps`). If both are present, they must agree.

All fields except `signal_type` are optional. The benchmark can vary what information is provided to study the effect of each source.

### BatchDenseSignalFunction

Methods that support batch evaluation can implement the `BatchDenseSignalFunction` protocol in addition to `DenseSignalFunction`. This protocol adds an `evaluate_batch(points)` method that receives all evaluation points at once, enabling the method to batch its own inference (e.g., a single forward pass through an LLM for all points). The benchmark automatically detects this via `isinstance` check and uses the batch path when available.

### Built-in methods

Two families are provided in `qval.methods`: LLM-based methods that prompt a generative backend and parse its output, and frozen-encoder methods that score images (and optionally text) through a pretrained vision model without any text generation.

LLM-based methods:

- **`LLMDirectMethod`** — prompts an LLM for numeric signal values. By default it uses one evaluation point per prompt, but `prompt_batch_size > 1` supports two batch protocols. `prompt_batch_mode="packed"` packs several points into one prompt and expects a comma-separated list of values in the same order (inside `<answer>` tags for answer-tag presets). `prompt_batch_mode="sequential"` asks for one datapoint per user turn while keeping prior numeric assistant replies in the conversation; multi-point groups are still transported through `generate_chat_batch()` round-by-round for efficient backend batching. `batch_size` controls only the transport chunk size, not the number of points grouped together. `prompt_batch_size=None` disables chunking inside `plan_prompt_groups` so each logical cluster (one trajectory for `prompt_grouping="trajectory"`, all points for `"contiguous"`/`"random"`) becomes a single un-chunked group. Sequential mode is currently disabled when budget-aware prompt truncation is active.
- **`LLMGVLMethod`** — a dedicated GVL-style numeric evaluator. Each prompt targets exactly one evaluation point, but includes the full stored trajectory context for that point: all surrounding same-trajectory state-action transitions are rendered in deterministic shuffled order, the target transition is always rendered separately at the end, and chronology markers such as `[Step 1/20]` are intentionally omitted. When `include_next_state=true`, only the target transition includes its next state; surrounding context transitions remain state-action only so the shuffle does not leak local order. This method requires `Dataset.trajectory_results` so it can reconstruct exact per-point trajectory context and fails fast when those stored trajectories are missing or misaligned.
- **`LLMCodeGenMethod`** — prompts an LLM once to generate a Python `signal_function(...) -> float` whose signature varies by signal type: `(state: str)` for STATE_VALUE/POTENTIAL, `(state: str, action: str)` for Q_VALUE/ADVANTAGE, and `(state: str, action: str, next_state: str)` for SHAPED_REWARD. The response is parsed with `extract_code` (supporting multiple fenced code blocks), validated statically with `validate_code` (syntax + top-level function check), and the compiled function's signature is verified with `validate_signature` against the expected parameter list. The compiled function is cached; subsequent evaluations just execute it. With `num_samples > 1`, N independent functions are generated in one batched LLM call; each is evaluated on the full point set and saved as `eval_{method}_s{i}_<ts>.json` with `config.sample_group = {method}`. After the per-sample files are saved, `_run_multi_sample_codegen` also saves a standalone `eval_{method}_mean_<ts>.json` artifact holding the per-point NaN-skipping mean of the N samples' predictions (NaN at a point only when every sample was NaN there). The mean artifact carries no `sample_group`, so `evaluate.py` treats it as its own eval method row rather than folding it into the per-sample aggregate statistics.
- **`LLMEurekaMethod`** — adapts Eureka-style iterative code search to qval’s signal-function interface. Each search iteration generates `num_samples` candidate `signal_function`s, compiles and executes valid candidates on the full evaluation-point set, then asks the same backend to judge candidates from full-batch summaries plus a sampled subset of datapoints rendered with the standard state/history/action formatting. The winning code, judge feedback, and any runtime exception summary observed while evaluating the winner are inserted into the next generation prompt's reflection block, so the LLM can see and avoid concrete failures (e.g. calls to unavailable sandbox builtins). The first runtime error per candidate is captured on `_CandidateResult.runtime_error` and the winning candidate's value is also exposed via `LLMEurekaMethod.last_runtime_error`. Generated functions may return either a scalar float or `(float, dict[str, float])`, where the optional component dictionary is used ... (very long line, trimmed to 1000 chars)
- **`LLMRankingMethod`** — prompts an LLM to rank the candidate actions listed in a `RankingPoint` by Q-value. Operates on `RankingPoint` objects (not `EvaluationPoint`), so it is not a `DenseSignalMethod` subclass. `rank(point)` returns a best-to-worst ordering of action numbers (1-indexed), or `None` on parse failure. `rank_batch(points)` uses `generate_chat_batch()` with optional chunking. The `parse_ranking()` function extracts the last valid comma-separated permutation of `[1..k]` from the response, where `k = len(point.candidates)` for that point.

Frozen-encoder methods (visual reward models, `backend_name` must reference an encoder-type backend; no prompting, no text generation):

- **`VLEMethod`** (method types `vlm_rm`, `vlm_sor`) — frozen CLIP / SigLIP encoder (via HuggingFace `AutoModel`). `vlm_rm` (Rocamonde et al. 2023) computes `cos(E_text(goal), E_image(state))`; optional `vle_baseline_prompt` + `vle_alpha` enable goal-baseline regularization that projects the state embedding onto the line between baseline and goal before measuring distance-to-goal. `vlm_sor` (Baumli et al. 2023) takes a list of negative goals and returns the softmax probability on the current goal over the full goal set ℒ; the head deduplicates the current goal from the negatives so the softmax denominator matches the paper's `Σ_{l' ∈ ℒ}`. Setting `vle_beta` gives the paper's thresholded binary variant.
- **`VIPMethod`** (method type `vip`, Ma et al. 2022) — ResNet-50 pretrained on Ego4D with a self-supervised goal-conditioned objective. Image-only: compares an encoded state image to an encoded goal image. `state_value` returns `-||φ(state) - φ(goal)||₂`; `q_value` uses `point.next_state`; `shaped_reward` returns `V(next; goal) - V(state; goal)` (paper eq. 1). Goal images come from either `trajectory_end` (last image of the point's trajectory, matching VIP's training setup) or `env_image` (a PNG at `vip_env_goal_image_path`). Checkpoint auto-downloads to `$HF_HOME/vip/model.pt`.
- **`LIVMethod`** (method types `liv_img`, `liv_txt`, Ma et al. 2023) — CLIP RN50 fine-tuned on EpicKitchens with the LIV objective. `liv_img` takes an image goal (same resolvers as VIP); `liv_txt` takes a text goal (same resolvers as VLE: per-point `obs.task.text`, `liv_goal_text` static override, or `context.task_description`). Both support all three signal types and two similarities: `liv_similarity: cosine` (paper default) or `liv_similarity: l2` (VIP-style). Checkpoint is pulled from HuggingFace Hub (`jasonyma/LIV`); the model class is loaded via the `clip` package (installed from `git+https://github.com/openai/CLIP.git`).

### Action sampling for ranking

Ranking points are constructed during collection (Step 1) when `ranking_actions` is configured. Each `RankingPoint` bundles a state with 2 to `k` shuffled `ActionCandidate` objects: the primary trajectory action plus as many unique alternatives as collection can obtain. Each candidate stores its provenance explicitly (`ACTOR_PRIMARY`, `RANKING_MANUAL`, or `RANKING_LLM`), so the actor-sampled action remains identifiable even after shuffling. Two sampling modes are supported:

- **LLM sampling** (`ranking_sampling_mode: "llm"`) — generates extra actions from the actor backend with separate sampling params (typically higher temperature for diversity). Actions are validated via `env.step()`, deduplicated by `resolved_action`, and resampled in batched rounds until no more unique candidates are found or the configured maximum is reached.
- **Manual sampling** (`ranking_sampling_mode: "manual"`) — uses a per-environment registered action sampler from `qval.action_samplers`. Built-in: `"frozen_lake"` enumerates from `{left, down, right, up}`. Samplers receive `(state, k, rng)` and return `list[ActionCandidate]`.

For non-pure environments such as Harbor, candidate validation uses fresh env instances plus restore-to-state before the single candidate step. `collection.restore_concurrency` controls how many of those restore-based validations can run in parallel during collection; it is separate from LLM generation batching.

### Ground truth estimation

Ground truth estimation methods (`MCMethod`, `MCTSMethod`) are `DenseSignalMethod` subclasses that compute ground truth signal values via rollouts. They share the same `evaluate(point) -> float` and `evaluate_batch(points) -> list[float]` interface as LLM-based methods.

Each GT method exposes:

- **`collect_returns(points)`** — collects raw per-rollout returns as `list[RawPointReturns]`, without aggregating them. This enables rollout reuse across estimation configs that differ only in aggregation.
- **primitive trajectory collectors** — `MCMethod` can collect rollouts from `point.state`, forced `(point.state, point.action)`, or `point.next_state` before reducing them into returns. Persistent reuse is built on these primitive trajectory families.
- **`rollout_identity()`** — returns a hashable tuple used for in-memory rollout sharing within a run. It excludes aggregation and rollout count, so methods that only differ in `num_rollouts` can share one larger collection and slice the needed prefix.
- **`evaluate_batch(points)`** — delegates to `collect_returns()` + `aggregate_returns()` to produce `list[float]`.

The `Policy` type bundles `(backend, sampling_params, system_prompt)` and is passed to GT methods at construction time to define the rollout behavior.

The standalone `aggregate_returns(raw_returns, aggregation)` function converts raw rollout returns into `SignalEstimate` values. For STATE_VALUE/Q_VALUE it aggregates returns directly. For ADVANTAGE (when `next_state_returns` is populated) it aggregates V(s) and V(s') independently then computes `V(s') - V(s)`.

The `create_gt_method()` factory dispatches to the appropriate implementation based on `EstimationConfig.method`:

- **`"mc"`** → `MCMethod`
- **`"mcts"`** → `MCTSMethod`

### MCMethod

`MCMethod` subclasses `DenseSignalMethod` and estimates ground truth by sampling N rollouts from a given state (or state-action pair). The `collect_returns()` method gathers raw returns, and `evaluate_batch()` delegates to `collect_returns()` + `aggregate_returns()`. The `discount_factor` parameter (default 1.0) applies geometric discounting to trajectory returns: `Σ γ^t * r_t`.

The `aggregation` field on `EstimationConfig` controls how returns are combined:

- **MEAN** — `sum(returns) / len(returns)`. Estimates the expected value under the rollout policy.
- **MAX** — `max(returns)`. Approximates the value under a best-of-N policy. Useful when the evaluated method targets an optimal policy rather than the actor's average behavior.

For ADVANTAGE estimation, the aggregation is applied independently to V(s) and V(s') returns before computing `A(s,a) = aggregate(V(s')) - aggregate(V(s))`.

### MCTSMethod

`MCTSMethod` subclasses `DenseSignalMethod` and uses Monte Carlo Tree Search to estimate ground truth values. It builds a search tree that focuses computation on more promising branches via UCB1 selection, which is especially valuable with MAX aggregation — MCTS is more likely to discover good action sequences than flat MC.

**Wide expansion variant**: When expanding a node, W children (actions) are generated via a single `generate_chat_batch()` call, then all non-terminal children are simulated in a single `batch_rollout_from_states()` call. Each expansion round makes exactly 2 batched LLM calls regardless of W.

**Budget**: `mcts_iterations` controls the total number of simulation returns. Each expansion round produces up to `mcts_expansion_width` returns. Rounds continue until the budget is exhausted.

**Algorithm**:
1. **Select** — walk the tree from root via UCB1 (`exploitation + c * sqrt(ln(parent_visits) / child_visits)`) to find an unexpanded non-terminal leaf.
2. **Expand** — generate `min(W, remaining_budget)` actions via `generate_chat_batch()`, step the environment for each, creating child nodes. Terminal children's returns are recorded immediately.
3. **Simulate** — batch rollout from all non-terminal children via `batch_rollout_from_states()`. Each child's total return = cumulative reward from root to child + simulation return.
4. **Backpropagate** — update visit counts and total values from each child up to the root.

**Signal type dispatch** mirrors MCMethod:
- **STATE_VALUE** / **POTENTIAL**: run MCTS from `point.state`
- **Q_VALUE**: force the first action via `env.step()`, add step reward, then run MCTS from the resulting state
- **ADVANTAGE**: run MCTS from `point.state` for V(s) and from `point.next_state` for V(s')
- **SHAPED_REWARD**: raises `ValueError` — not supported for rollout-based GT estimation

**Cumulative reward tracking**: each tree node stores the sum of step rewards from root to that node. For environments with per-step penalties (e.g., GEM games), this correctly accumulates rewards along the path.

**Discount factor**: each `MCTSNode` tracks its `depth` from the root. Step rewards at depth d are weighted by `γ^d`, and simulation returns from a child at depth d are scaled by `γ^d`. The discount factor is included in `rollout_identity()` to prevent sharing between methods with different γ values.

### Rollout reuse

When multiple `EstimationConfig` entries share the same rollout-generation parameters, `collect_grouped_gt()` groups them and collects primitive trajectories once per group. This avoids redundant LLM calls — for example, running both `mc_mean` and `mc_max` with 64 rollouts requires only 64 rollouts total, not 128. If one config asks for 32 rollouts and another asks for 64, the collector runs 64 once and the smaller config uses the first 32 stored samples.

The rollout identity includes a **rollout type** derived from the assumption: STATE_VALUE and POTENTIAL both map to `"v_s"`, Q_VALUE maps to `"q_sa"`, and ADVANTAGE maps to `"advantage"`. This means configs with STATE_VALUE and POTENTIAL assumptions share rollouts automatically.

For MC configs, the in-memory rollout identity key captures the rollout primitive family and the rollout-generation behavior, but not aggregation or rollout count. Persistent reuse is stricter and runtime-derived: trajectory stores are additionally bound to the dataset file content hash and the logical point namespace. Prompt guardrails such as prompt-budget limits and text-truncation floors are intentionally excluded from persistent compatibility because they are operational safety controls rather than rollout semantics.

SHAPED_REWARD assumption configs are excluded from rollout grouping entirely — they use trajectory-level comparison without MC estimation.

#### V(s) cross-group sharing

ADVANTAGE estimation requires both V(s) and V(s') rollouts. When a STATE_VALUE (or POTENTIAL) config with the same rollout parameters is also present, the V(s) component of ADVANTAGE is identical to what STATE_VALUE already collected. The `collect_grouped_gt()` utility exploits this:

1. V(s) groups are processed first, and their raw returns are cached.
2. ADVANTAGE groups look up matching V(s) cache entries by comparing rollout parameters (excluding rollout type).
3. When a match is found, `collect_returns()` receives the cached V(s) as `v_s_cache`, and only V(s') rollouts are collected — halving the ADVANTAGE GPU budget.

This means running both `assumption=state_value` and `assumption=advantage` with 64 rollouts requires `P×64` (STATE_VALUE) + `P×64` (ADVANTAGE V(s')) = `P×128` rollouts, rather than `P×64` + `P×128` = `P×192`. If compatible `Q_VALUE` rollouts are also available, ADVANTAGE can reuse their continuation trajectories for `V(s')` and avoid additional next-state sampling.

#### collect_grouped_gt()

The `collect_grouped_gt()` utility in `estimator.py` centralizes GT collection logic. It groups methods by rollout identity, handles V(s) cross-group sharing, and optionally persists MC primitive trajectories into append-only stores before deriving `(gt_values, raw_returns, aborted_points)`. The `aborted_points` dict maps method name → set of point indices that were aborted due to backend errors, enabling callers to check abort thresholds. It is used by:

- `scripts/pipeline/predict.py` for GT estimation and eval method prediction (config-driven backends)

Within a single process, the grouping key includes `id(method.policy.backend)` so that methods using different backend instances never share in-memory collections. Cross-run persistence does not depend on object identity; it uses normalized runtime descriptors plus the dataset content hash.

Because the persistent cache key is the SHA-256 of the JSON-serialized rollout-generation spec — including every field of the resolved `BackendConfig` — adding an optional field to `BackendConfig` invalidates the cache for all backends. The `mc_rollout_persistence.trust_sibling_hash` flag is the recovery escape hatch: it tells the rollout store to adopt a unique sibling directory with complete shard files under the same `dataset_fingerprint` parent when the expected hash directory is missing, renaming it into the expected path before logging a field-level diff between the cached and requested specs. Adoption is refused when more than one sibling exists or when two different specs in one run would claim the same physical sibling. See `docs/configuration.md` for usage guidance.

LLM call logging during `collect_returns()` is attributed to the first config's estimation phase. MC summary events are logged per-config with the correct aggregated values.

### Lockstep batched MC estimation

MC estimation uses lockstep batched rollouts via `generate_chat_batch()`. All rollouts across all evaluation points are flattened into a single batch:

- **STATE_VALUE** / **POTENTIAL**: P points × N rollouts = P\*N rollouts via `batch_rollout_from_states`
- **Q_VALUE**: P\*N rollouts via `batch_rollout_from_state_actions`
- **ADVANTAGE**: P\*2N rollouts (N from `point.state` + N from `point.next_state`) in one `batch_rollout_from_states` call. When a `v_s_cache` is provided (from a prior STATE_VALUE collection), only P\*N rollouts are needed for V(s')

Each step of the lockstep loop: filter active rollouts, build messages for all of them, make one `generate_chat_batch()` call, step environments, mark terminated rollouts as done. Rollouts that finish early naturally drop out of subsequent batches.

The `batch_size` parameter on `BenchmarkConfig` controls the maximum number of concurrent rollouts per lockstep batch. When the total number of rollouts exceeds `batch_size`, they are processed in chunks. This enables GPU memory management — set `batch_size` to match your backend's capacity (e.g., 256 for vLLM on a single GPU).

### Backend routing

Each `EstimationConfig` can specify a `backend_name` to select which backend to use for its rollouts. `predict.py` resolves named backends via the shared backends registry referenced by `backends_config_path`.

This enables comparing ground truth estimation under different rollout policies — e.g., MC estimation with a strong vs weak model.

Backend name is included in the rollout identity key, so estimations using different backends never share rollouts.

### Multi-assumption evaluation

A method produces ONE set of numbers, but each `EstimationConfig` specifies an `assumption` — how to interpret those numbers for ground truth comparison. This enables testing multiple hypotheses about what a method's output represents in a single experiment.

Three orthogonal axes control evaluation:
1. **Signal type** (`BenchmarkConfig.signal_type`): what the method is prompted to produce (affects prompts only).
2. **Assumption** (`EstimationConfig.assumption`): how to interpret method output for GT comparison — determines GT computation, any needed output transformation, and comparison granularity.
3. **Correlation metrics** (`correlation_methods`): Pearson, Spearman, Kendall's Tau, sign agreement.

#### Available assumptions

| Assumption | What it measures | GT needed | Comparison granularity |
|---|---|---|---|
| `state_value` | Direct state-value accuracy | V(s) via MC | Point-level |
| `q_value` | Direct Q-value accuracy | Q(s,a) via MC | Point-level |
| `advantage` | Per-transition action quality | A(s,a) via MC | Point-level |
| `potential` | Quality of Φ as approximation to V* | V*(s) via MC | Point-level |
| `shaped_reward` | Whether shaped reward preserves trajectory ranking | Trajectory returns (no MC) | Trajectory-level |

#### Potential extraction

When `signal_type=SHAPED_REWARD` and `assumption=POTENTIAL`, the method outputs per-transition F(s,a,s') values. Before comparison with GT V*(s), these are transformed into potential values via cumulative sum: Φ(s_0) = 0, Φ(s_k) = Σ_{i=0}^{k-1} F(s_i, a_i, s_{i+1}). This transform is implemented in `transforms.py`.

#### Trajectory-level comparison

For `assumption=SHAPED_REWARD`, method predictions are summed per trajectory and correlated with GT trajectory returns. No MC rollouts are needed — GT is just the environment's trajectory returns. This evaluation is especially useful for code-based methods (cheap per-transition evaluation).

#### Signal transforms (`transforms.py`)

The `transforms` module provides utilities used by `evaluate.py`:

- **`sum_per_trajectory(predictions, points, num_trajectories)`** — sums predictions per trajectory.
- **`extract_potential_values(predictions, points)`** — recovers Φ from F via cumulative sum.
- **`transform_predictions(predictions, points, signal_type, assumption)`** — applies the appropriate transform based on signal_type→assumption mapping.

## Observable Trajectory Store

The `trajectory_store` module provides observable-only trajectory types for persistence and prompts. These types strip hidden state, keeping only information visible to an observer: observation text, raw/extracted actions, reward, and termination flag.

### Types

- **`ObservableTransition`** — a single step: `observation`, `raw_action`, `extracted_action`, `resolved_action`, `reward`, `terminated`.
  - `raw_action` — the full model generation text (including `<think>` tokens).
  - `extracted_action` — the extractor output from the environment adapter (tier 2), or `None` when extraction failed entirely.
  - `resolved_action` — the formatted native action from the adapter (tier 3), e.g. `"right"` for gym action 2. `None` when the action was invalid or the adapter doesn't provide this.
- **`ObservableTrajectory`** — a sequence of transitions plus `total_reward` and `success`.

### Conversion

`from_trajectory_result(result, reward_signal_name)` converts a live llenvs `TrajectoryResult` to `ObservableTrajectory`. It uses `serialize_observation()` for text conversion, `serialize_action()` for `raw_action` (full generation), `extracted_action` (from the llenvs transition, tier 2) for `extracted_action`, and `resolved_action` (tier 3) directly from the transition.

### Save/Load

Trajectories are persisted as JSON:

```python
from qval import save_trajectories, load_trajectories

save_trajectories(trajectories, "trajectories.json", metadata={"env": "chain_sum"})
loaded = load_trajectories("trajectories.json")
```

JSON format:

```json
{
    "version": 1,
    "metadata": { "environment": "chain_sum", "model": "example-model" },
    "trajectories": [
        {
            "total_reward": 1.0,
            "success": true,
            "transitions": [
                {
                    "observation": "...",
                    "raw_action": "<think>...</think>42",
                    "extracted_action": "42",
                    "resolved_action": "42",
                    "reward": 1.0,
                    "terminated": true
                }
            ]
        }
    ]
}
```

### Integration

- **MethodContext**: `example_trajectories` accepts `list[ObservableTrajectory]`. These are formatted into structured prompts with turn-by-turn observation/action/reward display.
- **Context loader**: environment YAML files can specify `example_trajectories_file` to load trajectories from a JSON file relative to the contexts directory.
- **Collection script**: `collect_dataset.py` saves trajectory JSON via `collection.trajectories_path` alongside the pickle output.

## Crash Resilience

The pipeline scripts are designed to survive crashes (OOM kills, SLURM timeouts, exceptions) with minimal data loss:

- **Collection resume checkpoints**: `collect_dataset.py` saves a rolling checkpoint after each chunk of trajectories to a stable fingerprint-keyed path next to the dataset (`{dataset_stem}_{fp16}.checkpoint.pkl`, built by `run_checkpoint.py`). The fingerprint hashes every content-affecting input (environment identity and context, resolved system prompt, seed, resolved task indices, policy backend content descriptors) and excludes transport/chunking knobs, so a rerun with the same config computes the same path. On restart with `resume: true` (the default), the checkpoint is loaded, per-policy completed task multisets are subtracted, and only missing work runs — fully completed policies skip backend creation; a fully resumed run still creates a fallback backend when LLM ranking sampling needs one. The checkpoint contains retained `trajectory_results` with empty evaluation points (points are sampled fresh after collection; `resample_points.py` still works for manual salvage) and per-result `collection_policy_index`/`collection_backend_name` provenance. Failed trajectories and Harbor runtime-probe-risky trajectories are dropped before checkpointing and are retried on resume. The checkpoint is deleted on successful final save; `--fresh` ignores and removes it; orphans from abandoned configs are left in place.
- **Test-time scaling resume**: `test_time_scaling.py` keeps one rolling checkpoint per strategy (`{output_stem}_{strategy}_{fp16}.checkpoint.pkl`), flushed by the batched rollout's `on_trajectory_complete` callback the moment each trajectory finishes — a crash mid-wave loses only the trajectories still in flight, not the whole wave. Per-trajectory RNG (`base_seed + trajectory_index`) makes results independent of wave composition, so a resumed run executes only the missing trajectory indices and merges loaded + fresh results in trajectory order, reproducing the uninterrupted run's outputs exactly. Checkpoints are deleted only after every strategy's final artifacts are saved (the per-strategy pickles are written at the very end, so earlier deletion would orphan completed strategies). Requires `output_path`; abort-threshold counting and logged stats cover only the current run's portion; summary metadata records `resumed_by_strategy`.
- **GT rollout persistence**: `predict.py` uses `MCRolloutStore` — an append-only, shard-based store with atomic writes. Each chunk of rollouts is persisted immediately as a separate shard file. On restart with `resume=True`, already-collected rollouts are reloaded and only missing ones are re-collected.
- **Per-method prediction saves**: `predict.py` saves each method's prediction to its own timestamped JSON file immediately after that method completes and before any abort-threshold check for that method. Already-saved predictions survive crashes and threshold-triggered aborts. This includes `ranking_gt_*` artifacts, which are reused on rerun only when their saved config, flattened candidate count, ranking-point identity hash, rollout-generation descriptor hash, and effective return semantics still match the current run.
- **SIGTERM handling**: `ResourceMonitor` converts SIGTERM (sent by SLURM before timeout) into `SystemExit`, which is caught by crash handlers in both scripts. `collect_dataset.py` saves a partial dataset, including ranking-only resume runs; `predict.py` saves experiment logs.
- **Experiment logs**: Both scripts stream log records incrementally to JSONL sinks and do a final save on completion or crash.
- **Quota-aware sleep-and-retry**: `QuotaRetryBackendWrapper` wraps every constructed backend so that a `QuotaExhaustedError` from the backend (raised permissively by both CLI backends — Codex and Claude Code — for subprocess failures that don't match a more specific classifier) triggers a fixed sleep-and-retry schedule of one 5-minute sleep followed by five 1-hour sleeps. If retries exhaust, the error propagates to the top-level crash handler, which detects it and returns cleanly (exit 0) after the normal crash-summary and log-save path so persisted rollouts and completed groups stay usable on re-run. Set `quota_retry_policy: "abort"` on a `BackendConfig` to disable the retry loop and let the error propagate immediately.

  The Claude Code backend recognizes two `RecoverableInputError` subclasses as deterministic per-input failures distinct from quota exhaustion: `PromptTooLongError` (text matching the context-limit needles) and `RefusedByPolicyError` (text matching `usage policy` or `anthropic.com/legal/aup`, including the `is_error: true` / `subtype: success` payload shape Anthropic returns for AUP refusals). The quota retry wrapper does not sleep on either — they propagate immediately so the offending rollout is dropped without burning 5 hours retrying a request that will refuse forever.
- **CLI-backend code paths**: `codex` and `claude_code` share two pieces of code (set membership check `_CLI_BACKEND_TYPES`): (i) `sampling_params_from_config` strips temperature/top_p/top_k/thinking knobs that the underlying CLIs reject; (ii) `apply_cli_backend_tmux_prompt_hardening` appends the same `CLI_BACKEND_TMUX_PROMPT_HARDENING` block to the system prompt for Harbor runs in `text_exec_mode: tmux_session`, gated by the `cli_backend_tmux_prompt_hardening` pipeline flag. Backend construction and quota-retry are otherwise per-backend.

## Experiment Logging

The `experiment_logging` module provides detailed capture of every LLM call during an experiment. It consists of two main components:

### LoggingBackend

A `ModelBackend` wrapper that intercepts all `generate_chat` and `generate_chat_batch` calls. Since the same backend flows through trajectory collection, MC estimation, and method evaluation, wrapping it once captures everything with zero changes to methods, rollout, or MC estimator internals.

`LoggingBackend` forwards backend lifecycle as well: `close()` delegates to the
wrapped backend, and the wrapper can be used as a context manager. The
collection and prediction scripts explicitly close wrapped backends at script
boundaries so local model memory and API client sessions are released
deterministically after use.

Backends are first wrapped in `QuotaRetryBackendWrapper` (which handles quota-exhaustion retries), then in `LoggingBackend`, then in `SecondElicitationBackendWrapper`. Quota retries inside the quota wrapper are transparent to logging. Second elicitation calls are intentionally outside the logging wrapper, so the first truncated generation and the follow-up generation appear as separate log records with their own messages, sampling params, and outputs.

When `second_elicitation` is enabled and a generation returns `MAX_TOKENS`, the wrapper appends the partial assistant output plus `DEFAULT_EARLY_STOPPING_SUFFIX`, then sends a user message asking for the final answer while following the formatting instructions specified above exactly. The follow-up call uses `second_elicitation_max_tokens`, clears thinking-budget controls, sets `disable_thinking=True`, and clears `second_elicitation_suffix` to avoid recursion. This applies to direct qval method calls, MC rollouts, ranking sampling, and any other `generate_chat` / `generate_chat_batch` calls routed through the pipeline backend wrapper.

Each intercepted call records:
- **index** — monotonic counter across all calls
- **phase** — which stage of the pipeline made this call (e.g., `"trajectory_collection"`, `"mc_estimation"`, `"method:llm_direct"`)
- **call_type** — `"generate_chat"` or `"generate_chat_batch"`
- **timestamp** — ISO 8601 with local timezone
- **elapsed_seconds** — wall clock time for this call
- **messages** — full message list (or list-of-lists for batch calls)
- **sampling_params** — serialized sampling parameters
- **response/responses** — full response text, finish reason, token counts, metadata
- **prompt_tokens / completion_tokens** — token usage

If the wrapped backend raises, `LoggingBackend` logs the failing prompt payload and request metadata with an `error` field before re-raising. These failed-call records are preserved in JSONL for debugging but are excluded from per-generation truncation statistics.

The `error` payload always includes `type` and `message`. When the underlying exception carries them, it also includes `status_code` (read from the exception or its `response`), `body` (the parsed response JSON, verbatim), `provider_error` (OpenRouter's top-level `error` object on `MalformedResponseError`), and `backend_name` / `model_name` for capability-mismatch errors like `LogprobsNotReturnedError`. For `PartialBatchError` (raised when only part of a concurrent batch failed), the payload also includes `failure_count`, `result_count`, and `failure_samples` — up to five distinct per-item failures grouped by `(type, status_code, message)` with their `count` and `first_index`, each fully serialized through the same path so HTTP bodies of the underlying provider errors survive the rollup.

### ExperimentLogger

Central collector that stores all records and provides:
- **Phase management** — `set_phase(name)` and `phase(name)` context manager for tagging records
- **Filtering** — `get_phase_records(phase)` to retrieve records by stage
- **Annotation** — `annotate_last(phase, ...)` and `annotate_phase_records(phase, annotations)` for post-hoc enrichment (e.g., adding `extracted_answer` to method evaluation records)
- **Event logging** — `log_event(event_type, data)` for non-LLM-call records (e.g., trajectory summaries and rollout summaries). Event records have an `event_type` key instead of `call_type` and are excluded from the resource summary.
- **Resource aggregation** — `compute_resource_summary()` totals token counts, call counts, and timing by phase. Only counts records with a `call_type` key (LLM calls); event records are excluded.
- **Output** — `save_logs(path)` writes JSONL, `save_summary(path, ...)` writes summary JSON

### Integration with scripts

The experiment scripts set phase tags around each pipeline stage:

1. `"trajectory_collection:{backend}"` — during trajectory collection in `collect_dataset.py`. Multi-policy mode tags each backend separately. These phases also emit `"trajectory_summary"` events with return/step/success aggregates for that backend, plus `"skipped_trajectory"` events when per-task reset/step errors are filtered out instead of aborting the whole batch.
2. `"gt:{name}"` — during ground truth estimation in `predict.py`. When rollout reuse is active, LLM calls from `collect_returns()` are logged under the first config's phase in the group. Long-running estimation phases also emit matching progress info log lines while rollouts are being collected. Rollout-based GT phases emit `"rollout_summary"` events with rollout return/step/count aggregates.
3. `"method:{name}"` — during each method's evaluation in `predict.py` (e.g., `"method:llm_direct"`). After evaluation, annotates each method call record with `{"extracted_answer": <value>}`. Eval phases also emit progress info log lines as prediction batches advance.

### Output Files

The three-script workflow produces separate output files for each phase:

**`collect_dataset.py`** produces:
- **`{dataset_path}`** — pickle with evaluation points, trajectory returns, raw trajectory results, config, metadata (timestamped filename). Raw `TrajectoryResult` objects are included so points can be resampled later via `resample_points.py`. Metadata preserves requested `task_indices` and adds `collected_task_indices` for the successful trajectories actually stored. `ranking_points` remain auxiliary and may cover only a subset of `evaluation_points`, such as ranking-only resume runs that sample stored points before ranking collection.
- **`{dataset_path}.logs.jsonl`** — LLM call logs from the collection phase
- **trajectory JSON** (optional, via `collection.trajectories_path`) — observable trajectories (text-only, separate from the pickle)

**`resample_points.py`** produces:
- **`dataset_{timestamp}.pkl`** — resampled dataset with new evaluation points, recomputed trajectory returns, and (optionally) carried-over cleaned trajectory results. Task/count metadata is rewritten to match the kept trajectories, and trajectory-subset metadata refers to source-pickle trajectory indices.

**`predict.py`** produces:
- **`{predictions_dir}/gt/gt_{name}_{timestamp}.json`** — per-GT-method prediction JSON
- **`{predictions_dir}/eval/eval_{name}_{timestamp}.json`** — per-eval-method prediction JSON
- **`{predictions_dir}/predict.logs_{timestamp}.jsonl`** — LLM call logs from prediction phase

**`evaluate.py`** produces:
- **`{output-dir}/summary_{timestamp}.json`** — experiment info, dataset config/metadata, correlations, predicted/GT values

```python
from qval import ExperimentLogger, LoggingBackend

exp_logger = ExperimentLogger()
logging_backend = LoggingBackend(real_backend, exp_logger)

# Use logging_backend everywhere instead of real_backend
# Phase management is done by the scripts

# Save output
exp_logger.save_logs("results/logs.jsonl")
```

## Structured Observations

llenvs provides structured observation fields on the `Observation` dataclass:

- **`ObservationContent`** — a frozen dataclass with `.text` (str) and `.images` (tuple of `ImageContent`). Import from `llenvs.core.state`.
- **`Observation.task: ObservationContent | None`** — static task description, same every turn. Set by adapters at `reset()`, carried forward through `step()`.
- **`Observation.state: ObservationContent | None`** — dynamic per-step observation, changes each turn. Set by adapters at `step()` with updated feedback/game state.

The serialization layer (`methods/serialization.py`) uses these fields when available:

1. **`obs.state` set** → return `obs.state.text`. This is the canonical dynamic state for multi-turn environments (e.g., GEM game feedback with range, turns remaining, etc.). When `obs.state.data` contains `"tool_results"`, formatted tool results are appended after the state text.
2. **`obs.task` set, `obs.state` None** → return `obs.task.text`. For single-turn environments (e.g., reasoning-gym), the task IS the state — the evaluator needs the instance-specific question.
3. **Both None** → legacy fallback using `obs.prompt` / `obs.messages` heuristics.

### Action handling and history

llenvs provides a three-tier action model on `StepResult` and `Transition`:

1. **`action.text`** (tier 1) — raw LLM generation, including thinking tokens.
2. **`extracted_action`** (tier 2) — what the extractor pulled out. Always set when extraction succeeds; may be truncated tail text for invalid actions.
3. **`resolved_action`** (tier 3) — formatted native action (e.g., `"right"` for gym action 2). Only set when the action was valid and mapped to an environment action.

qval uses a priority chain for display: `resolved_action` → `extracted_action` → `strip_thinking(action.text)` → `UNPARSED_ACTION_PLACEHOLDER`. This gives the evaluator the most informative view: clean `"right"` for valid actions, extracted text for invalid ones.

`EvaluationPoint` carries three fields for this:
- **`extracted_action: str | None`** — extractor output (tier 2). Always set when extraction succeeds (the `raw` extractor as composite fallback guarantees this).
- **`resolved_action: str | None`** — formatted native action (tier 3). Comes directly from the llenvs adapter without modification.
- **`history: tuple[HistoryTurn, ...]`** — pre-computed `HistoryTurn(state_text, action_text, thought)` entries from prior transitions in the trajectory. Built at collection time in `collect_evaluation_points()`. The `thought` field is extracted from the raw action text via `extract_thought()` (supports both `<thought>` tags and classic `Thought:/Action:` format).
- **`current_thought: str | None`** — the actor's ReAct-style reasoning for the action at this evaluation point, extracted at collection time. Survives `prepare_display_points()` (which replaces the `Action` object).

`LLMDirectMethod` uses `point.history` for multi-turn prompt context. `build_direct_user_prompt()` uses `action_text_for_display()` for the current action. `prepare_display_points()` uses `action_text_for_display()` for clean action text, or raw `action.text` when `include_actor_thinking=True`. `from_trajectory_result()` reads `extracted_action` from the llenvs transition for observable trajectories.

#### Raw extractor fallback

All environment configs chain `RawGenerationExtractor` (registered as `"raw"`) as the last extractor in a composite. This guarantees `extracted_action` is always set — `raw` never returns `None`. A `truncate_tail` post-cleaner on the composite bounds the extracted text to 256 characters when structured extractors fail. Note that `resolved_action` (tier 3) may still be `None` when the extracted action doesn't map to a valid environment action.

### Tool-enabled serialization

`serialize_action()` handles tool-calling actions: when `action.tool_calls` is non-empty, it appends a `[Tool Calls]` section with formatted calls (using `format_tool_call()` from llenvs). `serialize_observation()` handles tool results: when `obs.state.data` contains `"tool_results"`, it appends a `[Tool Results]` section (using `format_tool_result_data()` from llenvs).

`action_text_for_display()` centralizes the action text selection logic. For tool-calling actions, it uses `serialize_action()` directly (since `resolved_action` and `extracted_action` cannot carry tool call info). For text-only actions, it applies the three-tier priority chain. This function is used by `_build_history()`, `build_direct_user_prompt()`, and `prepare_display_points()`.

## Dependencies

- **llenvs** — environment interface, trajectory types, model backends
- **scipy** — statistical correlation (pearsonr, spearmanr, kendalltau)
- **numpy** — array operations for correlation computation
- **pyyaml** — YAML config file loading
