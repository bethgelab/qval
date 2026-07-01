# Configuration

All configuration is done via frozen dataclasses. The top-level config is `BenchmarkConfig`, which composes the sub-configs. Configs can be constructed programmatically or loaded from YAML files.

## BenchmarkConfig

Top-level configuration shared across all pipeline scripts.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `signal_type` | `SignalType` | `STATE_VALUE` | Type of signal being evaluated: `state_value`, `q_value`, `advantage`, `potential`, `shaped_reward` |
| `correlation_methods` | `tuple[CorrelationMethod, ...]` | `(PEARSON, SPEARMAN, KENDALL_TAU)` | Correlation metrics: `pearson`, `spearman`, `kendall_tau`, `sign_agreement` |
| `estimations` | `tuple[EstimationConfig, ...]` | `(EstimationConfig(),)` | Ground truth estimation configs (one per ground truth to compute) |
| `points_config` | `EvaluationPointsConfig` | (defaults) | Evaluation point collection |
| `batch_size` | `int` | `256` | Max concurrent rollouts/trajectories per lockstep batch |
| `include_actor_thinking` | `bool` | `False` | When `True`, pass raw actor output (including `<think>` traces) to evaluation methods without extraction or cleaning |
| `include_current_thoughts` | `bool` | `False` | When `True`, show the actor's ReAct-style reasoning for the current evaluation point in a dedicated `**Actor's Reasoning:**` section in user prompts. `<thought>` tags are stripped from the displayed action text to avoid duplication |
| `include_history_thoughts` | `bool` | `False` | When `True`, include `[Turn i - Reasoning]` sections for history turns that have an extracted thought. When either thought flag is `True`, the `actor_reasoning_guidance` block is added to system prompts |
| `discount_factor` | `float` | `1.0` | Discount factor γ for trajectory return computation and GT estimation. Applied in `trajectory_return()`, `MCMethod`, and `MCTSMethod`. Value of 1.0 means undiscounted returns |
| `eval_methods` | `tuple[EvalMethodConfig, ...] \| None` | `None` | Structured eval method configs for `predict.py`. Each entry creates one eval-method instance |
| `backends` | `dict[str, BackendConfig]` | `{}` | Named backend registry used by all scripts |
| `default_backend` | `str \| None` | `None` | Default backend name when one is not specified |
| `collection` | `CollectionConfig` | (defaults) | Runtime settings for `collect_dataset.py` |
| `prediction` | `PredictionConfig` | (defaults) | Runtime settings for `predict.py` |

Each `EstimationConfig` produces an independent set of ground truth values. Method predictions are computed once and correlated against each estimation's ground truth independently.

## EstimationConfig

Configuration for a ground truth estimation method.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"mc_mean"` | Human-readable name (used in results and logging) |
| `assumption` | `SignalType` | `STATE_VALUE` | How to interpret method output for GT comparison: `state_value`, `q_value`, `advantage`, `potential`, `shaped_reward` |
| `method` | `str` | `"mc"` | Estimation algorithm: `"mc"` or `"mcts"` |
| `aggregation` | `AggregationMethod` | `MEAN` | How to aggregate rollout returns: `mean` or `max` |
| `num_rollouts` | `int` | `32` | Number of rollouts per estimation point (MC only) |
| `mcts_iterations` | `int` | `100` | Total number of simulation returns (MCTS budget) |
| `mcts_expansion_width` | `int` | `5` | Number of children per expansion (MCTS) |
| `mcts_exploration_constant` | `float` | `1.41` | UCB1 exploration constant (MCTS) |
| `backend_name` | `str \| None` | `None` | Named backend for rollouts (resolved from `backends` dict). `None` uses `default_backend` |
| `policy_name` | `str \| None` | `None` | Name of a registered scripted rollout policy. When set, GT estimation uses `ScriptedBackend` instead of an LLM backend |

**Note:** `rollout_temperature` and `rollout_max_tokens` are no longer supported. Configure rollout sampling under `backends.<name>.sampling` instead.

`backend_name` and `policy_name` are mutually exclusive. Setting both raises a config error.

The `assumption` field determines how method output is interpreted for ground truth comparison:

- **`state_value`** / **`q_value`** / **`advantage`** — direct comparison: GT is computed via MC/MCTS rollouts at the point level.
- **`potential`** — GT is V*(s) via MC/MCTS. When the method produces per-transition F values (`signal_type=shaped_reward`), Φ is extracted via cumulative sum before comparison. STATE_VALUE and POTENTIAL share the same rollout type ("v_s"), so their MC rollouts are reused.
- **`shaped_reward`** — trajectory-level comparison: method predictions are summed per trajectory and correlated with GT trajectory returns. No MC/MCTS rollouts are needed — `method` and other rollout fields are ignored.

The `aggregation` field controls how multiple rollout returns are combined into a single ground truth estimate:

- **`mean`** — average of all returns. Estimates the expected value under the rollout policy.
- **`max`** — maximum return across all rollouts. Approximates the value under an optimal (best-of-N) policy.

The `method` field selects the estimation algorithm:

- **`mc`** — flat Monte Carlo: N independent rollouts from the evaluation state. Simple, parallelizable.
- **`mcts`** — Monte Carlo Tree Search: builds a search tree with UCB1 selection and wide batch expansion. Focuses computation on promising branches. Especially effective with `max` aggregation.

### Scripted rollout policies

For environments with a known policy or exactly-solvable dynamics, GT estimation can use a registered scripted policy instead of an LLM rollout backend. Scripted policies plug in at the `Policy.backend` layer, so MC/MCTS, rollout reuse, and signal-type handling all stay unchanged.

- `policy_name: frozen_lake_4x4` selects a registered policy from `qval.optimal_policies`
- scripted policies use `ScriptedBackend`, ignore sampling params, and run with `turn_info=False`
- FrozenLake scripted policies now merge runtime `make_kwargs` (`desc`, `map_name`, `is_slippery`, `success_rate`, `reward_schedule`) into policy metadata and solve the exact MDP with dynamic programming, so deterministic and slippery variants both use the true transition model
- positive non-terminal rewards (for example `reward_schedule[2] > 0`) are intentionally rejected because the current planner is stationary and does not model the time horizon needed for truly optimal looping behavior
- prediction artifacts persist `policy_name` in GT config metadata for reproducibility

## EvaluationPointsConfig

Parameters for collecting evaluation points from trajectories.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_trajectories` | `int` | `10` | Number of trajectories to collect. When it exceeds the environment's available task count, bounded adapters (e.g. ALFWorld) cycle through the task pool to reach this many trajectories; unbounded adapters (Jericho/Craftax) use unique per-index seeds |
| `max_points_per_trajectory` | `int \| None` | `None` | Max points per trajectory (`None` = all) |
| `task_indices` | `tuple[int, ...] \| None` | `None` | Specific tasks (`None` = first N) |
| `shuffle_tasks` | `bool` | `True` | Shuffle all available task indices and take the first `num_trajectories`. Ensures representative sampling across task types in multi-type environments. Uses the collection seed for reproducibility |
| `sampling_strategy` | `str` | `"random"` | How to select points when `max_points_per_trajectory` is set: `"first"`, `"random"`, `"uniform"` |
| `max_trajectories_to_keep` | `int \| None` | `None` | Cap on trajectories saved after validation; excess trajectories are pruned |
| `early_turns_to_discard` | `int` | `1` | Number of initial transitions per trajectory to exclude from sampling |
| `late_turns_to_discard` | `int` | `1` | Number of final transitions per trajectory to exclude from sampling |

**Turn-range filtering** is applied per trajectory before the sampling strategy. With `early_turns_to_discard=2` and `late_turns_to_discard=3` on a 10-step trajectory, only steps at 0-indexed positions 2–6 are eligible for sampling. If the discards would leave no eligible steps, all transitions are used as a fallback. Both values can also be set per environment in the environment context YAML (overrides the collection config defaults).

Sampling strategies (apply after turn-range filtering, only when `max_points_per_trajectory` is set):

- **`random`** — randomly sample N transitions. Uses `trajectory_index` as deterministic seed for reproducibility.
- **`first`** — take the first N eligible transitions.
- **`uniform`** — evenly spaced across the eligible range.

When `max_points_per_trajectory` is `None`, all eligible transitions are used.

These parameters can be adjusted post-collection via `resample_points.py` when the dataset includes stored trajectory results, avoiding the need to re-collect trajectories. If the stored `trajectory_results` contain failed trajectories from an older or partial artifact, resampling drops those failed entries before recomputing points and returns. `resample_points.py` can also randomly sample a subset of stored trajectories first via `--trajectory-subset-count` and `--trajectory-subset-seed`, which is useful for the trajectory-only dataset pickles emitted by `compute_pass_at_k.py`. When `--unique-tasks` is added, that subset is balanced across all unique tasks instead of sampling raw trajectories directly: task identity prefers stored `task_name` when available and otherwise falls back to `task_index`, so repeated Pass@k samples from the same task collapse to a single candidate group. In that mode, `--trajectory-subset-count` must be a multiple of the number of unique tasks after failed trajectories are dropped, and the implied per-task quota must not exceed the smallest retained task bucket. The resampled dataset metadata records that balanced selection and still reports `trajectory_selection.selected_trajectory_indices` against trajectory positions in the source pickle rather than positions in the cleaned subset after failed trajectories are dropped.

## EvalMethodConfig

Configuration for an evaluation method instance. Allows running the same method type with different prompt strategies and policy assumptions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | (required) | Name for this method instance (used in prediction filenames) |
| `type` | `str` | (required) | Method type: `"llm_direct"`, `"llm_direct_gvl"`, `"llm_verifier"`, `"llm_codegen"`, `"llm_eureka"`, `"llm_ranking"` (ranking requires `signal_type: q_value`), `"sdpo_ranking"`, `"delta_belief_ranking"`, `"baseline_random"`, `"vlm_rm"`, `"vlm_sor"`, `"vip"`, `"liv_img"`, or `"liv_txt"` |
| `signal_type` | `SignalType` | (required) | Type of dense signal this method evaluates. Each method declares its own signal type, enabling mixed-signal-type prediction runs |
| `prompt_preset` | `str \| None` | (required for prompt-based types) | Name of the prompt preset to use (e.g., `"v0"`, `"v1"`, `"v2"`). Required for LLM-based prompt methods; may be omitted for encoder-based methods (`vlm_rm`, `vlm_sor`, `vip`, `liv_img`, `liv_txt`) and `baseline_random`. `baseline_random` ignores this field; if omitted it defaults to `"v0"` for metadata purposes |
| `policy_assumption` | `PolicyAssumption` | `OPTIMAL` | Policy assumption for prompts: `optimal` or `self` |
| `backend_name` | `str \| None` | `None` | Named backend for this method. When unset, uses `default_backend` |
| `num_samples` | `int` | `1` | Number of independent code candidates to generate. Supported by `llm_codegen` and `llm_eureka` |
| `search_iterations` | `int` | `3` | Number of iterative search rounds for `llm_eureka` |
| `judge_num_points` | `int` | `8` | Number of sampled datapoints shown to the judge prompt for `llm_eureka` |
| `num_verifications` | `int` | `1` | Number of repeated prediction passes to average. Supported by `llm_direct`, `llm_direct_gvl`, and `llm_verifier` |
| `prompt_batch_size` | `int` | `1` | Number of evaluation points to pack into one prompt for `llm_direct` or `llm_verifier`. `1` preserves the old one-point-per-prompt behavior. `batch_size` still controls transport chunking for `generate_chat_batch()`. `llm_direct_gvl` does not accept this field |
| `prompt_batch_mode` | `str` | `"packed"` | Batched `llm_direct` prompt protocol. `"packed"` sends one user message containing all datapoints in the prompt group. `"sequential"` sends one datapoint per user turn and keeps prior numeric assistant replies in the conversation. Supported only for `llm_direct` |
| `prompt_grouping` | `str \| None` | `None` | Grouping strategy for `llm_verifier`: `contiguous`, `random`, or `trajectory`. When omitted, `llm_verifier` defaults to `random`. `llm_direct_gvl` accepts only `trajectory` when this field is provided explicitly |
| `criteria` | `list[str] \| None` | `None` | Built-in verifier criterion IDs for `llm_verifier`. `None` means a single criterion-free pass |
| `disclose_step_penalty` | `bool` | `False` | When `True`, append the step penalty description to `reward_description` in this method's prompt. When `False` (default), the method's prompts do not mention the step penalty even if GT rollouts include it |
| `disclose_discount_factor` | `bool` | `False` | When `True`, use the real discount factor value in this method's prompts (e.g., "discounted cumulative reward (γ=0.9)"). When `False` (default), prompts use neutral "cumulative reward" language — neither claiming undiscounted nor revealing the exact γ value |
| `include_next_state` | `bool` | `True` | When `True`, next-state information is included in prompts (direct: **Next State** section; codegen: `next_state` parameter; ranking: per-candidate resulting states). When `False`, next-state is omitted from all prompt types. Only affects signal types with actions (Q_VALUE, ADVANTAGE, SHAPED_REWARD); STATE_VALUE/POTENTIAL are unaffected |
| `disable_prompt_truncation` | `bool` | `False` | When `True`, skip budget-aware prompt truncation for this method even when `min_*_chars` defaults are inherited from the environment context. Required to pair `llm_direct` with `prompt_batch_mode="sequential"` while other methods in the same run keep truncation enabled. No-op for `llm_codegen` (which does not support truncation in any case) |
| `seed` | `int \| None` | `None` | PRNG seed for `baseline_random`. Two runs with the same seed produce identical prediction sequences. Rejected for any other method type |
| `vle_baseline_prompt` | `str \| None` | `None` | `vlm_rm` only: text describing the generic / default environment state. Setting this together with `vle_alpha` enables goal-baseline regularization; leaving both unset gives raw cosine similarity |
| `vle_alpha` | `float \| None` | `None` | `vlm_rm` only: interpolation between raw cosine (α=0) and full line-projection (α=1) in the goal-baseline regularization |
| `vle_negative_goals` | `list[str] \| None` | `None` | `vlm_sor` only (required): list of competing goal strings forming the task set ℒ. The softmax denominator sums over {current goal} ∪ `vle_negative_goals`; duplicates of the current goal are dropped automatically |
| `vle_temperature` | `float \| None` | `None` | `vlm_sor` only (required): softmax temperature τ in `p = softmax(cos(·)/τ)` |
| `vle_beta` | `float \| None` | `None` | `vlm_sor` only: when set, thresholds the softmax probability and returns `1[p > β]` (Baumli et al.'s binary reward); when unset, returns the continuous probability |
| `vle_goal_text` | `str \| None` | `None` | `vlm_rm` / `vlm_sor`: static text goal override. When unset, falls back to `context.task_description`. Mutually exclusive with `vle_goal_per_point` |
| `vle_goal_per_point` | `bool` | `False` | `vlm_rm` / `vlm_sor`: when `True`, read each point's goal from `obs.task.text` (multi-task envs). Mutually exclusive with `vle_goal_text` |
| `vle_image_source` | `str \| None` | auto | `vlm_rm` / `vlm_sor`: `"state"` or `"next_state"`. Auto-picks `"next_state"` for `q_value` signal and `"state"` otherwise |
| `vle_multi_image_strategy` | `str \| None` | `"last"` | `vlm_rm` / `vlm_sor`: how to handle multi-image states — `"last"`, `"first"`, or `"mean"` |
| `vip_goal_source` | `str \| None` | `None` | `vip` only (required): `"trajectory_end"` (use the last image of the point's trajectory) or `"env_image"` (load a static PNG from `vip_env_goal_image_path`) |
| `vip_env_goal_image_path` | `str \| None` | `None` | `vip` only, required when `vip_goal_source="env_image"`: path to a goal-state PNG |
| `vip_multi_image_strategy` | `str \| None` | `"last"` | `vip` only: `"last"` or `"first"` when a state carries multiple images |
| `liv_similarity` | `str \| None` | `"cosine"` | `liv_img` / `liv_txt`: `"cosine"` (LIV paper default) or `"l2"` (VIP-style negative Euclidean) |
| `liv_goal_source` | `str \| None` | `None` | `liv_img` only (required): `"trajectory_end"` or `"env_image"`, same semantics as VIP |
| `liv_env_goal_image_path` | `str \| None` | `None` | `liv_img` only, required when `liv_goal_source="env_image"` |
| `liv_goal_text` | `str \| None` | `None` | `liv_txt` only: static text goal override. Falls back to `context.task_description` when unset. Mutually exclusive with `liv_goal_per_point` |
| `liv_goal_per_point` | `bool` | `False` | `liv_txt` only: per-point goal from `obs.task.text`. Mutually exclusive with `liv_goal_text` |
| `liv_multi_image_strategy` | `str \| None` | `"last"` | `liv_img` / `liv_txt`: multi-image handling, same options as VIP |

When `eval_methods` is set in `BenchmarkConfig`, `predict.py` iterates over these entries. Each entry creates a method instance with its own `signal_type`, `prompt_preset`, and `policy_assumption`, and predictions are saved as `eval/{name}_{timestamp}.json`. The `signal_type`, `prompt_preset`, `prompt_batch_size`, `prompt_batch_mode`, and `num_verifications` are persisted in the prediction artifact config for reproducibility. `llm_direct_gvl` also persists its fixed trajectory-grouping and deterministic-shuffle metadata. `llm_verifier` persists its grouping mode, criteria list, and score-scale identifier. `llm_eureka` persists its `search_iterations` and `judge_num_points` settings. Different eval methods within the same prediction run can target different signal types.

`llm_direct_gvl` is a dedicated GVL-style numeric method for trajectory-context prompting. It keeps the same numeric target types as the rest of the benchmark, but changes prompt construction substantially:

- each prompt targets exactly one datapoint
- the prompt includes all other stored transitions from that datapoint's trajectory as surrounding context
- surrounding context transitions are shown in deterministic shuffled order
- the target transition is rendered separately at the end and is the only transition the model is asked to score
- chronology markers such as `[Step 1/20]` or `[Turn 3/50]` are intentionally omitted
- when `include_next_state: true`, only the target transition includes its next state; surrounding context remains state-action only
- dataset loading fails fast unless `trajectory_results` are present and aligned with every `trajectory_index` / `step_index`
- history and actor-thought sections are intentionally disabled to avoid leaking chronology

`baseline_random` emits one uniform `[0, 1)` value per evaluation point, ignoring all inputs. It exists as a correlation floor: any informed method's reported correlation against ground truth is only meaningful relative to what an uninformed predictor achieves on the same dataset. The method requires no backend, no prompt preset, and no LLM machinery; the only meaningful field is `seed`, which makes the prediction sequence reproducible. The persisted prediction artifact records `method_type: baseline_random`, the seed (when set), and the declared `signal_type` so `evaluate.py` compares it against the matching GT signal. Minimal config:

```yaml
eval_methods:
  - name: random_sv
    type: baseline_random
    signal_type: state_value
    seed: 0
```

## Prompt Preview Script

`scripts/preview_prompts.py` renders representative prompt files without running a full prediction job. For `llm_direct`, it now writes explicit variant directories under `direct/<signal_type>/variants/` for each requested combination of `prompt_batch_mode`, `prompt_batch_size`, and `num_verifications`.

Each variant directory contains:

- `metadata.txt` describing the selected `prompt_batch_mode`, `prompt_batch_size`, `num_verifications`, and prompt-group layout
- `prompt_<i>_system.txt` plus either:
- `prompt_<i>_user.txt` for packed variants and singleton groups
- `prompt_<i>_user_turn_<j>.txt` for sequential variants with multi-point prompt groups

`num_verifications` does not change the prompt text itself. The preview therefore shows the unique prompt groups once and records in `metadata.txt` that the same prompt groups are reused across verification passes and then averaged per datapoint. For sequential variants, the preview writes the per-turn user prompts only; at runtime, `LLMDirectMethod` inserts canonicalized assistant replies between those turns.

Use `--direct-prompt-batch-sizes`, `--direct-prompt-batch-modes`, and `--direct-num-verifications` to choose which `llm_direct` variants to render. The size and verification flags accept comma-separated integer lists such as `--direct-prompt-batch-sizes 1,4 --direct-num-verifications 1,3`. The mode flag accepts `packed`, `sequential`, or both.

`prompt_batch_mode: sequential` is currently incompatible with budget-aware prompt truncation. If the method instance would use `max_prompt_tokens`/token-estimation-based truncation, construction fails fast instead of trying to estimate a growing multi-turn conversation budget. To run a sequential `llm_direct` method alongside methods that use truncation, set `disable_prompt_truncation: true` on the sequential method — this skips budget estimation for that method and suppresses the check, while other methods in the same run continue to truncate.

Because of those prompt constraints, `llm_direct_gvl` hard-validates the following:

- `max_history_turns` must be `0` (omitting it defaults it to `0`)
- `include_current_thoughts` must be `false`
- `include_history_thoughts` must be `false`
- `prompt_batch_size` must not be set
- `prompt_batch_mode` can only be omitted or left at its fixed `"packed"` value
- `prompt_grouping` can only be omitted or set to `"trajectory"`
- the dataset must contain aligned `trajectory_results`; there is no fallback reconstruction path yet

`llm_verifier` is a verifier-style adaptation that predicts one scalar per evaluation point from an ordered score-bin distribution instead of parsing a free-form numeric answer. In v1:

- it supports only `signal_type: state_value` and `signal_type: q_value`
- it requires a backend with generation-time token logprobs
- it requests the top 20 logprobs per token position (matching OpenRouter's documented `top_logprobs` cap and the upstream method)
- it uses fixed score bins `A` through `T`, mapped internally to a normalized `[0, 1]` scale, with the upstream-style banded rubric (`A`, `B-D`, `E-G`, `H-J`, `K-M`, `N-P`, `Q-S`, `T`) embedded in the system prompt
- batched prompts use one score tag per datapoint, e.g. `<score_1>A</score_1>`
- `num_verifications` repeats the same grouped prompts and averages successful predictions
- `criteria` must explicitly name built-in IDs; there is no implicit default criterion set
- if the backend cannot return logprobs at runtime (for example, an OpenRouter route to a provider that drops them for the chosen model), the backend raises `LogprobsNotReturnedError` and `predict.py` skips only this method, logging a `method_skipped` event and continuing with the remaining methods in the config

Built-in `llm_verifier` criterion IDs:

- `terminal_bench_correctness`
- `terminal_bench_error_detection`
- `terminal_bench_efficiency`
- `terminal_bench_resource_usage`
- `frozen_lake_correctness`
- `frozen_lake_hazard_avoidance`
- `frozen_lake_path_optimality`
- `webshop_correctness`
- `webshop_error_detection`
- `webshop_efficiency`
- `webshop_search_discipline`
- `open_apps_correctness`
- `open_apps_error_detection`
- `open_apps_efficiency`
- `open_apps_input_fidelity`
- `alfworld_correctness`
- `alfworld_error_detection`
- `alfworld_efficiency`
- `alfworld_precondition_awareness`

### Multi-sample code generation

When `num_samples > 1` on an `llm_codegen` method, `predict.py` generates N independent Python signal functions in a single batched LLM call (`generate_chat_batch`). Each requested sample index still produces its own prediction artifact, with the sample index embedded in the method-name token (e.g. `eval/codegen-s{i}_qv_{timestamp}.json`) and `sample_index`, `sample_group`, and `num_samples` in config metadata. If a sample aborts because of a recoverable backend/input failure, its artifact is still written with all-`NaN` values and `metadata.generation_aborted: true`; it does not silently disappear. Invalid generated code remains fatal and aborts the run instead of being skipped.

`llm_eureka` is an LLM-judged adaptation of Eureka-style iterative code search. Each iteration generates `num_samples` candidate `signal_function`s, executes every valid candidate on the full point set being evaluated, then asks the same backend to choose the best candidate from compact per-candidate summaries plus a sampled subset of datapoints. The winning code, judge feedback, and any runtime exception summary observed while evaluating the winner are then fed into the next iteration's reflection block, so the LLM can see and avoid the specific failure (e.g. attempts to call unavailable sandbox builtins). Generated functions may return either a scalar float or `(float, dict[str, float])`, where the first float is the actual signal and the optional component dictionary is used only for judge-time diagnostics and reflection and must be built with explicit literal keys — the codegen sandbox does not expose `locals`/`globals`/`vars`/`dir`/`eval`/`exec`, and any reference to them raises `NameError` at runtime. `judge_num_points` controls how many datapoints are shown to the judge; all full-dataset execution statistics still use the complete point set passed to the method. The judge call uses deterministic sampling settings derived from the method sampler, and when prompt budgeting is enabled the sampled datapoints are re-truncated against the fully assembled judge prompt rather than against a direct-prompt approximation.

During evaluation (`evaluate.py`), predictions sharing the same `sample_group` are automatically aggregated: individual per-sample correlations are computed normally, then replaced in the summary by a single group entry containing `aggregate_correlations` (mean, std, min, max, median across samples) and a `per_sample` list with the original per-sample results. Ungrouped methods pass through unchanged.

Use a non-zero `temperature` when generating multiple samples to ensure diversity across code outputs.

### Encoder-based methods: `vlm_rm`, `vlm_sor`, `vip`, `liv_img`, `liv_txt`

These methods score states against a goal using a frozen pretrained vision model. They don't prompt any LLM — the encoder runs under `torch.inference_mode()` and returns a scalar per point. Each encoder is registered in `shared/configs/backends.yaml` with a dedicated `type`:

- `type: huggingface_vle` — CLIP / SigLIP via `transformers.AutoModel` (used by `vlm_rm`, `vlm_sor`). `model` is the HF model ID (e.g. `google/siglip-base-patch16-224`, `openai/clip-vit-large-patch14`).
- `type: vip` — ResNet-50 pretrained on Ego4D (Ma et al. 2022). The checkpoint is vendored: downloaded on first use from `https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt` and cached under `$HF_HOME/vip/`.
- `type: liv` — CLIP-RN50 fine-tuned on EpicKitchens (Ma et al. 2023). The checkpoint is pulled from HuggingFace Hub (`jasonyma/LIV`). Requires the `clip` package from `git+https://github.com/openai/CLIP.git` to build the RN50 architecture.

Per-method differences:

- **`vlm_rm`** (Rocamonde et al. 2023 — VLM-RMs). Text goal, cosine similarity. Raw cosine by default; setting both `vle_baseline_prompt` and `vle_alpha` enables goal-baseline regularization that projects the image embedding onto the line between baseline and goal in embedding space before measuring distance-to-goal. Supports signal types `state_value`, `q_value`, `shaped_reward`.
- **`vlm_sor`** (Baumli et al. 2023 — VLMs as a Source of Rewards). Text goal + negative goals forming the task set ℒ. Returns the softmax probability on the current goal; `vle_beta` flips to the paper's `1[p > β]` thresholded-binary variant. The head auto-deduplicates any negative that equals the current goal so the softmax denominator matches the paper's `Σ_{l' ∈ ℒ}` (each goal counted exactly once).
- **`vip`** (Ma et al. 2022 — Value-Implicit Pre-Training). Image-to-image only. The value at inference is `V(s; g) = -||φ(state) - φ(goal)||₂` (paper eq. 1); `state_value` uses `point.state`, `q_value` uses `point.next_state`, `shaped_reward` returns `V(next; g) - V(state; g)`. Goal image comes from the trajectory's last state (`vip_goal_source: trajectory_end`, matches the paper's training setup) or a pre-rendered PNG (`vip_goal_source: env_image` + `vip_env_goal_image_path`).
- **`liv_img` / `liv_txt`** (Ma et al. 2023 — Language-Image Value learning). CLIP-initialized fine-tune that keeps both encoders usable. `liv_img` takes an image goal (same resolvers as `vip`); `liv_txt` takes a text goal (same resolvers as `vlm_rm`: `liv_goal_text` static override, `liv_goal_per_point` for per-episode `obs.task.text`, or default to `context.task_description`). `liv_similarity: cosine` is the paper's reported default; `l2` flips to the VIP-style negative Euclidean for apples-to-apples comparisons.

Missing-image handling: every encoder-based method marks a point aborted and returns `NaN` for it when any required image (state, next_state for Q/shaped, or goal) is unavailable. The aborted index set is surfaced via `method.last_aborted_indices`, same contract as the LLM-based methods. All three signal types (`state_value`, `q_value`, `shaped_reward`) are supported; `advantage` and `potential` are rejected.

### PromptPreset

Prompt presets compose named building blocks by reference, making the final prompt content fully transparent from the preset definition alone. Every variable text piece in prompt assembly is a named block registered in the block registry (`_BLOCK_REGISTRY` in `qval.prompt_presets`).

**Block registry:** Each block is a `PromptFragment` with `category="vb_preset"`. Blocks with `{signal_name}` placeholders are templates — formatted at prompt-build time via `resolve_block()`. Built-in blocks:

| Block name | Used in | Description |
|---|---|---|
| `role_opener_direct_default` | Direct system prompt opener | `"You are an expert at estimating {signal_name} functions..."` |
| `role_opener_codegen_default` | Codegen system prompt opener | `"You are an expert at writing {signal_name} functions..."` |
| `role_opener_verifier_default` | Verifier system prompt opener | `"You are an expert verifier for reinforcement learning environments..."` |
| `closing_direct_default` | Direct system prompt closing | `"After any reasoning, respond with ONLY a single numeric {signal_name} estimate..."` |
| `codegen_closing_default` | Codegen user prompt closing | `"You may use the collections, itertools, json, math, re, statistics, and string standard library modules..."` |
| `approximation_guidance_v1` | System prompts (after signal def) | Guidance discouraging exhaustive enumeration |
| `codegen_constraints_v1` | Codegen user prompt (before closing) | Constraints prohibiting recursive search, tree expansion |
| `efficiency_guidance_v1` | System prompts (after approximation guidance) | Vague hint that reaching the goal in fewer steps is preferable |
| `actor_reasoning_guidance` | System prompts (when `include_current_thoughts` or `include_history_thoughts` is `True`) | Describes the actor's ReAct-style reasoning provided in the prompt, guiding the evaluator to focus on action quality |
| `closing_direct_answer_tags` | Direct system prompt closing (V2) | `"After any reasoning, provide your numeric {signal_name} estimate as a single number inside <answer> tags..."` |
| `closing_ranking_answer_tags` | Ranking system prompt closing (V2) | `"After any reasoning, output a comma-separated list of action numbers from best to worst inside <answer> tags..."` |

Built-in presets:

- **`v0`** — All required slots use the default blocks, no optional blocks. Produces the standard prompt text.
- **`v1`** — Same required blocks as v0, plus `approximation_guidance_v1`, `codegen_constraints_v1`, and `efficiency_guidance_v1`.
- **`v2`** — Same guidance blocks as v1, but uses `<answer>` tag closing instructions for direct and ranking methods. Codegen closing is unchanged. Methods automatically use tag-aware extractors (tag-first, with fallback to raw numeric/permutation extraction).

`PromptPreset` fields (all block-name references):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | (required) | Unique identifier |
| `role_opener_direct` | `str` | (required) | Block name for the direct method system prompt opener |
| `role_opener_codegen` | `str` | (required) | Block name for the codegen method system prompt opener |
| `role_opener_verifier` | `str` | `"role_opener_verifier_default"` | Block name for the verifier method system prompt opener |
| `closing_direct` | `str` | (required) | Block name for the direct method closing instruction |
| `codegen_closing` | `str` | (required) | Block name for the codegen user prompt closing |
| `approximation_guidance` | `str \| None` | `None` | Block name for guidance after signal type definition, or `None` |
| `codegen_constraints` | `str \| None` | `None` | Block name for codegen user prompt constraints, or `None` |
| `efficiency_guidance` | `str \| None` | `None` | Block name for efficiency hint after approximation guidance, or `None`. Rendered in system prompts only when `MethodContext.efficiency_guidance` is set at runtime |
| `uses_answer_tags` | `bool` | `False` | When `True`, methods use `<answer>` tag extraction and tag-aware user-prompt closings |

`register_preset()` validates that all block-name references point to registered blocks.

### PolicyAssumption

Controls the policy-dependent language in signal type prompts:

- **`OPTIMAL`** — prompts describe values "assuming optimal play thereafter". Use for GT estimation with MC-Max or MCTS.
- **`SELF`** — prompts describe values "assuming you are the agent making all future decisions". The evaluator LLM acts as the policy.

## BackendConfig

Configuration for named backends. All scripts resolve backends by name; no per-script backend construction is supported.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | (required) | Backend type: `openai`, `huggingface`, `vllm`, `vllm_singularity`, `anthropic`, `openrouter`, `litellm`, `codex`, `claude_code` |
| `model` | `str` | (required) | Model name or path. For `litellm`, use litellm's `provider/model` format (e.g. `gemini/gemini-2.5-flash`, or `litellm_proxy/<model>` for a LiteLLM proxy/gateway) |
| `backend_url` | `str` | `"http://localhost:8000/v1"` | URL for OpenAI-compatible backends |
| `api_base` | `str \| None` | `None` | Custom endpoint URL for the `litellm` backend (e.g. a LiteLLM proxy/gateway). `None` lets litellm pick the provider's default endpoint or read `LITELLM_PROXY_API_BASE` for `litellm_proxy/` models. API keys are never configured here — litellm reads the provider's native env var (`LITELLM_PROXY_API_KEY`, `GEMINI_API_KEY`, ...). The YAML parser raises if set on a non-`litellm` backend |
| `device` | `str` | `"auto"` | Device for HuggingFace backend |
| `dtype` | `str` | `"auto"` | dtype for HuggingFace/`vllm`/`vllm_singularity` backends |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory for `vllm` and `vllm_singularity` |
| `max_model_len` | `int \| None` | `None` | Maximum sequence length for `vllm` and `vllm_singularity`. Also enables budget-aware history truncation when `min_action_chars`/`min_observation_chars` are set in the environment config |
| `tensor_parallel_size` | `int` | `1` | Number of GPUs for `vllm` and `vllm_singularity` tensor parallelism. Overridden by `--tensor-parallel-size` CLI flag when set |
| `vllm_kwargs` | `dict` | `{}` | Extra keyword arguments passed to the in-process vLLM `LLM` constructor for the `vllm` backend (e.g., `enforce_eager: true`, `quantization: awq`). No effect on `vllm_singularity` — use `singularity_extra_vllm_args` for `vllm serve` flags there. In YAML, write as a plain dict; internally stored as a tuple-of-tuples for hashability |
| `config_overrides` | `dict` | `{}` | Extra `-c key=value` overrides for the `codex` backend (e.g., `model_reasoning_effort: low`). Ignored by other backend types. In YAML, write as a plain dict; internally stored as a tuple-of-tuples for hashability |
| `claude_code_kwargs` | `dict` | `{}` | Extra keyword arguments forwarded to `ClaudeCodeBackend` for the `claude_code` backend. Useful keys: `effort` (`low`/`medium`/`high`/`xhigh`/`max`), `bare` (skip OAuth and use `ANTHROPIC_API_KEY` only), `tools` (comma-separated tool allowlist; defaults to empty so the backend has no tool access), `permission_mode`, `max_budget_usd`, `fallback_model`, `extra_args` (raw CLI flags as a list). Ignored by other backend types. In YAML, write as a plain dict; internally stored as a tuple-of-tuples for hashability |
| `singularity_sif` | `str \| None` | `None` | Path to the Singularity `.sif` image for the `vllm_singularity` backend. `None` falls back to the `LLENVS_SIF` env var (typically set by sourcing `llenvs/bin/_cluster.sh`). Ignored by other backend types |
| `singularity_binds` | `list[str]` | `[]` | Extra `--bind` specs forwarded to `singularity exec` for the `vllm_singularity` backend. Empty list falls back to `LLENVS_BINDS` (space-separated). In YAML, write as a plain list; internally stored as a tuple of strings for hashability. Ignored by other backend types |
| `singularity_extra_vllm_args` | `list[str]` | `[]` | Extra CLI flags passed to the in-container `vllm serve` for the `vllm_singularity` backend (e.g., `["--swap-space", "4"]`). In YAML, write as a plain list; internally stored as a tuple of strings for hashability. Ignored by other backend types |
| `singularity_startup_timeout` | `float` | `900.0` | Seconds to wait for `vllm serve`'s `/health` endpoint to come up for the `vllm_singularity` backend. Large VLMs with torch.compile warmup can need 1800+. Ignored by other backend types |
| `singularity_cuda_visible_devices` | `str \| None` | `None` | Value for `CUDA_VISIBLE_DEVICES` inside the container for the `vllm_singularity` backend (e.g., `"0,1"`) — pins `vllm serve` to a subset of the job's GPUs so the rest stay free for other workloads on the same node. `None` inherits whatever `CUDA_VISIBLE_DEVICES` the parent process sees. Ignored by other backend types |
| `enable_thinking` | `bool` | `True` | Pass `enable_thinking` to the chat template (HuggingFace/`vllm`). Set `false` to disable thinking tokens. Part of the backend identity **only** for in-process `vllm`/`huggingface`, which bake it into the instance at construction — two such backends with different `enable_thinking` are never shared. For all other types (`vllm_singularity`, API backends) it is not forwarded to the constructed backend (thinking is a per-request concern via `sampling.thinking_budget`/`disable_thinking`), so two backends differing only in `enable_thinking` share a single instance — e.g. a `react` and a `thinking` `vllm_singularity` config for the same model reuse one container instead of spawning two |
| `max_concurrency` | `int` | `64` | Maximum concurrent requests for API backends (openrouter, litellm, openai). For CLI backends (`codex`, `claude_code`), each request spawns a CLI subprocess — keep this small (e.g., `4`) |
| `rate_limit_wait` | `float` | `0.0` | Seconds to wait before retrying after a 429 rate-limit error (openrouter, litellm). `0` uses SDK default backoff |
| `rate_limit_max_retries` | `int` | `2` | Maximum number of rate-limit retries before giving up (openrouter, litellm) |
| `connect_timeout` | `float \| None` | `None` | TCP connect timeout in seconds for API backends. `None` uses SDK defaults (5s). Increase for unreliable networks. For CLI backends (`codex`, `claude_code`), this value is also reused as the per-request subprocess timeout; `None` keeps the underlying backend's 600s default. For `litellm`, this maps onto litellm's single per-request timeout (connect + read) |
| `api_max_retries` | `int \| None` | `None` | Maximum SDK-level retries for transient errors (timeouts, connection errors, 5xx). `None` uses SDK default (2). For `litellm`, maps onto litellm's `num_retries` (off by default — the pipeline's own per-slot transient retry already covers this) |
| `quota_retry_policy` | `str` | `"sleep_and_retry"` | How to handle `QuotaExhaustedError` raised by the backend. `"sleep_and_retry"` (default) transparently sleeps on a fixed schedule (one 5-minute retry followed by five 1-hour retries, i.e., up to 5 h 5 min across 7 attempts) before letting the error propagate. `"abort"` propagates on first occurrence. The CLI backends (`codex`, `claude_code`) raise `QuotaExhaustedError` permissively for any subprocess failure, timeout, or missing-output that doesn't match more specific classifiers — context-limit text becomes `PromptTooLongError`, authentication failures become a plain `RuntimeError`, and Anthropic Usage Policy refusals (text containing "usage policy" or `anthropic.com/legal/aup`, including the `is_error: true` / `subtype: success` shape) become `RefusedByPolicyError`, a deterministic per-input recoverable error that aborts the offending rollout (and therefore its containing point under MC's all-or-nothing rule). The quota retry wrapper does not retry on `RefusedByPolicyError`; only `QuotaExhaustedError` triggers the sleep schedule. Other backends never raise `QuotaExhaustedError` today. Only the wrapper logic is config-driven — the sleep schedule itself is hardcoded to keep the configuration surface small |
| `openrouter_provider` | `OpenRouterProviderPreferences \| None` | `None` | Request-level provider routing for OpenRouter backends. See [OpenRouterProviderPreferences](#openrouterproviderpreferences). The YAML parser raises if set on any non-`openrouter` backend type |
| `sampling` | `BackendSamplingConfig` | (defaults) | Per-call sampling parameters for this backend |

### BackendSamplingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `top_k` | `int \| None` | `None` | Top-k sampling cutoff |
| `max_tokens` | `int \| None` | `2048` | Maximum tokens per generation |
| `thinking_budget` | `int \| None` | `None` | Max thinking tokens per `<think>` block. `vllm` and `huggingface` honor it via in-process logits processors. `openrouter` forwards it as `extra_body.reasoning.max_tokens`, which OpenRouter routes to the upstream provider's native budget knob (Anthropic `thinking.budget_tokens`, Gemini, Qwen) or derives an effort bucket from for OpenAI/Grok-style models — non-reasoning models silently ignore it. `litellm` forwards it as litellm's cross-provider `thinking={"type": "enabled", "budget_tokens": N}` parameter; providers without a budget concept drop it. `openai`/`anthropic`/`codex`/`claude_code`/`vllm_singularity` cannot intervene at the token level and silently ignore this field; a warning is logged at backend creation when it is set on one of those types |
| `thinking_budget_soft_ratio` | `float \| None` | `None` | Soft ratio for thinking budget (0.0–1.0). Honored only by `vllm` and `huggingface` (the soft-pressure logit boost is a logits-processor effect). Has no analogue in OpenRouter's or litellm's reasoning API — when set on `openrouter` or `litellm` together with `thinking_budget`, a warning is logged that the soft-ratio component is ignored even though the hard cap is honored. Silently ignored on the other API backends |
| `early_stopping_suffix` | `bool` | `True` | When `True` and `thinking_budget` is set, pass `DEFAULT_EARLY_STOPPING_SUFFIX` as `thinking_budget_suffix` on `SamplingParams`. Helps the model transition cleanly from reasoning to answering when the thinking budget is exhausted. Honored only by `vllm` and `huggingface` (the suffix is forced via logits intervention). Silently ignored on every API backend, including `openrouter` — the default `True` is not surfaced as a warning to avoid spam |
| `reasoning_effort` | `str \| None` | `None` | Reasoning effort for `openrouter` and `litellm` backends: `xhigh`, `high`, `medium`, `low`, `minimal`, or `none`. Use `none` to explicitly disable reasoning for a backend. For `openrouter` it is sent as `reasoning.effort` in `extra_body`; for `litellm` it is sent as litellm's top-level `reasoning_effort` parameter. Mutually exclusive with `thinking_budget` on both types, because each accepts one of a token budget or an effort level |
| `reasoning_exclude` | `bool \| None` | `None` | OpenRouter-only `reasoning.exclude` value (rejected at load time for `litellm`, which has no equivalent). When true, OpenRouter may still spend reasoning tokens, but reasoning text is omitted from the response. This does not prevent empty visible answers; use `reasoning_effort: none` to disable reasoning |
| `second_elicitation` | `bool` | `False` | When `True`, enable a follow-up LLM call when generation hits MAX_TOKENS. Uses `DEFAULT_EARLY_STOPPING_SUFFIX` as the elicitation suffix, asks the model to follow the formatting instructions specified above exactly, and disables thinking/reasoning for the follow-up call when the backend supports per-call control. Applied by the pipeline backend wrapper to direct methods, MC rollouts, ranking sampling, and other `generate_chat` / `generate_chat_batch` calls |
| `second_elicitation_max_tokens` | `int` | `256` | Token budget for the follow-up call when `second_elicitation` is enabled |

OpenRouter reasoning tokens are counted as completion tokens and returned separately from visible assistant content. A response can therefore have `completion_tokens == max_tokens` and `text == ""` when the model spends the entire output budget on reasoning before producing final content. If `thinking_budget / max_tokens >= 0.75`, qval logs a warning because this is especially likely and the local `thinking_budget_soft_ratio` / `early_stopping_suffix` controls cannot force an OpenRouter model to transition from reasoning to final answer.

For direct-value methods where the visible `<answer>` is mandatory, prefer an explicit no-reasoning OpenRouter configuration:

```yaml
gemma4_26_or_no_reasoning:
    type: openrouter
    model: google/gemma-4-26b-a4b-it
    max_concurrency: 32
    connect_timeout: 30.0
    api_max_retries: 5
    enable_thinking: false
    sampling:
        temperature: 0.2
        top_p: 0.95
        max_tokens: 512
        reasoning_effort: none
```

The same caveat applies to `litellm` backends pointing at reasoning models (reasoning tokens count toward `max_tokens` on most providers); the `thinking_budget / max_tokens >= 0.75` warning fires for `litellm` too.

### LiteLLM backends

`type: litellm` routes requests through the [litellm](https://docs.litellm.ai) SDK, giving access to 100+ providers — including LiteLLM proxy/gateway servers — through one backend type. Auth is environment-only: litellm reads the provider's native env var based on the model prefix (`GEMINI_API_KEY` for `gemini/...`, `LITELLM_PROXY_API_KEY` for `litellm_proxy/...`, etc.).

```yaml
qwen36_litellm_16k:
    type: litellm
    model: "litellm_proxy/Qwen/Qwen3.6-35B-A3B"
    api_base: "https://llm.mlcloud.uni-tuebingen.de"
    max_concurrency: 16
    rate_limit_wait: 61
    connect_timeout: 30.0
    api_max_retries: 5
    enable_thinking: true
    sampling:
        temperature: 0.6
        top_p: 0.95
        top_k: 20
        max_tokens: 16384
```

Notes:

- The llenvs `LiteLLMBackend` passes `drop_params=True`, so request parameters a provider doesn't support (e.g. `presence_penalty` on Anthropic) are silently dropped instead of erroring. Logprobs are the exception: a logprobs request that comes back empty raises `LogprobsNotReturnedError`.
- `thinking_budget` and `reasoning_effort` work as described in [BackendSamplingConfig](#backendsamplingconfig); `reasoning_exclude` and `openrouter_provider` are rejected at load time.

### OpenRouterProviderPreferences

Per-backend provider-routing preferences for `type: openrouter` backends. Forwarded on every chat-completion call as the OpenRouter API's `provider` request-body object (via the OpenAI SDK's `extra_body` kwarg). Fields map 1:1 to [OpenRouter's provider-routing schema](https://openrouter.ai/docs/features/provider-routing). All fields are optional; `None` means unset (omitted from the request).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `order` | `list[str] \| None` | `None` | Provider names in preference order. OpenRouter tries them in sequence |
| `only` | `list[str] \| None` | `None` | Whitelist — restrict routing to these providers |
| `ignore` | `list[str] \| None` | `None` | Blacklist — exclude these providers |
| `allow_fallbacks` | `bool \| None` | `None` | When `false`, fail the request rather than routing outside `order`/`only` |
| `require_parameters` | `bool \| None` | `None` | When `true`, only route to providers supporting every requested parameter |
| `data_collection` | `str \| None` | `None` | `"allow"` or `"deny"` — forbid routing to providers that log prompts |
| `quantizations` | `list[str] \| None` | `None` | Filter providers by quantization variant (e.g. `["fp8", "bf16"]`) |
| `sort` | `str \| None` | `None` | Routing priority: `"price"`, `"throughput"`, or `"latency"` |

Example:

```yaml
qwen36_or_4k:
    type: openrouter
    model: "qwen/qwen3.6-plus-preview:free"
    max_concurrency: 16
    rate_limit_wait: 61
    openrouter_provider:
        only: ["DeepInfra"]
        allow_fallbacks: false
        sort: "throughput"
    sampling:
        temperature: 0.6
        max_tokens: 4096
```

Notes:

- Internally, list fields are stored as tuples so `BackendConfig` stays hashable for the prediction backend cache. An explicit empty list (`only: []`) round-trips to an empty tuple and is forwarded to OpenRouter as an empty JSON array (distinct from `None`, which is omitted).
- YAML parsing rejects unknown keys and invalid `sort`/`data_collection` enum values at load time.
- Setting `openrouter_provider` on any non-`openrouter` backend raises `ValueError` at load time.
- Two `BackendConfig`s that differ only in `openrouter_provider` receive separate `OpenRouterBackend` instances (the prediction backend cache treats the field as part of the identity).
- Model-slug shortcuts such as `:nitro` or `:floor` in `model:` coexist with this block; the shortcut steers a slug-level preset while `openrouter_provider` further constrains per-request routing.
- For ranking sampling (`collect_dataset.py`), the single shared `SamplingParams` requires uniform `openrouter_provider` across all ranking backends — mismatched settings raise at runtime. When a ranking backend has `openrouter_provider` set, `collection.ranking_sampling` must also be configured.

Collection and prediction logs include per-phase `stats` events derived per generation (not per batch call). These events report `total_generations`, `truncated_generations`, `truncated_generations_pct`, `finish_reason_counts`, token totals, and the truncation-relevant generation settings observed in that phase such as `max_tokens`, `thinking_budget`, `thinking_budget_suffix`, `second_elicitation`, and backend `enable_thinking`. When a phase mixes multiple sampling configurations, the event records `sampling_settings_variants` instead of a single `sampling_settings` object.

The logs also include outcome summary events. Collection emits `trajectory_summary` events per collection backend plus one overall `collection_summary` event with trajectory return stats, step-count stats, `success_count`, `success_rate`, `num_ranking_points`, `num_skipped_trajectories`, and `completed` (boolean indicating whether the run finished successfully or was aborted). When collection is interrupted by an error, the `collection_summary` is still emitted with `completed: false` and an `abort_reason` field describing the exception. When a trajectory is skipped because collection recovered from a task-specific reset or step error, collection also emits a `skipped_trajectory` event with the `task_index`, optional `task_name`, `error_kind`, and `error` text. Prediction emits `rollout_summary` events for rollout-based GT methods with rollout return stats, rollout step-count stats, total rollout count, and rollouts-per-point stats. These summary events are separate from the truncation/resource `stats` events.

## CollectionConfig

Runtime settings for `collect_dataset.py` (all optional).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adapter` | `str` | `"reasoning_gym"` | Environment adapter: `reasoning_gym`, `gem`, `agentgym`, `gymnasium`, `alfworld`, `harbor` |
| `env_name` | `str \| None` | `None` | Optional override for env name |
| `backend_names` | `tuple[str, ...] \| None` | `None` | Backend names from `backends` for collection (multi-policy) |
| `max_steps` | `int \| None` | `None` | Max steps per episode |
| `seed` | `int` | `42` | Environment seed |
| `make_kwargs` | `dict \| None` | `None` | Extra adapter-specific kwargs (gymnasium `make()`, native AlfWorld options like `task_types`, Harbor settings like `exec_timeout`/`start_timeout`, `command_soft_timeout`, `text_exec_mode: tmux_session`, `tmux_bootstrap_if_missing: true`, `environment_type: podman-hpc` for single-container or supported compose-backed Harbor tasks, etc.) |
| `step_penalty` | `float \| None` | `None` | Per-step penalty for gymnasium envs |
| `turn_info` | `bool` | `False` | Inject `[Turn N/max]` into observations |
| `context_name` | `str \| None` | `None` | Environment context name for `shared/configs/environments/` |
| `contexts_dir` | `str` | `"shared/configs/environments/"` | Contexts directory |
| `system_prompt` | `str \| None` | `None` | Override system prompt for collection. When unset, collection first uses `system_prompt_file` from the environment context (if present), then composes a prompt from the adapter's registered env prompt, the prompting scheme instructions, and scheme examples. Falls back to the raw adapter prompt for unregistered adapters |
| `trajectories_path` | `str \| None` | `None` | Optional path to save trajectory JSON |
| `harbor_state_capture` | `str` | `"replay"` | Harbor-only state capture mode. `replay` keeps the current replay-based restore path. `snapshot_exact` captures exact runtime checkpoints during live collection |
| `runtime_probing` | `bool` | `False` | Enable runtime probing during Harbor collection. Captures container runtime state (processes, mounts, listening ports, staging content) at each step and annotates states with filesystem-restore risk signals |
| `discard_runtime_probe_risky_trajectories` | `bool` | `True` | When Harbor runtime probing flags a trajectory as restore-risky, discard the whole trajectory before point extraction and ranking collection. Set to `False` to keep risky trajectories and use the probe annotations only as diagnostics |
| `debug` | `bool` | `False` | Enable verbose collection debug logging for `qval` and `llenvs` during `collect_dataset.py`. Useful for diagnosing slow generation, environment stepping, Harbor probing, and verifier execution |
| `ranking_actions` | `int \| None` | `None` | Target maximum candidate actions per ranking point (k). `None` disables ranking collection |
| `ranking_sampling_mode` | `str` | `"llm"` | `"llm"` samples alternatives from the actor backend; `"manual"` uses a registered action sampler |
| `ranking_backend_names` | `tuple[str, ...] \| None` | `None` | Optional backend names used only for LLM ranking sampling. When set, collection backends are closed before ranking sampling starts, and these backends are created just for ranking collection |
| `ranking_sampling_strategy` | `str` | `"avoid_seen"` | LLM ranking sampling strategy. `"avoid_seen"` asks each auxiliary slot not to repeat already accepted actions; `"independent"` keeps the old actor-style prompt path |
| `ranking_sampler_name` | `str \| None` | `None` | Name of the manual action sampler (required when mode is `"manual"`, e.g. `"frozen_lake"`) |
| `ranking_sampling` | `BackendSamplingConfig \| None` | `None` | Sampling parameters for extra LLM-sampled actions (only used with `"llm"` mode) |
| `restore_concurrency` | `int` | `16` | Maximum concurrent restore-based ranking validations for non-pure environments. `1` preserves sequential restore+step validation. Has no effect on pure-step environments or LLM API batching |

Ranking prediction runs require these ranking points; ranking GT runs for any non-SHAPED_REWARD estimation config and always computes Q-values via MC rollouts. When a matching Q_VALUE eval-point GT exists, ACTOR_PRIMARY candidate Q-values are reused from the eval GT and only alternative candidates require fresh rollouts. When no matching Q_VALUE GT exists, all candidates (including ACTOR_PRIMARY) are rolled out from scratch. To enable ranking collection, set `collection.ranking_actions >= 2` (and choose `ranking_sampling_mode`). Collection will try to gather up to `k` unique candidates per point, keep partial points with `2..k` candidates when that is all the environment or sampler can provide, and drop points with fewer than 2 candidates. Shared datasets can therefore include `ranking_points` and still be reused for non-`q_value` experiments; those prediction runs ignore ranking points instead of recomputing auxiliary ranking GT.

Each stored ranking candidate includes explicit provenance:

- `ACTOR_PRIMARY` marks the original actor-sampled action from the trajectory.
- `RANKING_MANUAL` marks an auxiliary action produced by a registered manual sampler.
- `RANKING_LLM` marks an auxiliary action produced by separate LLM sampling during ranking collection.

All ordinary methods and signal types, including ordinary `q_value` prediction runs, continue to predict only the actor-sampled actions from `evaluation_points`. The extra ranking candidates are used only for ranking-related evaluation and auxiliary GT Q-value computation.

When ranking collection uses LLM sampling:

- dataset metadata records the ranking-sampling strategy
- if `ranking_backend_names` is set, metadata records that explicit backend list
- otherwise metadata records the fallback single backend name (the last collection backend)
- multiple ranking backends are assigned deterministically in round-robin order by auxiliary candidate slot
- `batch_size` still controls LLM generation chunking, while `restore_concurrency` separately controls restore-based candidate validation for non-pure environments

With `ranking_sampling_strategy: "avoid_seen"`, ranking collection runs slot-by-slot:
slot 1 avoids the primary actor action, slot 2 avoids the primary action plus
the accepted slot-1 action, and so on. Points are still sampled in parallel
within each slot. With `ranking_sampling_strategy: "independent"`, ranking
collection keeps the older actor-style prompt path and does not inject the
avoid-list constraint.

When `ranking_backend_names` is set, `collect_dataset.py` closes the collection
backends after trajectory collection, creates the ranking backends only for the
ranking phase, and closes them again as soon as ranking collection finishes.

To use a dedicated no-thinking ranking sampler, define ordinary backend entries with `enable_thinking: false` and reference them through `ranking_backend_names`.

For Harbor datasets:

- `harbor_state_capture: replay` is the default and preserves the existing replay-based MC restore path.
- `harbor_state_capture: snapshot_exact` is opt-in and currently targets Harbor tasks running through `environment_type: podman-hpc`.
- Snapshot-mode Harbor collection now prefilters the Harbor task set before rollout starts. If task indices are chosen implicitly, unsupported tasks are skipped and later eligible tasks are used to preserve the requested trajectory count when possible. If `points_config.task_indices` explicitly names unsupported Harbor tasks, collection fails fast.
- Snapshot artifacts are stored next to the dataset pickle and `predict.py` automatically switches Harbor GT rollouts to exact restore for datasets collected in snapshot mode.
- Snapshot-mode dataset metadata records the Harbor task filtering result, and `scripts/pipeline/report_harbor_snapshot_eligibility.py` prints the same eligibility summary without starting collection.
- Replay validation only applies to replay-mode Harbor datasets.
- The provided TerminalBench collection configs keep replay validation disabled by default. For those runs, `runtime_probing: true` plus `discard_runtime_probe_risky_trajectories: true` is the primary replay-safety path.
- When replay validation is disabled, collection skips all post-collection replay-probe capture. `ReplaySpec.probe_outputs` remains unset.
- **Ranking collection on Harbor**: When `ranking_actions` is set, each candidate action requires stepping the environment. Since Harbor environments are non-pure (`pure_step=False`), each candidate step creates a fresh container, restores to the evaluation point's state, steps, then tears down. This uses the same restore mode as the main collection (replay or snapshot_exact). Candidate-validation containers skip runtime probing and snapshot capture to minimize overhead, and `collection.restore_concurrency` controls how many of these restore+step validations can run in parallel. This is separate from LLM request chunking via `batch_size`.

### Harbor on HPC clusters

Harbor environments support two HPC container runtimes via `environment_type` in `make_kwargs`:

**podman-hpc** — for clusters with podman (no Docker):

```yaml
make_kwargs:
  text_exec_mode: tmux_session
  tmux_bootstrap_if_missing: true
  command_soft_timeout: 60
  environment_type: podman-hpc
  podman_command: podman  # if the binary is "podman" rather than "podman-hpc"
```

Supports single-container tasks, compose-backed tasks, and exact checkpoint snapshots.

**apptainer-hpc** — for clusters with Apptainer/Singularity (no Docker or podman):

```yaml
make_kwargs:
  text_exec_mode: tmux_session
  tmux_bootstrap_if_missing: true
  command_soft_timeout: 60
  environment_type: apptainer-hpc
  apptainer_command: apptainer  # or "singularity"
  sif_cache_dir: /shared/sif_cache
  rootfs_mode: sandbox
```

- Single-container tasks only (compose tasks are filtered out during collection)
- SIF images must be pre-built on a login node and placed in `sif_cache_dir` (shared filesystem)
- `rootfs_mode: sandbox` skips the overlay probe and always uses writable sandbox copies; use this when your cluster's overlay path is known not to work
- Overlay mode keeps `/app` and `/tests` writable via host-backed binds; sandbox mode uses the writable rootfs directly
- Uses `--cleanenv --contain --no-home` for host isolation
- `text_exec_mode: tmux_session` runs text-mode TerminalBench steps inside a persistent tmux-backed shell. Shell-local state (`cd`, `export`, background jobs, virtualenv activation) persists across commands instead of resetting each step
- The tmux-session shell exports a small noninteractive baseline for common prompt-prone tools: `DEBIAN_FRONTEND=noninteractive`, `DEBCONF_NONINTERACTIVE_SEEN=true`, `TZ=Etc/UTC`, `APT_LISTCHANGES_FRONTEND=none`, `NEEDRESTART_MODE=a`, and `GIT_TERMINAL_PROMPT=0`
- When `text_exec_mode: tmux_session`, collection runs a one-time tmux session preflight over the selected Harbor task set before model sampling starts. This fails early if any chosen image cannot start the session
- `tmux_bootstrap_if_missing: true` attempts a bounded package-manager install of `tmux` inside the task image when it is absent. For replay-heavy runs, preinstalled `tmux` is still preferred because bootstrap cost is paid again on fresh-container restores
- `runtime_probing: true` records filesystem-restore risk signals. With the default `discard_runtime_probe_risky_trajectories: true`, Harbor trajectories flagged by these probes are removed before dataset point extraction and ranking collection
- `command_soft_timeout` sets the per-command recoverable timeout for Harbor text mode. Live model-issued commands that exceed this limit are interrupted with `Ctrl-C`, then `Ctrl-\`, then a TUI-oriented escape sequence if needed. If the shell recovers, the timeout is recorded as a normal observation
- `trajectory_timeout` sets the live per-trajectory wall-clock budget for Harbor text mode. The default is `900`, and task `recommended_timeout_sec` can only tighten that budget. Replay, snapshot restore, replay validation, and replay-cache filesystem restores do not consume this live budget; restored continuations start from a fresh live budget after restore completes
- Harbor replay, restore, and replay validation use the same `command_soft_timeout` as normal collection. If a replayed command times out and recovery succeeds, replay continues with the timeout observation preserved in message history. Probe commands (run out-of-band, not via `env.step`) use the hard `exec_timeout`
- When replay validation is enabled, its probe commands run out-of-band against the restored runtime rather than through `env.step(...)`, so they do not consume episode steps or trigger Harbor verifier/truncation side effects
- Tasks requiring `allow_internet: false` are filtered (Apptainer uses host networking)
- `singularity-hpc` is accepted as an alias and normalized to `apptainer-hpc`
- Supports filesystem checkpoint/restore via tar-based sandbox snapshots (`export_checkpoint`/`restore_checkpoint`). Only available in sandbox mode
- `pid_namespace: true` enables `--pid` namespace isolation for the container, required for meaningful process-level runtime probing
- The prediction-time replay cache (`harbor_replay_cache: true`) uses these filesystem checkpoints to avoid redundant replay: the first GT rollout from a given state replays the full trajectory and caches the resulting rootfs; the remaining N-1 rollouts restore from the cached tar
- This cache only restores filesystem state. It does not reconstruct process state or persistent shell-session state such as tmux-backed working directories, exported variables, background jobs, or shell-local functions. Keep it disabled for `text_exec_mode: tmux_session`
- If a filesystem checkpoint export fails, prediction continues by falling back to normal replay for that state instead of aborting the run; the failure only disables that cache entry

These kwargs flow through to collection and prediction automatically via the environment context. During collection, a runtime eligibility filter prunes tasks that the selected runtime cannot handle.

## PredictionConfig

Runtime settings for `predict.py` (all optional).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adapter` | `str \| None` | `None` | Environment adapter (used for GT estimation) |
| `gt_backend_name` | `str \| None` | `None` | Backend name for GT estimation |
| `eval_backend_name` | `str \| None` | `None` | Backend name for eval methods |
| `max_history_turns` | `int \| None` | `None` | Max history turns in collection and prediction prompts. When set, only the last N turns are shown to the model, with a truncation notice prepended to the history section |
| `context_name` | `str \| None` | `None` | Environment context name for `shared/configs/environments/` |
| `contexts_dir` | `str` | `"shared/configs/environments/"` | Contexts directory |
| `system_prompt` | `str \| None` | `None` | Override system prompt for GT rollouts. When unset, prediction falls back to `system_prompt_file` from the environment context (if present), then the adapter default prompt, then the dataset metadata prompt |
| `make_kwargs` | `dict \| None` | `None` | Extra adapter-specific kwargs (gymnasium `make()`, native AlfWorld options like `task_types`, Harbor settings like `exec_timeout`/`start_timeout`, `command_soft_timeout`, `text_exec_mode: tmux_session`, `tmux_bootstrap_if_missing: true`, `environment_type: podman-hpc` for single-container or supported compose-backed Harbor tasks, etc.) |
| `cli_backend_tmux_prompt_hardening` | `bool` | `False` | When `True`, CLI-backed Harbor GT rollouts (`codex` or `claude_code`) running with `text_exec_mode: tmux_session` append extra system-prompt guidance that the model is not in a normal CLI tool session and must emit one non-interactive shell command for the external tmux runner. This guidance is included in the GT rollout generation hash |

## Programmatic Usage

```python
from qval import (
    BenchmarkConfig,
    EstimationConfig,
    EvaluationPointsConfig,
    SignalType,
    CorrelationMethod,
    AggregationMethod,
)

config = BenchmarkConfig(
    signal_type=SignalType.Q_VALUE,
    correlation_methods=(CorrelationMethod.SPEARMAN,),
    estimations=(
        EstimationConfig(
            name="mc_mean",
            assumption=SignalType.Q_VALUE,
            aggregation=AggregationMethod.MEAN,
            num_rollouts=64,
            backend_name="rollout",
        ),
        EstimationConfig(
            name="mc_max",
            assumption=SignalType.Q_VALUE,
            aggregation=AggregationMethod.MAX,
            num_rollouts=64,
            backend_name="rollout",
        ),
    ),
    points_config=EvaluationPointsConfig(
        num_trajectories=20,
        max_points_per_trajectory=5,
        task_indices=(0, 1, 2, 3, 4),
        sampling_strategy="random",
    ),
    batch_size=256,  # max concurrent rollouts per lockstep batch
)
```

## YAML Configuration

Configs can be loaded from YAML files using `BenchmarkConfig.from_yaml()` or constructed from dictionaries using `BenchmarkConfig.from_dict()`.

### Loading from YAML

```python
from qval import BenchmarkConfig

config = BenchmarkConfig.from_yaml("configs/experiment.yaml")
```

### Loading from a dictionary

```python
config = BenchmarkConfig.from_dict(
    {
        "signal_type": "q_value",
        "backends": {"eval": {"type": "openai", "model": "gpt-4o"}},
        "default_backend": "eval",
        "collection": {"backend_names": ["eval"]},
        "estimations": [
            {"name": "mc_mean", "num_rollouts": 64, "backend_name": "eval"}
        ],
    }
)
```

## Pipeline Configs

The pipeline uses step-specific config files plus a shared backend registry:

- `shared/configs/backends.yaml` — shared backend registry (required)
- pass@k config for `scripts/pipeline/compute_pass_at_k.py` (optional step 0)
- test-time scaling config for `scripts/pipeline/test_time_scaling.py` (optional online guided-action experiment)
- collection config for `scripts/pipeline/collect_dataset.py`
- prediction config for `scripts/pipeline/predict.py`
- evaluation config for `scripts/pipeline/evaluate.py`

### Backends Registry

```yaml
backends:
    eval:
        type: openai
        model: Qwen/Qwen3-8B
        backend_url: http://localhost:8000/v1
        enable_thinking: true
        sampling:
            temperature: 0.0
            max_tokens: 2048
    rollout:
        type: openai
        model: Qwen/Qwen3-32B
        backend_url: http://localhost:8001/v1
        enable_thinking: true
        sampling:
            temperature: 1.0
            max_tokens: 2048

default_backend: eval
```

### Pass@k Config (compute_pass_at_k)

```yaml
backends_config_path: shared/configs/backends.yaml
output_path: data/pass_at_k/frozen_lake_4x4.json

batch_size: 256

context_name: frozen_lake_4x4
backend_name: eval
seed: 42

samples_per_task: 20
k_values: [1, 5, 10, 20]
num_tasks: 10
shuffle_tasks: true
max_aborted_points: 0.1
```

`PipelinePassAtKConfig` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `backends_config_path` | `str` | `"shared/configs/backends.yaml"` | Path to the backends registry YAML |
| `output_path` | `str \| None` | `None` | Path for the JSON output file |
| `batch_size` | `int` | `256` | Maximum concurrent trajectories per batch |
| `context_name` | `str \| None` | `None` | Environment context YAML name (required) |
| `contexts_dir` | `str` | `"shared/configs/environments/"` | Directory containing environment context YAMLs |
| `backend_name` | `str \| None` | `None` | Single backend name to evaluate |
| `seed` | `int` | `42` | Random seed for task shuffling |
| `system_prompt` | `str \| None` | `None` | Override system prompt |
| `make_kwargs` | `dict \| None` | `None` | Extra kwargs passed to the environment adapter |
| `max_history_turns` | `int \| None` | `None` | Truncate history to last N turns |
| `debug` | `bool` | `False` | Enable verbose debug logging |
| `samples_per_task` | `int` | `20` | Number of independent runs per task (N) |
| `k_values` | `tuple[int, ...]` | `(1, 5, 10, 20)` | Which k values to compute Pass@k for |
| `num_tasks` | `int \| None` | `None` | Number of tasks to evaluate (None = all available). If it exceeds the environment's available task count, it is capped at that count (with a warning) for bounded adapters such as ALFWorld; unbounded adapters (Jericho/Craftax) use `range(num_tasks)` |
| `task_indices` | `tuple[int, ...] \| None` | `None` | Explicit task indices (overrides num_tasks/shuffle) |
| `shuffle_tasks` | `bool` | `True` | Whether to shuffle task order before selection |
| `max_aborted_points` | `float` | `0.1` | Abort after saving partial artifacts if the fraction of dropped trajectories exceeds this threshold. Dropped trajectories are excluded from Pass@k denominators. Override via `--max-aborted-points` |
| `success_threshold` | `float` | `1.0` | Minimum cumulative `reward_signal_name` return for a trajectory to count as a Pass@k success. Matches binary-reward environments exactly at the default. Raise this for dense-reward environments (Craftax `craftax_reward`, Jericho `game_score`) where "outcome reward ≥ 1" is not the right success criterion. Override via `--success-threshold` |
| `cli_backend_tmux_prompt_hardening` | `bool` | `False` | When `True`, CLI-backed Harbor Pass@k runs (`codex` or `claude_code`) using `text_exec_mode: tmux_session` append extra system-prompt guidance reminding the model that it is not in a normal CLI tool session and must emit one non-interactive shell command for the external tmux runner. Mirrors the prediction-step flag |

### Test-Time Scaling Config (test_time_scaling)

```yaml
backends_config_path: shared/configs/backends.yaml
output_path: data/test_time_scaling/frozen_lake_4x4.json

context_name: frozen_lake_4x4
actor_backend_name: actor
seed: 42

samples_per_task: 5
num_tasks: 10
shuffle_tasks: true
success_threshold: 1.0

guided_actor:
  k: 8
  scorer:
    name: direct_q
    type: llm_direct
    signal_type: q_value
    prompt_preset: v1
    backend_name: evaluator
    include_next_state: true

self_consistency:
  mode: open_ended       # or `discrete`
  backend_name: selector # required for open_ended; the selector LLM
  max_candidate_chars: 4000
```

`PipelineTestTimeScalingConfig` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `backends_config_path` | `str` | `"shared/configs/backends.yaml"` | Path to the backends registry YAML |
| `output_path` | `str \| None` | `None` | Path for the JSON summary output file. Per-strategy trajectory pickle/JSON siblings are written next to it |
| `batch_size` | `int` | `256` | Batch size for scorer method calls |
| `context_name` | `str \| None` | `None` | Environment context YAML name (required) |
| `contexts_dir` | `str` | `"shared/configs/environments/"` | Directory containing environment context YAMLs |
| `actor_backend_name` | `str \| None` | `None` | Backend used to sample candidate actions. When unset, uses the backend registry default |
| `guided_actor.k` | `int` | `1` | Number of candidate actions sampled at each environment step |
| `guided_actor.scorer` | `EvalMethodConfig \| None` | `None` | Action scorer used to select among the sampled candidates. Accepts **any** benchmarked method type (individual-prediction or ranking-based — see below); its `signal_type` must be `q_value`, `advantage`, or `shaped_reward` (action-conditioned) |
| `self_consistency` | `TestTimeScalingSelfConsistencyConfig \| None` | `None` | Adds a compute-matched self-consistency baseline strategy (samples `guided_actor.k` candidates per step and selects by consensus). Requires `guided_actor.k > 1`. See the table below |
| `seed` | `int` | `42` | Random seed for task shuffling and fallback candidate selection |
| `system_prompt` | `str \| None` | `None` | Override system prompt |
| `make_kwargs` | `dict \| None` | `None` | Extra kwargs passed to the environment adapter |
| `max_history_turns` | `int \| None` | `None` | Truncate actor and scorer history to the last N turns unless the scorer overrides it |
| `debug` | `bool` | `False` | Enable verbose debug logging |
| `samples_per_task` | `int` | `1` | Number of independent guided trajectories per task |
| `num_tasks` | `int \| None` | `None` | Number of tasks to evaluate (None = all available). If it exceeds the environment's available task count, it is capped at that count (with a warning) for bounded adapters such as ALFWorld; unbounded adapters (Jericho/Craftax) use `range(num_tasks)` |
| `task_indices` | `tuple[int, ...] \| None` | `None` | Explicit task indices (overrides num_tasks/shuffle) |
| `shuffle_tasks` | `bool` | `True` | Whether to shuffle task order before selection |
| `max_aborted_points` | `float` | `0.1` | Abort if the fraction of skipped trajectories for a strategy exceeds this threshold |
| `success_threshold` | `float` | `1.0` | Minimum cumulative `reward_signal_name` return for a trajectory to count as successful |
| `cli_backend_tmux_prompt_hardening` | `bool` | `False` | Mirrors the Pass@k and prediction-step prompt hardening flag for CLI-backed Harbor runs |
| `max_concurrent_trajectories` | `int` | `32` | Wave size: how many trajectories are rolled out concurrently within a strategy, batching their per-step actor (and dense-signal scorer) generations into single calls. `1` reproduces serial rollout. For restore-based environments this is also the number of persistent environment instances created per wave, so lower it when each instance is a container |
| `restore_concurrency` | `int` | `16` | Maximum concurrent restore-based candidate/main steps for non-pure environments (bounds the transient environment instances opened while scoring `include_next_state` candidates). Ignored for pure-step environments |
| `resume` | `bool` | `True` | Reuse completed trajectories from a prior crashed/aborted run of the same config. Each strategy keeps a rolling checkpoint at `{output_stem}_{strategy}_{fp16}.checkpoint.pkl` next to the outputs, keyed by a fingerprint of every content-affecting input; on restart only missing trajectories run. Requires `output_path` (or `--output-dir`). The `--fresh` CLI flag disables resume and removes the current config's checkpoints |
| `harbor_replay_cache` | `bool` | `False` | Harbor-only. Wraps the per-candidate replay restore in a filesystem cache so the `k` candidates restored from the same state at a step share one replay instead of replaying `k` times — a large saving when a next-state scorer steps every candidate. Only active on the `apptainer-hpc` runtime with `replay` restore (the only mode test-time scaling uses); ignored otherwise. Do not combine with `text_exec_mode: tmux_session`, whose shell-session state the filesystem cache cannot restore |

The actor backend and scorer backend are configured independently: `actor_backend_name` selects the model that samples actions, while `guided_actor.scorer.backend_name` selects the evaluator model. The script always runs a `baseline_k1` strategy that samples one action per step with the actor backend and no scorer. If `guided_actor.k > 1` or a scorer is configured, it also runs a `guided` strategy. If `self_consistency` is configured it also runs a `self_consistency` strategy with `k = guided_actor.k` (matched compute). Sampling `k > 1` and choosing uniformly at random is marginally equivalent to a single sample from the actor distribution, so the primary random baseline is `k=1`; `baseline_random` exists only as a diagnostic scorer.

When `guided_actor.scorer.include_next_state` is true, each candidate action is stepped before scoring so the scorer can see `(state, action, next_state)`. This requires a pure-step environment or adapter restore support. On non-pure adapters (e.g. Harbor/TerminalBench) each candidate is probed by spawning a fresh instance and restoring it to the current state — test-time scaling always uses `replay` restore, since a fresh run has no captured snapshots — before stepping. That makes `include_next_state: true` expensive on such adapters (one container restore per candidate per step); enable `harbor_replay_cache` to amortize the `k` same-state restores into one, or set `include_next_state: false` to skip candidate stepping entirely (candidates are then scored as `(state, action)` and only the selected action is stepped).

The scorer may be any benchmarked method, in one of two families. **Individual-prediction** scorers (`llm_direct`, `llm_verifier`, `llm_codegen`, `llm_eureka`, the vision methods `vlm_rm`/`vlm_sor`/`vip`/`liv_img`/`liv_txt`, and `baseline_random`) predict a value per candidate and the guided actor picks the highest. **Ranking-based** scorers (`llm_ranking`, `sdpo_ranking`, `sdpo_short_expert_continuation`, `delta_belief_ranking`) rank a step's candidates directly; the top-ranked candidate is selected. The score-based ranking methods (SDPO and delta-belief) always step candidates (`include_next_state` is forced true) and require a backend with `supports_full_scoring`; `sdpo_short_expert_continuation` additionally requires its `expert_rollout_store`. Vision scorers must supply an explicit goal — `vip_env_goal_image_path` / `liv_env_goal_image_path` / `liv_goal_text` — because no `Dataset` exists at test time; `goal_source: trajectory_end` is rejected. Two scorers are special: `llm_di ... (very long line, trimmed to 1000 chars)

Within each strategy, trajectories are rolled out in concurrent waves of `max_concurrent_trajectories`, advancing in lockstep: every decision step issues one batched actor generation across all in-flight trajectories' candidates (`active × k`, chunked by `batch_size`) and, for dense-signal scorers, one batched scoring call. Pure-step environments share a single environment instance; restore-based environments use one persistent instance per concurrent trajectory (reset in parallel, closed at wave end), with transient per-candidate environments bounded by `restore_concurrency`. Per-trajectory tie-break/fallback RNG is derived from `seed + trajectory_index`, so results are independent of the wave size. A single trajectory that errors is isolated and counted as a skipped trajectory (subject to `max_aborted_points`) without aborting the rest of the wave.

With `resume: true` (the default), each strategy's checkpoint is flushed the moment a trajectory completes — a crash or Ctrl-C mid-wave keeps everything already finished, losing only the trajectories still in flight. Only successful trajectories are checkpointed; failed/skipped ones are retried on the next run. Because the per-trajectory RNG depends only on `seed + trajectory_index`, a resumed run merges loaded and fresh results in trajectory order and reproduces exactly what an uninterrupted run would have produced. The fingerprint covers the environment context, resolved system prompt, seed, resolved task list, actor/scorer/selector backend content (model, sampling — not transport fields like `backend_url` or `tensor_parallel_size`), and the per-strategy settings; `batch_size`, `max_concurrent_trajectories`, `restore_concurrency`, `max_aborted_points`, and `success_threshold` are excluded, so changing them still resumes. A config change yields a different checkpoint filename — old checkpoints are left in place as orphans (safe to delete manually). Checkpoints are removed after all strategies' final artifacts are saved. The abort threshold and the logged stats cover only the current run's portion; summary metadata records `resumed_by_strategy` and `resume_fingerprints`. Do not run two identical configs against the same output directory concurrently (last writer wins).

`TestTimeScalingSelfConsistencyConfig` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `"discrete"` | `discrete` votes on each candidate's resolved action (small, enumerable action spaces; steps every candidate, so a pure-step env or restore support is required). `open_ended` uses Universal Self-Consistency: a selector LLM picks the most consistent action among the candidates' extracted actions, without stepping (works on restore-based envs) |
| `backend_name` | `str \| None` | `None` | Selector LLM backend. Required when `mode: open_ended`. The selector prompt invites a short reasoning preamble (at most two or three sentences) and requires the reply to *end* with `"The most consistent response is Response X"`, so `max_tokens` must be large enough to cover that reasoning — plus any `<think>` block for a thinking model — and still emit the final line. Too small a budget truncates before the answer, which parses as a failure (`all_scores_nan`) and falls back to a random candidate |
| `max_candidate_chars` | `int` | `4000` | `open_ended` only: each candidate's action text is head-truncated to this many characters in the selector prompt. Only affects the selector's view — the executed action is never truncated |

The recorded per-step `scores` in transition metadata hold the vote counts (`discrete`) or a one-hot vector for the selected candidate (`open_ended`); an unparseable `open_ended` selection — or a selector call that fails transiently and exhausts its retries — records the `all_scores_nan` fallback and a random choice. When every `open_ended` candidate renders to the same action text the selector call is skipped (the choice is outcome-invariant) and the first candidate is recorded as a one-hot selection.

### Collection Config (collect_dataset)

```yaml
backends_config_path: shared/configs/backends.yaml
environment: chain_sum
dataset_path: shared/data/datasets/chain_sum/dataset.pkl

discount_factor: 0.95
batch_size: 256
max_aborted_points: 0.1

collection:
    adapter: reasoning_gym
    backend_names: [eval, rollout]
    contexts_dir: shared/configs/environments/
    harbor_state_capture: replay

points_config:
    num_trajectories: 50
    max_points_per_trajectory: 10

# Optional: replay-validation audit for Harbor replay-mode environments
replay_validation:
    enabled: true
    target_points: 300
    max_discards: 50
```

`PipelineCollectionConfig.max_aborted_points` applies the same thresholded-abort policy used in prediction: if too many trajectories are dropped after recoverable per-trajectory failures, `collect_dataset.py` saves the current partial dataset and logs, then raises. Dropped trajectories are logged via `skipped_trajectory` and are excluded from saved `trajectory_results`, observable trajectory JSON, evaluation-point extraction, and downstream pass@k reuse.

`PipelineCollectionConfig.resume` (default `true`) makes trajectory collection crash-recoverable: after each chunk, completed trajectories are checkpointed to a stable fingerprint-keyed file next to the dataset (`{dataset_stem}_{fp16}.checkpoint.pkl`); on restart with the same config the checkpoint is loaded and only missing work runs, per policy — a fully completed policy skips its backend creation entirely. The fingerprint covers every content-affecting input (environment identity and context, resolved system prompt, seed, the resolved task-index list, policy backend content descriptors); `batch_size` and other transport/chunking knobs are excluded, so re-chunking still resumes, while changing e.g. a policy's model does not. Failed, skipped, and Harbor runtime-probe-discarded trajectories are never checkpointed and are retried on resume. Each stored result carries `collection_policy_index`/`collection_backend_name` provenance; checkpoints without it (or with a different fingerprint) are ignored with a warning. Evaluation points, returns, and ranking are always computed fresh after collection — resume covers only the trajectory-collection stage. The checkpoint is deleted after the final dataset save; orphans from abandoned configs are left in place (safe to delete manually). The `--fresh` CLI flag disables resume and removes the current config's checkpoint. Dataset metadata records `resumed_trajectories` and `resume_fingerprint`; `num_attempted_trajectories`/`drop_rate` cover only the current run's portion. Do not run two identical configs against the same dataset path concurrently (last writer wins).

For ranking-only resume runs, `resume_ranking_from` loads an existing dataset, reuses its stored `evaluation_points`, `trajectory_returns`, and `trajectory_results`, and skips fresh trajectory collection. You can optionally add:

```yaml
resume_ranking_from: shared/data/datasets/source_dataset.pkl
resume_ranking_subset:
    count: 100
    seed: 123
```

`resume_ranking_subset` samples `count` stored evaluation points uniformly without replacement before ranking collection. If `seed` is omitted, the selection uses `collection.seed`. The output dataset still keeps the full source `evaluation_points`; only `ranking_points` are restricted to the sampled subset. The saved dataset also preserves the source dataset's point-sampling config so later resampling continues to inherit the correct defaults. Collection metadata records the selected point indices and effective seed under `ranking_point_selection`.

Ranking-only resume is a strict same-environment workflow. When the source dataset metadata includes `env_name`, `adapter`, `max_steps`, or `make_kwargs`, `collect_dataset.py` validates those values against the current collection config and fails fast on any mismatch instead of attempting ranking collection against a different environment setup. For the step limit, the comparison uses the effective runtime value: `make_kwargs.max_steps` if present, otherwise top-level `max_steps`. If both are present, they must match.

#### ReplayValidationConfig

Optional audit for Harbor replay-mode collection. When enabled, selected evaluation-point states are replayed in fresh containers, fingerprinted with probe commands, and compared against a replayed reference baseline. This is only used when `collection.harbor_state_capture: replay`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Whether to run the replay-validation audit |
| `target_points` | `int` | `300` | Desired number of validated points (warning if not met) |
| `max_discards` | `int` | `50` | Maximum discarded points before erroring out |

### Prediction Config (predict)

```yaml
backends_config_path: shared/configs/backends.yaml
dataset_path: shared/data/datasets/chain_sum/dataset.pkl
predictions_dir: data/predictions/chain_sum

discount_factor: 0.95
batch_size: 256

prediction:
    eval_backend_name: eval

mc_rollout_persistence:
    resume: true

estimations:
    - name: mc_mean
      assumption: q_value
      method: mc
      aggregation: mean
      num_rollouts: 64
      backend_name: rollout

    - name: optimal_mean
      assumption: q_value
      method: mc
      aggregation: mean
      num_rollouts: 1
      policy_name: frozen_lake_4x4

eval_methods:
    - name: llm_direct_sv
      type: llm_direct
      signal_type: state_value
      prompt_preset: v0
      backend_name: eval
    - name: llm_codegen_qv
      type: llm_codegen
      signal_type: q_value
      prompt_preset: v0
      backend_name: eval
    - name: llm_eureka_qv
      type: llm_eureka
      signal_type: q_value
      prompt_preset: v1
      backend_name: eval
      num_samples: 4
      search_iterations: 3
      judge_num_points: 8
```

`PipelinePredictionConfig` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backends_config_path` | `str` | `"shared/configs/backends.yaml"` | Path to shared backend registry |
| `dataset_path` | `str \| None` | `None` | Path to dataset pickle |
| `predictions_dir` | `str \| None` | `None` | Output directory for prediction JSONs |
| `batch_size` | `int` | `256` | Max concurrent rollouts per batch |
| `include_actor_thinking` | `bool` | `False` | Pass raw actor output to eval methods |
| `include_current_thoughts` | `bool` | `False` | Show actor's ReAct reasoning for current point in prompts |
| `include_history_thoughts` | `bool` | `False` | Show actor's ReAct reasoning in history turns |
| `discount_factor` | `float` | `1.0` | Discount factor γ for GT rollout returns |
| `step_penalty` | `float \| None` | `None` | Per-step penalty applied to GT rollout environments. When set, GT rollouts include a `StepPenalty` reward and the reward signal is forced to `None` (sum all signals) |
| `max_aborted_points` | `float` | `0.1` | Abort the run if the fraction of backend-aborted points exceeds this threshold. GT groups, ranking GT groups, and per-method prediction artifacts are saved before threshold enforcement, so reruns can resume from already-written outputs. Compatible `ranking_gt_*` artifacts are reused only when the flattened candidate layout, ranking-point identity, rollout-generation descriptor, and effective return semantics (`reward_signal_name`, `discount_factor`, `step_penalty`) all match. For multi-sample codegen (`llm_codegen` with `num_samples > 1`) the threshold is evaluated *per method* against the fraction of fully aborted samples — a sample whose code failed to parse/compile saves an all-NaN artifact with `generation_aborted: true` but only counts as a single aborted unit regardless of point count. Override via `--max-aborted-points` |
| `estimations` | `tuple[EstimationConfig, ...]` | `()` | GT estimation configs |
| `eval_methods` | `tuple[EvalMethodConfig, ...] \| None` | `None` | Eval method configs |
| `prediction` | `PredictionConfig` | (defaults) | Runtime prediction settings |
| `harbor_replay_cache` | `bool` | `False` | Enable filesystem-based replay cache for Harbor GT rollouts. First restore for each state replays and caches the rootfs as a tar; subsequent restores untar from cache. This is only appropriate when restore semantics are effectively filesystem-only, such as Harbor `independent_exec` runs that do not rely on persistent shell-session state. Do not combine it with `text_exec_mode: tmux_session`. Requires `replay` mode with `apptainer-hpc` runtime; ignored for other runtimes/modes |
| `skip_ranking_predictions` | `bool` | `False` | When `True`, the ranking prediction branch is skipped even if the dataset carries `ranking_points` and the estimation configs would otherwise trigger it. A single INFO log line records the skip, including the number of ranking points that were ignored. Use this to reuse a ranking-enabled dataset for a non-ranking prediction run without regenerating or pruning the dataset |
| `mc_rollout_persistence` | `MCRolloutPersistenceConfig` | (defaults) | Persist MC rollout trajectories during prediction and reuse them across reruns |

**Path resolution.** Config path literals fall into three buckets, discriminated by the literal itself:

- **Shared inputs** — `dataset_path`, `contexts_dir`, and `backends_config_path` carry a `shared/` prefix and resolve relative to the repository root (pipeline commands are run from there). They live under `shared/` so multiple catalogs reuse the same datasets, environment contexts, and backends without re-collecting.
- **Catalog outputs** — `predictions_dir`, the evaluate `output_dir`, and the evaluate `prediction_sources` paths stay `data/...` and join the **catalog root** that owns the config — the ancestor directory containing `catalog.py`. A generated config under `catalogs/qval_benchmark/configs/prediction/<env>/` therefore writes its outputs to `catalogs/qval_benchmark/data/predictions/<env>/...`, and an evaluation config under `catalogs/qval_benchmark/configs/evaluation/<env>/` reads its `prediction_sources` from `catalogs/qval_benchmark/data/predictions/<env>/...` and writes its summary under `catalogs/qval_benchmark/data/evaluations/<env>/...`. Absolute paths are kept as given, so a `prediction_sources` entry may also point at another catalog by absolute path.
- **CLI overrides** — `--dataset`, `--output-dir`, `--predictions-root`, and `--results-root` are used as given, relative to the current working directory.

`MCRolloutPersistenceConfig` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | `str \| None` | `None` | Root directory for rollout stores. `None` means `predictions_dir/rollouts` |
| `resume` | `bool` | `True` | Load existing compatible rollout stores before collecting missing rollouts |
| `trust_sibling_hash` | `bool` | `False` | Recovery escape hatch for additive `BackendConfig` changes. When the rollout cache key (the generation hash) is invalidated by a schema change but the underlying rollouts are still semantically valid, set this to `True` to adopt an orphan sibling directory under the same `dataset_fingerprint` parent in place of the missing or empty expected hash directory. Adoption requires exactly one sibling whose manifest lists at least one shard and whose listed shard files exist on disk; multiple siblings raise, and broken siblings are rejected as recovery candidates. On adoption, the sibling is renamed into the expected path so the dirname keeps tracking the manifest's `generation_hash` — i.e., a follow-up run for the same spec finds the rollouts at the expected path without needing the flag again. A WARNING logs the field-level diff between the cached and requested specs so the user can confirm the change is benign |

The rollout store layout is `{predictions_dir or mc_rollout_persistence.dir}/rollouts/{namespace}/{primitive}/{dataset_sha256}/{generation_hash}/`. The `generation_hash` is a SHA-256 of the JSON-serialized rollout-generation spec (environment, policy, prompt) — including every field of the resolved `BackendConfig`. Adding an optional field to `BackendConfig` therefore changes the hash for every existing backend, even when the new field defaults to `None`. `trust_sibling_hash` exists for that case: schema-only churn that didn't affect what the rollouts would have been. Do not enable it when an actual policy property changed (model, sampling parameters, system prompt, environment kwargs). Within a single `predict.py` run the flag also refuses to adopt the same physical sibling for two semantically different specs — that would silently share rollouts across methods via the in-memory cache. Disambiguate manually if you need that.

Shard granularity is derived automatically: each rollout group flushes one shard per `max(1, batch_size // num_rollouts)` points — i.e., one shard per natural parallel lockstep batch. This keeps durability aligned with the configured `batch_size` without a separate knob. For `batch_size=4, num_rollouts=4`, that's one shard per point (safest for slow/expensive rollouts like TerminalBench with Codex).

Signal type is no longer a top-level prediction config field. Each eval method declares its own `signal_type`, enabling a single prediction run to evaluate multiple signal types.

#### Dual-layer step_penalty and discount_factor

`step_penalty` and `discount_factor` can be configured at both collection and prediction levels, enabling a single collected dataset to be reused across experiments with different penalty/discount settings.

**Per-signal-type data flow:**

| Signal type | GT source | Uses collection step_penalty? | Uses prediction step_penalty? | Uses collection discount_factor? | Uses prediction discount_factor? |
|-------------|-----------|-------------------------------|-------------------------------|----------------------------------|----------------------------------|
| SHAPED_REWARD | Stored `trajectory_returns` | Yes (baked in) | No | Yes (baked in) | No |
| STATE_VALUE | MC rollouts at prediction time | No | Yes (in GT rollout env) | No | Yes (in `trajectory_return()`) |
| Q_VALUE | MC rollouts at prediction time | No | Yes (in GT rollout env) | No | Yes (in `trajectory_return()`) |
| ADVANTAGE | MC rollouts at prediction time | No | Yes (in GT rollout env) | No | Yes (in `trajectory_return()`) |
| POTENTIAL | MC rollouts at prediction time | No | Yes (in GT rollout env) | No | Yes (in `trajectory_return()`) |

**Per-method prompt disclosure:**

| Setting | Disclosed to eval method? | Controlled by |
|---------|--------------------------|---------------|
| `step_penalty` | Only if `disclose_step_penalty: true` | `EvalMethodConfig.disclose_step_penalty` |
| `discount_factor` | Only if `disclose_discount_factor: true` | `EvalMethodConfig.disclose_discount_factor` |
| `max_steps` | Always (via `environment_notes`) | Automatic from dataset metadata |
| Efficiency hint | When either penalty or discount is active but not disclosed, and the preset includes an `efficiency_guidance` block | Automatic from disclosure settings + preset |

When `disclose_step_penalty` is `False` (default), the eval method's prompts do not mention the step penalty — but GT rollouts still include it. This allows studying whether methods can produce useful dense signals without being told about the penalty. When `disclose_discount_factor` is `False` (default), prompts use neutral "cumulative reward" language instead of mentioning the specific discount factor value. The method's `MethodContext` retains the real `discount_factor` for any non-prompt uses; only the prompt text is affected. The actual GT computation still uses the real `discount_factor` from the prediction config.

When either setting is active but not disclosed, and the prompt preset includes an `efficiency_guidance` block (e.g., V1), a vague efficiency hint is injected into the system prompt. This provides an implicit signal that reaching the goal efficiently matters, without revealing exact penalty or discount values. V0 preset has no efficiency guidance block, so no hint is possible with V0.

Prediction outputs are written to:

- `predictions_dir/gt/<gt-method>_<YYYYmmdd_HHMMSS>.json`
- `predictions_dir/eval/<method>_<YYYYmmdd_HHMMSS>.json`
- `predictions_dir/rollouts/...` (or `mc_rollout_persistence.dir`) for persistent MC trajectory shards and manifests
- `predictions_dir/predict.logs_<YYYYmmdd_HHMMSS>.jsonl` for the streaming prediction log

#### Persistent MC rollout storage

MC GT collection writes append-only trajectory shards as work completes instead of treating rollout returns as ephemeral. Compatibility is checked against:

- the dataset file content hash (SHA-256 of the dataset pickle bytes)
- the rollout primitive (`state`, `state_action`, or `next_state`)
- the logical point namespace (`evaluation_points` or `ranking_candidates`)
- the resolved rollout-generation settings that can change the sampled trajectory distribution (backend or scripted policy, sampling params, system prompt, turn info, and environment/restore semantics)

The stored source of truth is the trajectory, not discounted returns. This means the same saved MC rollouts can be reused across:

- changed `discount_factor`
- changed aggregation (`mean`, `max`, etc.)
- changed `step_penalty`
- changed reward-signal selection (`reward_signal_name` vs total reward)
- resumed or expanded runs with a larger `num_rollouts`

For Harbor-backed environments such as TerminalBench, pure runtime-infrastructure
selection does not affect compatibility: the rollout store ignores keys such as
`environment_type`, `podman_command`, `apptainer_command`, `trials_dir`, and
`sif_cache_dir` when building the rollout-generation identity. Semantics-relevant
Harbor settings like `text_exec_mode`, timeouts, flags, and restore mode are
still included.

Prompt guardrails also do not affect persistent compatibility. Settings such as
`max_model_len`, `min_action_chars`, `min_observation_chars`, and
`min_current_observation_chars` may change how aggressively prompts are
truncated to stay within a backend's context window, but they are treated as
operational safety controls rather than rollout semantics and therefore do not
change the rollout-store identity.

`resume: false` skips loading existing shards for the current run but still appends newly collected trajectories to the store.

#### GT prediction resume

GT predictions are saved incrementally as each estimation group completes, so a crash during GT collection preserves predictions for groups that finished before the failure. On rerun, `predict.py` checks `predictions_dir/gt/` for existing GT prediction files and reuses those that are compatible with the current configuration. A prediction is compatible when its config (assumption, method, aggregation, rollout count, backend, discount factor, step penalty) matches the current estimation config, its value count matches the number of evaluation points, its metadata fingerprints (dataset hash, point ordering hash) match the current dataset, and it does not contain `NaN`/`null` values.

Incompatible predictions (from changed configs, different datasets, different point sets, or saved `NaN` values) are ignored and recomputed. With MC rollout persistence and `resume: true`, rerunning a NaN-bearing GT method still reuses compatible trajectory shards already on disk and collects only the rollouts missing from the store. This allows incremental recovery: fix a crash, rerun, and only the missing GT methods or failed rollout slots are collected.

Prediction also validates the loaded dataset before any work starts. Datasets whose stored `trajectory_results` still contain failed trajectories, or whose evaluation/ranking points reference invalid trajectory indices, are rejected instead of being reused silently.

#### Transient API error resilience

MC rollout collection retries transient API errors with exponential backoff (up to 6 retries, starting at 4 seconds and capped at 8 seconds). Transient detection is provider-agnostic: it covers connection/time-out style SDK errors, backend responses with status codes `429`, `500`, `502`, `503`, or `529`, and `MalformedResponseError` (raised by `llenvs` when a provider returns HTTP 200 with a structurally invalid body — e.g. OpenRouter responding with `choices: null` when an upstream provider fails), regardless of whether they come from OpenAI or another SDK.

For rollout collection, `llenvs` first retries failed batch slots individually up to 10 times with 2s, 4s, 8s, then 8s delays. If a slot still fails after those runner-level retries, the point is marked aborted instead of crashing the whole GT shard immediately. Eval-side batched generation keeps any successful partial results and retries only the failed transient subset with the same 10-retry capped-backoff policy. Test-time scaling's batched generation uses the same per-slot policy — both the actor's candidate sampling and the open-ended self-consistency selector — so a dropped connection retries only the failed prompts; a selector slot that still fails degrades that trajectory's selection to the random fallback rather than aborting the run. Non-transient errors still propagate immediately.

Dataset and prediction paths can be provided as:

- a directory containing timestamped files (latest selected automatically), or
- a base name like `data/env/dataset.pkl` (resolved to the latest timestamped sibling).

### Evaluation Config (evaluate)

Most runs need **no evaluation config at all**. `evaluate.py` runs in *discovery
mode* by default — invoked with no single-evaluation arguments it walks the
catalog's `data/predictions/` tree, pairs every GT/eval combination using the
registry's applicability rules, and writes one summary per combination. An
evaluation config is the opt-in *single-evaluation mode*, for custom comparisons
that discovery's automatic pairing does not produce: an explicit `(gt, eval)`
subset, `gt_vs_gt_comparisons`, `method_name_prefix` to disambiguate artifacts
across runs, or sources drawn from another catalog by absolute path. The fields
below apply to that mode.

```yaml
dataset_path: shared/data/datasets/chain_sum/dataset.pkl
output_dir: data/evaluations/chain_sum

prediction_sources:
    - path: data/predictions/chain_sum/gt
      type: gt
      include_methods: [mc_mean, mc_max]
    - path: data/predictions/chain_sum/eval
      type: eval
      include_methods: [llm_direct]

correlation_methods:
    - pearson
    - spearman
    - kendall_tau
    - sign_agreement

# Optional: only evaluate selected GT/eval pairs.
# If omitted, evaluate.py computes the full GT x eval Cartesian product.
# eval accepts a single method name or a list for 1:N pairing.
comparisons:
    - gt: sv_mc_mean
      eval: [llm_direct_sv_self, llm_direct_sv_optimal]
    - gt: qv_mc_mean
      eval: llm_codegen_qv_optimal
```

Each `prediction_sources` entry can also accept `paths` (a list of file or directory paths). Each path may be a concrete prediction JSON, a directory of prediction JSONs, or a base path that resolves to timestamped siblings like `eval_llm_direct_*.json`. Method names must be unique across all sources or evaluation will error.

Each entry also accepts `method_name_prefix`, applied to every loaded prediction's `method_name` after `include_methods` / `exclude_methods` filtering. Use it to disambiguate artifacts that share a method name across runs (for example, the same GT computed under two different backends). Filters and comparison-entry references use the original (un-prefixed) names; `gt_vs_gt_comparisons`, `comparisons`, and `summary.json` all use the prefixed names.

```yaml
prediction_sources:
    - path: data/predictions/terminal_bench/100pt_lite-easy_40ms_ds32/gt
      type: gt
      method_name_prefix: ds_
    - path: data/predictions/terminal_bench/100pt_lite-easy_40ms_codex/gt
      type: gt
      method_name_prefix: codex_

gt_vs_gt_comparisons:
    - lhs: ds_qv_mc_mean
      rhs: codex_qv_mc_mean
```

These `data/...` paths are catalog-relative (they join the owning catalog root, as above). To compare across two catalogs, give one source an absolute path.

When multiple timestamped prediction artifacts belong to the same filename family, evaluation selects the newest file as the canonical artifact without warning on the duplicate family by itself. Corrupt older siblings emit warnings but do not fail the load; a corrupt newest artifact emits a warning and fails the load.

When `comparisons` is provided, only the listed `(gt, eval)` pairs are evaluated, in the listed order. The `eval` field accepts a single method name or a list of names (1:N pairing). When `comparisons` is omitted or set to `null`, all GT predictions are compared against all eval predictions.

When `comparisons` is specified, signal types are resolved per eval prediction from its artifact config, so mixed-signal-type evaluation is supported. When no comparisons are specified (Cartesian product mode), all eval predictions must share the same signal type or `evaluate.py` raises an error. You can also set `signal_type` in the evaluation config to override explicitly.

### `evaluate.py` discovery mode

Run with no single-evaluation arguments, `scripts/pipeline/evaluate.py` discovers standardized prediction experiments under the active catalog's `data/predictions/` (the bundled catalog: `catalogs/qval_benchmark/data/predictions/`) and evaluates each runnable environment/dataset/actor/modality directory in-process. It evaluates compatible pointwise Q-value/state-value correlations and ranking Q-value correlations:

- eval JSON filenames must be `<method>[-<variant>]_<qv|sv>_<YYYYMMDD_HHMMSS>.json`
- GT JSON filenames must be `qv_<method>_<YYYYMMDD_HHMMSS>.json`, `sv_<method>_<YYYYMMDD_HHMMSS>.json`, `ranking_gt_qv_<method>_<YYYYMMDD_HHMMSS>.json`, or `ranking_gt_sv_<method>_<YYYYMMDD_HHMMSS>.json`
- the JSON `method_name` must equal the filename basename without the timestamp suffix
- the JSON `method_type` must match the containing directory (`gt` or `eval`)
- when present, eval `config.signal_type` and pointwise GT `config.assumption` must match the qv/sv signal token in the filename

Discovery evaluates only the hard-coded method catalog for each actor/modality. Valid JSON files whose method names are not expected for that actor/modality are logged in `detected_prediction_files` but are not evaluated.

When several valid JSON artifacts exist for the same method in one `gt/` or `eval/` directory, the newest timestamp is selected and older siblings are logged as `superseded`. Discovery passes the exact selected JSON paths into each evaluation, so stale siblings in the same directory are not loaded. Pointwise length mismatches are not prefiltered by discovery; the evaluation compares the shared prefix of GT and eval values. Ranking artifacts are routed to the ranking path when compatible `ranking_gt_qv_*` artifacts exist.

Some environment/dataset pairs can list multiple expected GT actors. Discovery creates one evaluation job per present GT actor and reports any missing expected GT directories. Generated output directories always include the GT actor token, for example `<dataset>_<gt_actor>_<eval_actor>_<modality>`, so the runs do not overwrite each other and plotting can distinguish GT-set ablations. TerminalBench is treated as Q-value-only; its pointwise Q-value methods are evaluated against every listed TerminalBench GT actor, while ranking methods are evaluated only against TerminalBench GT actors that provide ranking GT artifacts.

Pass `--missing-report path/to/report.json` to write a JSON report with these sections:

- `missing` lists expected standardized directories, methods, or GT signal families that are absent
- `high_null_prediction_files` lists selected latest valid prediction artifacts whose `values` list is more than 10% JSON `null` values, including the null count, total prediction count, fraction, and percentage
- `detected_prediction_files` lists every prediction JSON considered by discovery, including selected files, superseded duplicate runs, malformed JSON, method-name mismatches, method-type mismatches, and files that do not follow the supported qv/sv filename convention

The detection status distinguishes `selected_for_evaluation` from `selected_ignored_non_pointwise`, `superseded`, and other skipped states. Only entries with `evaluable: true` are evaluated. The high-null section is intentionally experiment-level rather than file-system-level: it considers only latest selected valid artifacts (`selected_for_evaluation`, `selected_not_evaluated`, or `selected_ignored_non_pointwise`) so old superseded runs and convention-violating files do not inflate the report. Every detected file row also includes null counts and percentages when its JSON `values` field is a list.

### GT × GT correlations

Use `gt_vs_gt_comparisons` to correlate two ground-truth predictions directly (for example, comparing aggregation choices or a new GT estimator against an established baseline). Each entry names an `lhs` and an `rhs`; `rhs` accepts a single method or a list for 1:N expansion.

```yaml
gt_vs_gt_comparisons:
    - lhs: qv_mc_mean
      rhs: qv_mc_max
    - lhs: sv_mc_mean
      rhs: [sv_mc_max, sv_mcts_mean]
```

Both sides must carry the same `assumption` (resolved from each prediction's stored config) and must have the same number of values; mismatches raise. `SHAPED_REWARD` is rejected because the existing GT×eval path canonically uses the dataset's trajectory returns for that assumption, making GT×GT semantics ambiguous.

Results land in a dedicated top-level `gt_vs_gt_estimations` key in the output `summary_<ts>.json`, parallel to `estimations`. `gt_vs_gt_comparisons` can be combined with ordinary `comparisons` in the same run, or used alone — when no eval predictions exist and only GT×GT pairs are configured, `evaluate.py` still runs.

## Scripts

The experiment workflow is split into independent scripts with well-defined data boundaries:

0. **`compute_pass_at_k.py`** (optional) — Evaluate model competency on environments → timestamped `pass_at_k_<ts>.json`
   - **`compute_pass_at_k_from_dataset.py`** — Re-derive Pass@k from a stored dataset pickle (no inference)
   - **`compute_pass_at_k_from_predictions.py`** — Pass@k over MC rollouts persisted by `predict.py`
   - **`test_time_scaling.py`** — Run an online guided-action experiment that samples k candidate actions per step and optionally scores them with a configured method
1. **`collect_dataset.py`** — Collect trajectories and evaluation points → timestamped `dataset_<ts>.pkl`
2. **`predict.py`** — Generate GT estimates and/or eval method predictions → timestamped prediction JSONs
3. **`evaluate.py`** — Compute correlations between GT and eval predictions → `summary.json`. Runs in discovery mode by default (walks the active catalog's `data/predictions/`, reports missing combinations, and evaluates every discovered experiment in-process); pass `--config` for a single explicit evaluation
4. **`audit_predictions.py`** (diagnostic) — Check prediction JSON artifacts for load failures, short outputs, and null-heavy predictions

This separation enables:
- Evaluating model competency before committing to expensive data collection
- Collecting data once and running multiple GT/eval experiments against it
- Adding new GT methods or eval methods without recomputing expensive trajectory rollouts
- Running GT estimation and eval methods independently or together
- Multi-policy trajectory collection (multiple backends per dataset)

### `scripts/pipeline/compute_pass_at_k.py` — Step 0: Model Competency (optional)

Evaluates whether a model achieves sufficient success rate on an environment to serve as an "optimal policy" for ground truth estimation. Runs N independent trajectories per task, groups by task, and computes Pass@k using the combinatorial formula `1 - C(n-c, k) / C(n, k)`.

Success semantics: a trajectory counts as a Pass@k success when its undiscounted cumulative return under the environment's `reward_signal_name` meets `success_threshold` (default `1.0`). `step_penalty` contributions are stripped from the sum so control cost never flips a passing trajectory into a failure. For dense-reward environments (Craftax `craftax_reward`, Jericho `game_score`), raise `success_threshold` to the per-environment meaningful cutoff.

#### CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--config` | Yes | Path to pass@k YAML config |
| `--output-dir` | No | Override output directory |
| `--tensor-parallel-size` | No | Override `tensor_parallel_size` for vLLM backends |
| `--max-aborted-points` | No | Override the dropped-trajectory threshold for this run |
| `--success-threshold` | No | Override the success-criterion threshold for this run |

#### Output

JSON report with per-task results (including `task_index`, `task_name` when available, and requested vs retained samples per task) and macro-averaged aggregates (mean, std, min, max per k). Metadata also records requested, retained, and dropped trajectory counts plus the drop rate, the effective `success_threshold`, and the `reward_signal_name` used to derive success. Console summary is also printed.

### `scripts/pipeline/test_time_scaling.py` — Guided Online Action Selection

Runs a test-time scaling experiment against an environment. At every step, the actor backend samples `guided_actor.k` candidate actions. If a scorer is configured, those candidates are converted into temporary `EvaluationPoint`s and scored with the configured method; the highest finite score is selected. If all scores are `NaN`, selection falls back to a random candidate and records `fallback_reason: all_scores_nan`. With no scorer and `k=1`, the strategy is the ordinary actor baseline.

While each strategy runs, the script emits periodic progress info lines (`test_time_scaling:{strategy}: {done}/{total} ({pct}%)`) so long runs are visible while they execute. They are throttled to roughly one line per 10% of completed trajectories (plus the first and final update) and count every finished trajectory — successful or skipped — including any restored from a resume checkpoint. Like the other pipelines' progress lines, they are printed only and are not persisted in the experiment JSONL log.

The script writes a JSON summary plus per-strategy trajectory artifacts. Each transition stores `info.test_time_scaling` metadata containing the candidate action texts, scores, selected index, scorer usage, and fallback reason. The summary reports success rate, mean return, mean steps, mean candidate count, scorer steps, fallback steps, and per-task success rates for `baseline_k1` and, when configured, `guided`.

### `scripts/pipeline/compute_pass_at_k_from_dataset.py` — Pass@k from a stored dataset

Computes the same Pass@k report as `compute_pass_at_k.py` but reads the actor trajectories out of a `Dataset` pickle instead of running new rollouts. Use this when you want to retroactively assess the Pass@k of a previously collected dataset without repeating inference. Requires the dataset to carry `trajectory_results` (datasets produced with `--strip-trajectories` or resampled-and-stripped pickles can't be used).

#### CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--dataset` | Yes | Path to the `Dataset` pickle (supports timestamped-sibling resolution) |
| `--output-path` | No | Override JSON output path. Defaults to `<dataset_stem>_pass_at_k_<ts>.json` next to the dataset |
| `--k-values` | No | Override `k` values. Defaults to `1 5 10 20` |
| `--success-threshold` | No | Override the success-criterion threshold. Defaults to `1.0` |

The script resolves `reward_signal_name` from the dataset's `config` (falling back to `metadata`), then uses `dataset.metadata["task_indices"]` (when present) to recover per-task requested counts so that "every trajectory for this task dropped" surfaces correctly in the report.

### `scripts/pipeline/compute_pass_at_k_from_predictions.py` — Pass@k from MC rollouts

Computes Pass@k from the persisted MC rollouts in a `predict.py` output directory. Each stored trajectory is treated as one attempt by the policy to solve the environment starting from the point's state/action — the signal type the rollout was used to estimate (V(s), Q(s,a), advantage) does not affect Pass@k.

The script emits two orthogonal task groupings in one JSON report:

- `by_point`: each eval-point `(trajectory_index, step_index)` and each ranking candidate `(trajectory_index, step_index, candidate_index)` is its own "task". Rollouts across primitives (`state`, `state_action`, `next_state`) and across different GT estimations for the same point are pooled.
- `by_env_task`: every rollout aggregates under the underlying environment `task_name`, irrespective of which eval point or ranking candidate it came from.

ACTOR_PRIMARY ranking candidates reuse the eval-point rollouts rather than having their own; they're excluded from the `by_point` grouping by default to avoid double-counting. Pass `--include-actor-primary-rankings` to surface them as separate rows (borrowing storage from the eval point).

#### CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--prediction-dir` | Yes | Path to the `predict.py` output directory; must contain `rollouts/` |
| `--dataset` | No | Override the Dataset pickle. Defaults to the `*.pkl` sitting in `--prediction-dir` |
| `--rollout-store-root` | No | Override the rollouts directory. Defaults to `<prediction-dir>/rollouts` |
| `--output-path` | No | Override JSON output path. Defaults to `<prediction-dir>/pass_at_k_<ts>.json` |
| `--k-values` | No | Override `k` values. Defaults to `1 5 10 20` |
| `--success-threshold` | No | Override the success-criterion threshold. Defaults to `1.0` |
| `--include-actor-primary-rankings` | No | Include ACTOR_PRIMARY ranking candidates in the `by_point` grouping (off by default) |
| `--skip-fingerprint-check` | No | Skip validating that the rollout-store `<sha256>` path segment matches the loaded dataset. Use only when intentionally mixing artifacts |

#### Output

Single JSON report with `metadata`, `rollout_provenance` (namespaces, primitives, generation hashes seen, total rollouts pooled), plus `by_point` and `by_env_task` sections each shaped like `compute_pass_at_k.py`'s output (`aggregate` + `per_task`). `by_point.per_task[*].task_key` is a list of ints (the point key tuple) so the JSON is readable; `by_env_task.per_task[*].task_key` is typically the `task_name` string. Corrupt rollout shards surface as warnings and are skipped — the script still emits a report from the rollouts that did load.

### `scripts/pipeline/collect_dataset.py` — Step 1: Data Collection

Collects trajectories, extracts evaluation points, and computes trajectory returns. Saves results to a `Dataset` pickle. GT estimation is deferred to `predict.py`. Collection config does not include `signal_type`; dense-signal semantics start in prediction/evaluation.

During collection, the script emits periodic progress info lines so long batched trajectory runs are visible while they execute. These progress updates are printed only and are not persisted in the experiment JSONL log. The script still emits `trajectory_summary` events for each collection backend, one final `collection_summary` event covering the full merged dataset (including attempted, skipped, and dropped counts plus `num_ranking_points` and `completed` status), and a `skipped_trajectory` event whenever a per-task generation/reset/step failure is filtered instead of aborting the whole run. Filtered trajectories are logged and dropped; they are not reused as saved trajectories or evaluation points. If the run is aborted mid-way, a `collection_summary` with `completed: false` and `abort_reason` is emitted before saving partial results.

#### CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--config` | (required) | Path to collection YAML config |
| `--output-dir` | (from config) | Override the directory containing the timestamped dataset artifact |
| `--tensor-parallel-size` | (none) | Override `tensor_parallel_size` for all vLLM backends. Non-vLLM backends ignore this flag. On SLURM, auto-derived from `$SLURM_GPUS_ON_NODE` |
| `--max-aborted-points` | (from config) | Override the dropped-trajectory threshold for this run |

#### Output Files

- **`{dataset-path}`** — pickle with evaluation points, trajectory returns, config, and metadata
- **`{dataset-path}.logs.jsonl`** — LLM call logs from collection phase
- **`{trajectories-path}`** (optional) — human-readable trajectory JSON

### `scripts/pipeline/predict.py` — Step 2: Predictions

Loads a Dataset pickle, runs GT methods (MC/MCTS rollouts) and/or evaluation methods (LLM-based), and saves per-method prediction JSON files.

GT estimation reconstructs the environment from dataset metadata and runs MC/MCTS rollouts. Eval methods load a context YAML and run LLM-based evaluation.

During prediction, both GT estimation and eval-method execution emit periodic progress info lines so batched inference and rollout-heavy phases are visible while they run. These progress updates are printed only and are not persisted in `predict.logs_*.jsonl`. Rollout-based GT methods additionally emit `rollout_summary` events with return/step/count aggregates for the sampled rollouts, and `rollout_abort` events when a point-local recoverable environment failure aborts further rollout collection for that point while retaining any already completed rollouts. This includes replay/restore timeouts, shell-continuation failures, Harbor tmux-session death during command execution, and classified Jericho native faults such as `SIGFPE` or explicit emulator halts.

#### CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | (from config) | Path to dataset pickle |
| `--config` | (required) | Path to prediction YAML config |
| `--output-dir` | (from config) | Output directory for prediction JSONs |
| `--tensor-parallel-size` | (none) | Override `tensor_parallel_size` for all vLLM backends. Non-vLLM backends ignore this flag. On SLURM, auto-derived from `$SLURM_GPUS_ON_NODE` |
| `--max-aborted-points` | (from config) | Override the aborted-point threshold for this run |
| `--max-eval-points` | (none) | Restrict prediction to the first N evaluation points from the dataset |

Eval methods are configured through top-level `eval_methods` entries in the prediction YAML. Omit `eval_methods` to skip eval predictions entirely.

**Metadata-inherited settings:** `turn_info`, `system_prompt`, `make_kwargs`, `seed`, `max_steps`, `env_name`, and `adapter` are read from the dataset metadata when not explicitly provided via config. For the step limit, `predict.py` uses the effective runtime value: `make_kwargs.max_steps` if present, otherwise top-level `max_steps`; if both are present, they must match. `step_penalty` is configured at the prediction level (not inherited from dataset metadata) to enable experimenting with different penalties on the same collected dataset.

Eval and ranking prediction artifacts are written before `max_aborted_points` is enforced for that method, so a threshold-triggered abort still preserves the method output that was already computed.

**Runtime environment notes:** When the dataset metadata carries an effective step limit, `predict.py` automatically populates `MethodContext.environment_notes` with it. This uses `make_kwargs.max_steps` when present and otherwise falls back to top-level `max_steps`, requires both representations to agree when both are present, and renders as a `## Episode Configuration` section in system prompts.

#### Output Files

- **`{predictions_dir}/gt/{name}_{timestamp}.json`** — per-GT-method prediction JSON
- **`{predictions_dir}/eval/{name}_{timestamp}.json`** — per-eval-method prediction JSON
- **`{predictions_dir}/predict.logs_{timestamp}.jsonl`** — LLM call logs from prediction phase

#### Prediction JSON Format

Each prediction file contains:

```json
{
    "method_name": "mc_mean",
    "method_type": "gt",
    "values": [0.5, 0.3, null, 0.8],
    "config": {
        "assumption": "state_value",
        "method": "mc",
        "aggregation": "mean",
        "num_rollouts": 128,
        "backend_name": "rollout"
    },
    "metadata": {
        "backend_name": "rollout",
        "timestamp": "2025-01-15T10:35:00+00:00"
    }
}
```

Eval prediction files additionally persist method-specific metadata in `config`, including `method_type`, `policy_assumption`, `signal_type`, and `discount_factor`, so artifacts remain self-describing even after they are separated from the original YAML config.

NaN values are serialized as `null` in JSON and restored to `float('nan')` on load.

### `scripts/pipeline/evaluate.py` — Step 3: Correlations

Purely computational (no LLM calls, no environment). Loads dataset(s) and prediction JSONs, computes correlations, and writes summary JSONs. It runs in two modes.

#### Discovery mode (default)

With no single-evaluation arguments, the script discovers prediction artifacts under `--predictions-root`, pairs every compatible GT/eval combination, and evaluates each one in-process. The filename, selection, and pairing rules are detailed under [`evaluate.py` discovery mode](#evaluatepy-discovery-mode) above; in short, only combinations with existing compatible GT and eval artifacts are evaluated (Q-value methods against Q-value GTs, state-value against state-value, ranking-QV against `ranking_gt_qv_*`), one job per present GT actor. Each pair is evaluated exactly as a single-config run would be — there are no intermediate config files.

| Arg | Default | Description |
|-----|---------|-------------|
| `--catalog` | bundled `qval_benchmark` | Catalog `.py` file (or a directory containing `catalog.py`) selecting the experiment matrix; the discovery roots default under it |
| `--predictions-root` | `<catalog>/data/predictions` | Root containing standardized prediction subfolders |
| `--output-root` | `<catalog>/data/evaluations` | Root for per-experiment summary folders |
| `--missing-report` | (none) | JSON path for the missing-directory/missing-method report |
| `--dry-run` | off | Print discovered jobs and missing combinations without evaluating |
| `--verbose-configs` | off | With `--dry-run`, print the full per-job config dictionaries |
| `--exclude-zero-points` | off | Drop paired entries where either side is exactly zero before correlating |

A job that raises is logged and counted as a failure so one bad experiment does not abort the batch; a non-zero failure count makes the script exit non-zero.

#### Single-config mode

Pass any of `--config`, `--dataset`, or `--predictions-dir` to run exactly one explicit evaluation.

| Arg | Default | Description |
|-----|---------|-------------|
| `--config` | (none) | Evaluation config YAML naming `prediction_sources` and `comparisons` |
| `--dataset` | (from config) | Path to dataset pickle |
| `--predictions-dir` | (none) | Directory with prediction JSONs (alternative to a config's `prediction_sources`) |
| `--output-dir` | (from config) | Output directory for the summary |

For each selected GT prediction × eval prediction pair:
- Reads `assumption` from the GT prediction config
- Applies `transform_predictions()` on eval values (handles SHAPED_REWARD→POTENTIAL via cumsum)
- Computes correlations (PEARSON, SPEARMAN, KENDALL_TAU, SIGN_AGREEMENT)

Pair selection works as follows:

- If `comparisons` is omitted, the script computes the full Cartesian product of all loaded GT predictions and all loaded eval predictions.
- If `comparisons` is provided, the script evaluates only those explicit pairs and errors if any referenced method name was not loaded by `prediction_sources`.

For SHAPED_REWARD assumption:
- Uses `trajectory_returns` from the dataset as GT
- Sums eval predictions per trajectory via `sum_per_trajectory()`
- Correlates trajectory-level sums with trajectory returns

Pointwise prediction length mismatches are tolerated in both modes: GT and eval values are truncated to their shared prefix before correlation.

#### Output Files

- **`{output-dir}/summary_<ts>.json`** — experiment info, config, correlations, predicted/GT values

### `scripts/pipeline/audit_predictions.py` — Prediction Artifact Audit

Scans prediction JSON artifacts under `catalogs/qval_benchmark/data/predictions` by default and reports:

- files that cannot be loaded as prediction artifacts because they are corrupt, incomplete, or schema-invalid
- loaded prediction files with fewer than 100 values
- loaded GT ranking prediction files (`prediction_format: ranking_q_values` or `ranking_gt_*`) with fewer than 400 flattened candidate values
- loaded prediction files with more than 10% null values

By default, only JSON files directly inside `gt/` or `eval/` directories are treated as prediction artifacts, so rollout-store manifests under prediction outputs are ignored. Codegen prediction artifacts are also skipped by default; pass `--include-codegen` when you want to audit them.

#### CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--root` | `catalogs/qval_benchmark/data/predictions` | Root directory to scan recursively |
| `--min-predictions` | `100` | Minimum count for ordinary prediction files |
| `--min-gt-ranking-predictions` | `400` | Minimum count for GT ranking prediction files |
| `--max-null-rate` | `0.10` | Maximum allowed null fraction before a file is reported |
| `--format` | `text` | Report format: `text` or `json` |
| `--all-json` | off | Scan every `.json` file under `--root`, including auxiliary manifests |
| `--include-codegen` | off | Include codegen prediction artifacts in the audit |
| `--fail-on-issues` | off | Exit with status 1 when any issue category is non-empty |

**`summary.json`** example:

```json
{
    "experiment_info": {
        "dataset_source": "shared/data/datasets/frozen_lake/dataset_20260302_120000.pkl",
        "predictions_dir": null,
        "env_name": "frozen_lake",
        "signal_type": "state_value",
        "num_evaluation_points": 50,
        "num_trajectories": 10,
        "gt_methods": ["mc_mean"],
        "eval_methods": ["llm_direct"],
        "correlation_methods": ["pearson", "spearman", "kendall_tau", "sign_agreement"],
        "timestamp": "2025-01-15T10:35:00+00:00"
    },
    "dataset_config": { "signal_type": "state_value", "...": "..." },
    "dataset_metadata": { "env_name": "frozen_lake", "...": "..." },
    "estimations": {
        "mc_mean": {
            "assumption": "state_value",
            "granularity": "point",
            "ground_truth_values": [0.5, 0.3, "..."],
            "methods": {
                "llm_direct": {
                    "correlations": {
                        "pearson": { "correlation": 0.72, "p_value": 0.001, "num_points": 50 },
                        "spearman": { "correlation": 0.68, "p_value": 0.003, "num_points": 50 },
                        "kendall_tau": { "correlation": 0.65, "p_value": 0.005, "num_points": 50 }
                    },
                    "predicted_values": [0.6, 0.4, "..."]
                }
            }
        }
    }
}
```

### Adapters

| Adapter | Environments | Multi-Turn | Reward Signal | System Prompt |
|---------|-------------|-----------|---------------|---------------|
| `reasoning_gym` | Single-turn reasoning tasks (chain_sum, letter_counting, etc.) | No | `correctness` | Auto (answer tags) |
| `gem` | GEM games and tasks (GuessTheNumber, Sudoku, Wordle, etc.) | Yes (games) | `correctness` | None (GEM manages observations) |
| `agentgym` | AgentGym environments (alfworld, maze, webshop, etc.) | Yes | `agentgym_native` | Auto (from adapter) |
| `gymnasium` | Standard Gymnasium environments (FrozenLake, CartPole, etc.) | Yes | `gym_reward` | None (built internally) |
| `harbor` | Harbor containerized datasets (TerminalBench, swe-bench, etc.) | Yes | `harbor` | Auto (terminal-agent prompt) |
| `alfworld` | Native ALFWorld household tasks via llenvs | Yes | `task_completion` | None (built internally) |
| `open_apps` | Computer-use application tasks (OpenApps) | Yes | `task_completion` | Auto (from adapter) |
| `jericho` | Interactive fiction text adventures (Zork, Anchorhead, etc.) | Yes | `game_score` | Auto (per-game: verbs, scoring, mechanics) |
| `webshop` | E-commerce product search and purchase (WebShop) | Yes | `purchase_match` | Auto (shopping instructions) |
| `craftax` | Open-ended survival game (Craftax Classic and Full) | Yes | `craftax_reward` | Auto (survival gameplay) |

This table documents the full `llenvs` adapter layer. The bundled QVal benchmark
catalog uses only four of them — `alfworld`, `gymnasium` (FrozenLake), the
OpenApps adapter, and `harbor` (TerminalBench). The others (`reasoning_gym`,
`gem`, `jericho`, `webshop`, `craftax`) are available through `llenvs` but are not
part of the shipped catalog.

### Backends

| Backend | Use Case | Key Args |
|---------|----------|----------|
| `openai` | Remote OpenAI-compatible API (OpenAI, vLLM server, etc.) | `backends.<name>.backend_url` |
| `huggingface` | Local inference via HuggingFace transformers | `backends.<name>.device`, `backends.<name>.dtype` |
| `vllm` | In-process vLLM inference (no separate server needed) | `backends.<name>.gpu_memory_utilization`, `backends.<name>.max_model_len`, `backends.<name>.dtype`, `backends.<name>.tensor_parallel_size`, `backends.<name>.vllm_kwargs` |
| `vllm_singularity` | In-container `vllm serve` via Singularity — bare-metal parent process spawns `singularity exec … vllm serve …` as a sibling subprocess and talks to it over HTTP. Use when the model needs a newer vLLM than the cluster's native install (e.g., gemma-4 requires vLLM ≥ 0.19). See `llenvs/docs/guides/singularity.md` for `.sif` setup and cluster profile | Reuses `backends.<name>.model`, `backends.<name>.tensor_parallel_size`, `backends.<name>.gpu_memory_utilization`, `backends.<name>.max_model_len`, `backends.<name>.dtype`, `backends.<name>.max_concurrency`. Adds `backends.<name>.singularity_sif` (defaults to `$LLENVS_SIF`), `backends.<name>.singularity_binds` (defaults to `$LLENVS_BINDS`), `backends.<name>.singularity_extra_vllm_args`, `backends.<name>.singularity_startup_timeout`, `backends.<name>.singularity_cuda_visible_devices` |
| `anthropic` | Anthropic API (Claude models) | API key via `ANTHROPIC_API_KEY` env var |
| `openrouter` | OpenRouter proxy to multiple providers | API key via `OPENROUTER_API_KEY` env var |
| `litellm` | 100+ providers via the litellm SDK, incl. LiteLLM proxy/gateway servers (virtual keys) | Model in litellm `provider/model` format; `backends.<name>.api_base` for a gateway. API key via the provider's native env var (`LITELLM_PROXY_API_KEY`, `GEMINI_API_KEY`, ...) |
| `codex` | Subprocess wrapper around the `codex exec` CLI | Requires the `codex` binary on `PATH`. `backends.<name>.config_overrides` forwards `-c key=value` overrides (e.g., `model_reasoning_effort: low`). Temperature / top_p / top_k / stop sequences / thinking / second-elicitation are not exposed by the CLI — `sampling_params_from_config` strips them for this backend type |
| `claude_code` | Subprocess wrapper around the `claude --print` CLI | Requires the `claude` binary on `PATH` (Claude Code v2.1+). `backends.<name>.claude_code_kwargs` forwards constructor kwargs (`effort`, `bare`, `tools`, `permission_mode`, `max_budget_usd`, `fallback_model`, `extra_args`, …). Tool access is disabled by default (`tools=""`) so the model produces a single shell command per turn for tmux-style harnesses; set `tools` and `permission_mode` to enable native Claude Code tool calls. Temperature / top_p / top_k / stop sequences / thinking / second-elicitation are not exposed by the CLI — `sampling_params_from_config` strips them for this backend type. Auth: OAuth/keychain by default; pass `claude_code_kwargs.bare: true` to use `ANTHROPIC_API_KEY` only (required on headless cluster jobs) |

### Examples

```bash
# Step 1: Collect dataset (FrozenLake). The environment is taken from the
# collection config's env_name/context_name — there is no --env-name flag.
python scripts/pipeline/collect_dataset.py \
    --config shared/configs/collection/frozen_lake/8x8_scripted.yaml

# Step 2: Predict — GT + eval methods. The methods come from the config's
# eval_methods / estimations; the backend from each method's backend_name.
python scripts/pipeline/predict.py \
    --config catalogs/qval_benchmark/configs/prediction/frozen_lake/100pt_8x8_q35-27-or_text.yaml

# Step 3: Evaluate — discovery mode pairs every prediction automatically.
python scripts/pipeline/evaluate.py

# ...or score one explicit comparison with a custom evaluation config:
python scripts/pipeline/evaluate.py \
    --config catalogs/qval_benchmark/configs/evaluation/frozen_lake/8x8_q35-27-or_text.yaml

# To predict with a different backend, point the method's backend_name at
# another entry in shared/configs/backends.yaml. For multi-environment runs,
# loop predict over the generated configs and run evaluate once (discovery).
```

### Experiment Config YAML Format

Experiment configs are split into three step-specific files plus a shared backend registry.

### Multi-Assumption (Prediction Config)

A single prediction config can evaluate method output under multiple assumptions simultaneously. GT estimations use `assumption` on each entry; eval methods use `signal_type` on each entry:

```yaml
estimations:
    - name: as_shaped_reward
      assumption: shaped_reward

    - name: as_potential_mc64_max
      assumption: potential
      method: mc
      num_rollouts: 64
      aggregation: max
      backend_name: rollout

    - name: as_advantage_mc64
      assumption: advantage
      method: mc
      num_rollouts: 64
      backend_name: rollout

eval_methods:
    - name: llm_direct_sv
      type: llm_direct
      signal_type: state_value
      prompt_preset: v1
      backend_name: eval
    - name: llm_codegen_qv
      type: llm_codegen
      signal_type: q_value
      prompt_preset: v1
      backend_name: eval
    - name: llm_eureka_qv
      type: llm_eureka
      signal_type: q_value
      prompt_preset: v1
      backend_name: eval
      num_samples: 4
      search_iterations: 3
      judge_num_points: 8
```

### Minimal YAML

A prediction config needs at minimum a dataset path and predictions dir. Signal types are declared per eval method:

```yaml
dataset_path: shared/data/datasets/<env>/dataset.pkl   # shared input (repo-relative)
predictions_dir: data/predictions/<env>                # catalog output (joins the catalog root)
```

### Error Handling

Unknown keys raise `ValueError` immediately, catching typos:

```yaml
# This will raise: ValueError: Unknown keys: signl_type
signl_type: q_value
```

Invalid enum values also raise `ValueError` with the list of valid options:

```yaml
# This will raise: ValueError: Invalid signal_type: 'reward'. Valid values: [...]
eval_methods:
  - name: test
    type: llm_direct
    signal_type: reward
    prompt_preset: v0
```

## Running on a SLURM Cluster

Below is a complete example of running qval experiments on a GPU cluster via SLURM, using vLLM as the LLM backend.

### Setup

The simplest approach uses `backends.<name>.type: vllm`, which loads the model in-process (no separate server). Alternatively, you can start a vLLM server and set `backends.<name>.type: openai` with `backend_url` — useful when multiple scripts share one server, or when you need the server running independently.

The example below uses the server-based approach for illustration:
1. vLLM runs on the same node as the benchmark scripts (started in the SLURM job).
2. The evaluated method and the MC rollout backend both query the local vLLM server via its OpenAI-compatible API.
3. Each SLURM job runs one environment: collect data, predict, evaluate.

### Directory layout

```
shared/
  configs/
    backends.yaml                       # backend registry
    environments/                       # per-environment context YAMLs
      alfworld_40ms_react_tags.yaml
      frozen_lake_8x8_react_tags.yaml
      open_apps_recovery_som_text.yaml
      tblite_easy_40ms_react_tags.yaml
    collection/<env>/*.yaml             # dataset collection configs (hand-written)
  data/datasets/<env>/*.pkl             # collected Dataset pickles
catalogs/qval_benchmark/
  catalog.py                            # single source of truth
  configs/
    prediction/<env>/*.yaml             # generated prediction + GT configs
    evaluation/<env>/*.yaml             # optional custom-comparison configs
  data/
    predictions/<env>/{EVAL_*,GT_*}/    # per-method prediction JSONs
    evaluations/<env>/                  # correlation summaries
scripts/
  pipeline/{collect_dataset,predict,evaluate}.py
  slurm/                                # cluster wrappers (site-specific)
```

### Example config — `shared/configs/backends.yaml`

```yaml
backends:
    eval:
        type: openai
        model: Qwen/Qwen3-8B
        backend_url: http://localhost:8000/v1
        enable_thinking: true
        sampling:
            temperature: 0.0
            max_tokens: 2048
    rollout:
        type: openai
        model: Qwen/Qwen3-32B
        backend_url: http://localhost:8001/v1
        enable_thinking: true
        sampling:
            temperature: 1.0
            max_tokens: 2048

default_backend: eval
```

### SLURM script (example template)

The repo ships per-config wrappers under `scripts/slurm/` (`run_collection.sh`,
`run_prediction.sh`, `run_evaluation.sh`; see [`getting_started.md`](getting_started.md)
§9). The self-contained single-job template below is an alternative you can adapt
when you want one job to serve a model, collect, predict, and evaluate, then tear
the server down:

```bash
#!/bin/bash
#SBATCH --job-name=vb-frozen-lake
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-0

COLLECT_CONFIG="shared/configs/collection/frozen_lake/8x8_scripted.yaml"
PREDICT_CONFIG="catalogs/qval_benchmark/configs/prediction/frozen_lake/100pt_8x8_q35-27-or_text.yaml"
EVAL_CONFIG="catalogs/qval_benchmark/configs/evaluation/frozen_lake/8x8_q35-27-or_text.yaml"

# Outputs (datasets under shared/, predictions/evaluations under the catalog) are
# created by the pipeline scripts; no need to pre-make them.
mkdir -p logs

# Start vLLM server (matches backends.rollout)
python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-8B" --port 8000 \
    --max-model-len 4096 --gpu-memory-utilization 0.9 &
VLLM_PID=$!

# Wait for ready...

# Step 1: Collect dataset (the environment comes from the collection config)
python scripts/pipeline/collect_dataset.py \
    --config "$COLLECT_CONFIG"

# Step 2: Predict (GT + eval methods). The methods to run come from the config's
# eval_methods / estimations — there is no --methods flag.
python scripts/pipeline/predict.py \
    --config "$PREDICT_CONFIG"

# Step 3: Evaluate. With a single eval config this scores that one comparison;
# omit --config to discover and pair every prediction automatically.
python scripts/pipeline/evaluate.py \
    --config "$EVAL_CONFIG"

kill $VLLM_PID
```

### Submitting

Save the template above (e.g. as `submit_frozen_lake.sh`) and submit it:

```bash
mkdir -p logs
sbatch submit_frozen_lake.sh
```

This submits a single job that starts its own vLLM server, collects data, predicts, evaluates, and shuts down the server.

For a simpler SLURM setup without server management, set `backends.<name>.type: vllm` — the model is loaded in-process and released when the script exits.

### Output Files

**`collect_dataset.py` produces:**
- **`{dataset-path}`** — pickle with evaluation points, trajectory returns, config, metadata. Dataset metadata keeps requested `task_indices` and also records `collected_task_indices` for the trajectories that were actually saved. Runs that collect zero trajectories fail instead of writing an empty dataset.
- **`{dataset-path}.logs.jsonl`** — LLM call logs from the collection phase
- **`{trajectories-path}`** (optional) — human-readable trajectory JSON

**`predict.py` produces:**
- **`{predictions_dir}/gt/{name}_{timestamp}.json`** — per-GT-method prediction JSON
- **`{predictions_dir}/eval/{name}_{timestamp}.json`** — per-eval-method prediction JSON
- **`{predictions_dir}/predict.logs_{timestamp}.jsonl`** — LLM call logs from prediction phase

**`evaluate.py` produces:**
- **`{output-dir}/summary_{timestamp}.json`** — experiment info, dataset config/metadata, correlations, predicted/GT values

**`evaluate.py`** (discovery mode) reads standardized prediction folders under the active catalog's `data/predictions/<env>/` and evaluates every discovered experiment with matching pointwise `qv`/`sv` or ranking `qv` GT and eval artifacts. It compares Q-value methods only with Q-value GTs, state-value methods only with state-value GTs, and ranking Q-value methods only with ranking Q-value GTs. Environment/dataset pairs may have multiple expected GT actors; each present GT actor is evaluated separately, and output directories always include the GT actor token. TerminalBench is Q-value-only, so no TerminalBench state-value jobs are emitted. It skips combinations whose prediction files are absent and logs missing method/model/environment/modality combinations. Use `--dry-run` to inspect discovered jobs without writing summaries and `--missing-report <path>` to write a JSON report containing both missing combinations and every prediction JSON file detected, including whether each file was selected, superseded by a newer timestamp, or skipped for violating the naming convention.

**`logs.jsonl`** — one JSON object per line for every LLM call and event record:

```json
{
    "index": 0,
    "phase": "method:llm_direct",
    "call_type": "generate_chat",
    "timestamp": "...",
    "elapsed_seconds": 0.3,
    "messages": [
        { "role": "system", "content": "..." },
        { "role": "user", "content": "..." }
    ],
    "sampling_params": { "temperature": 0.0, "max_tokens": 2048 },
    "response": {
        "text": "The state value is approximately 0.75...",
        "finish_reason": "END_OF_TEXT",
        "prompt_tokens": 500,
        "completion_tokens": 20,
        "metadata": {}
    },
    "annotation": { "extracted_answer": 0.75 }
}
```

The logs capture full response text including thinking tokens (`<think>...</think>`), making it easy to debug model behavior. If a backend call fails, the run still crashes, but the failing record is written first with the full prompt payload, serialized sampling params, backend metadata, an `error` object, and prompt-size summaries (`message_summary` for single calls, `message_summaries` for batches). Use standard tools like `jq` to filter and analyze:

```bash
# Count LLM calls by phase
jq -r 'select(.call_type) | .phase' logs.jsonl | sort | uniq -c

# Extract all method evaluation responses
jq 'select(.phase | startswith("method:"))' logs.jsonl

# Total tokens used
jq 'select(.call_type) | [.prompt_tokens, .completion_tokens] | add' logs.jsonl | paste -sd+ | bc
```

### Scaling tips

- **More environments**: Loop over environments in a shell script, or use SLURM job arrays with one env per array task. Each pickle is single-env.
- **`batch_size`**: Controls the maximum number of concurrent rollouts/trajectories per lockstep batch. For a single GPU with vLLM, 128-512 is typical.
- **In-process vs. server vLLM**: `backends.<name>.type: vllm` loads the model in-process (simplest). `backends.<name>.type: openai` with a separate `vllm serve` process is useful when sharing a server across scripts or keeping the model loaded between runs.
- **Separate model for rollouts vs. evaluation**: Define multiple entries in `backends` and reference them via `backend_name` in estimation or eval configs.
- **Multi-GPU tensor parallelism**: Set `#SBATCH --gres=gpu:N` and the SLURM scripts automatically pass `--tensor-parallel-size N` to all pipeline scripts. The CLI flag overrides `tensor_parallel_size` on all vLLM backends, so you only change the GPU count in one place. Non-vLLM backends (openai, anthropic, openrouter) ignore the flag. You can also set `tensor_parallel_size` per backend in YAML for non-SLURM use.

## Environment Config Repository

The `shared/configs/environments/` directory contains per-environment context YAML files that provide task/reward descriptions and extractor configuration for dense signal methods. Each file is named `{context_name}.yaml` (house style: `<env>_<variant>`) and gets loaded into a `MethodContext`.

### Directory Layout

```
shared/configs/environments/
  alfworld_40ms_react_tags.yaml
  alfworld_40ms_vision.yaml
  frozen_lake_8x8_react_tags.yaml
  frozen_lake_8x8_vision.yaml
  open_apps_recovery_som_text.yaml
  open_apps_vision_recovery_som.yaml
  tblite_easy_40ms_react_tags.yaml
```

### YAML Format

Each environment config file supports these fields:

| Field | Required | Description |
|-------|----------|-------------|
| `task_description` | No | Natural language description of the environment, prompt format, and expected response |
| `reward_description` | No | Description of the scoring/reward mechanism |
| `evaluator_extractor` | No | Extractor config for parsing evaluator LLM output to numeric values |
| `environment_extractor` | No | Extractor config for parsing actor LLM output into environment actions. Passed to the llenvs adapter's `answer_extractor` parameter. When absent (`None`), the adapter's default extractor is used (e.g., `TagBasedExtractor` for reasoning-gym and GEM, `RawGenerationExtractor` for gymnasium and craftax) |
| `ranking_environment_extractor` | No | Separate extractor for ranking candidate sampling. When set, LLM ranking responses are parsed with this extractor before `env.step(...)`, and Harbor ranking env factories also receive it as `answer_extractor`. If extraction fails, the candidate is skipped instead of sending raw text to the environment. Default: `None` (uses `environment_extractor`) |
| `example_trajectories_file` | No | Path to a JSON file containing example trajectories (relative to the contexts directory). Loaded into `MethodContext.example_trajectories` as `list[ObservableTrajectory]` |
| `system_prompt_file` | No | Path to a text file containing the actor / rollout system prompt (relative to the contexts directory). Used by `collect_dataset.py` and GT rollout construction in `predict.py` when the top-level config does not set `system_prompt` |
| `invalid_action_text` | No | Placeholder assistant text to store in trajectory history when the actor response cannot be extracted into an executable action. Default depends on the adapter; command environments typically use `"[invalid action]"` |
| `invalid_action_observation` | No | Custom reminder shown to the model when action extraction fails. This is prepended to any fallback env observation generated by the adapter |
| `advance_on_invalid` | No | Real fallback command executed by extraction-based adapters when action extraction fails. Use this only when the environment has a verified safe invalid/no-op command (for example Jericho `wait` or the fixed sentinels used by the ALFWorld and WebShop adapters) |
| `include_actor_thinking` | No | When `true`, pass raw actor output (including `<think>` traces) to evaluation methods without extraction or cleaning. Default: `false` |
| `include_current_thoughts` | No | When `true`, show the actor's ReAct reasoning for the current point in an `**Actor's Reasoning:**` section. Default: `false` |
| `include_history_thoughts` | No | When `true`, include `[Turn i - Reasoning]` sections for history turns with extracted thoughts. Default: `false` |
| `include_state_text_when_images` | No | For multimodal context configs, controls whether serialized state text is kept when an observation also has images. Default: `true`. Text-only prompt builders ignore images, so they keep using serialized state text |
| `reward_signal_name` | No | Name of the reward signal to use for trajectory returns and GT estimation. `null` means use total reward across all signals. Default: `"correctness"` |
| `min_action_chars` | No | Floor on action length after budget-aware truncation. When the prompt exceeds the model's context budget, history entries are selectively truncated starting from earliest entries, but no action will be shorter than this many characters. `null` means actions cannot be truncated. Applied during collection, GT rollouts, and eval method prompts |
| `min_observation_chars` | No | Floor on observation length after budget-aware truncation. Same selective truncation as `min_action_chars`, but observations are truncated before actions. `null` means observations cannot be truncated. Applied during collection, GT rollouts, and eval method prompts |
| `min_current_observation_chars` | No | Floor on the live current observation after budget-aware truncation. Unlike `min_observation_chars` (which applies to history observations), this controls the current state shown to the model. Truncated only as a last resort after history and supplementary content are exhausted. `null` means the current observation cannot be truncated. Applied during GT rollouts, eval method prompts, and ranking prompts |

#### Budget-Aware Truncation

When any truncation floor is set and the backend's context length is known, qval applies **multi-phase** truncation:

1. **If total prompt fits** within the available space → no truncation
2. **Truncate history observations** from earliest entry toward latest, each down to `min_observation_chars` at most
3. **If still over** → truncate **history actions** from earliest, each down to `min_action_chars`
4. **If still over** → truncate **supplementary observations** (e.g., candidate next-states in ranking prompts, next-state in direct method) down to `min_observation_chars`
5. **If still over** → truncate **current observation** down to `min_current_observation_chars`
6. **If still over** → accept the overflow (vLLM is the source of truth for the final length check)

The available prompt space is derived from `max_model_len - max_tokens` (from backend config). Token estimation uses a tiered approach:

1. **Exact**: Local backends (vLLM, HuggingFace) expose a tokenizer — uses `len(tokenizer.encode(text))`
2. **Reference**: API backends — uses a fixed reference tokenizer (Qwen3-0.6B, loaded once at experiment start). ~5-10% error vs target model, much better than char heuristic
3. **Heuristic**: Fallback if no tokenizer is available — `chars / 3.5`

When the context length cannot be determined (API backend without `max_model_len` in config), uniform truncation is used as a fallback (each entry truncated to the `min_*_chars` floor).

Text fields use YAML's `>-` folded block scalar: newlines in the file become spaces in the loaded string (prose flows naturally), while blank lines create real paragraph breaks and more-indented lines are preserved literally. This avoids arbitrary newlines in prompts while keeping the YAML readable.

### Extractor Configuration

Extractors are configured using llenvs' `AnswerExtractor` protocol and `answer_extractor_registry`. Each extractor spec is a YAML dictionary with:

| Key | Required | Description |
|-----|----------|-------------|
| `type` | Yes | Registry name: `tag_based`, `numeric`, `regex`, `gsm8k`, `multiple_choice`, `raw`, `boxed`, `last_line`, `code_block`, `pattern_answer`, `single_line`, or `composite` |
| `pre_cleaners` | No | List of pre-extraction cleaners: `strip_thinking_tokens`, `strip_special_tokens`, and other cleaners exposed by llenvs |
| `post_cleaners` | No | List of post-extraction cleaners: `strip_trailing_punctuation`, `strip_surrounding_quotes`, `strip_latex_dollars`, plus parameterized cleaners such as `truncate_tail` |
| *(other keys)* | No | Passed as kwargs to the extractor constructor (e.g., `tag_name: answer` for `tag_based`) |

When `pre_cleaners` or `post_cleaners` are specified, the extractor is wrapped in a `CleanedExtractor`. Without cleaners, the raw extractor is returned.

The `composite` type takes an `extractors` list of sub-specs, tried in order until one succeeds. The `single_line` type takes an `inner` extractor spec, rejects extractions with more than one non-empty line, and can also enforce a `max_chars` limit on the accepted line.

**Fallback extraction**: For short command environments, prefer strict structured extraction first, then a bounded raw fallback such as `single_line(raw, max_chars=96)`, and finally use `truncate_tail` only as a hard cap on already-extracted text. This keeps bare one-line commands recoverable while rejecting multiline blobs and oversized single-line junk. Use permissive unbounded `raw` fallback only in environments where the full raw response is genuinely the action format, such as shell-command settings.

**Two extraction points:**

- **Evaluator extractor** — parses the evaluator LLM's response to get a numeric signal value (used by `LLMDirectMethod`). Default: `CleanedExtractor(NumericExtractor(), pre_cleaners=[strip_thinking_tokens, strip_special_tokens])`
- **Environment extractor** — parses the actor LLM's raw generation into environment actions. Passed to the adapter's `answer_extractor` parameter when creating the environment. Default: `None` (uses the adapter's built-in default, typically `TagBasedExtractor`). Useful when the model's output format doesn't match the adapter's default expectations (e.g., GEM prompts for `\boxed{}` format but the model may produce plain numbers instead).

Action cleaning (stripping thinking tokens, extracting from tags) is handled by llenvs adapters via `extracted_action` and `resolved_action` on `StepResult`/`Transition`. qval uses a three-tier priority chain: `resolved_action` (formatted native action) → `extracted_action` (extractor output) → raw `action.text` with thinking stripped.

### Example

```yaml
task_description: >-
    The agent solves arithmetic problems involving chains of addition and
    subtraction. Each problem is a single expression like "123 + 456 - 78 ="
    with 2-6 integer terms (1-4 digits each).

    The agent must compute the result and respond with the numeric answer
    wrapped in <answer>...</answer> tags.

reward_description: >-
    1.0 if the normalized answer equals the correct result, 0.0 otherwise.

# Extract last number from evaluator response
evaluator_extractor:
    type: numeric
    pre_cleaners: [strip_thinking_tokens, strip_special_tokens]
```

### Composite Extractor Example

When the environment's expected answer format differs from the adapter's default, configure `environment_extractor` to handle multiple formats. For short command environments, use a strict fallback that rejects multiline content and then matches the whole response against a simple command regex. Use permissive `raw` fallback only when the environment truly expects unconstrained raw text. If ranking should remain more permissive than rollout collection, override it separately with `ranking_environment_extractor`:

```yaml
# GEM games prompt for \boxed{} format, but models may produce plain numbers
environment_extractor:
    type: composite
    extractors:
        - type: boxed
        - type: numeric
        - type: raw # fallback: always succeeds, returns full text
    pre_cleaners: [strip_thinking_tokens, strip_special_tokens]
    post_cleaners:
        - type: truncate_tail
          config:
              max_chars: 256 # bound length when structured extractors fail

# Jericho / ALFWorld / Craftax / FrozenLake style: strict single-line fallback
environment_extractor:
    type: composite
    extractors:
        - type: single_line
          inner:
              type: tag_based
              tag_name: action
        - type: regex
          pattern: "^([A-Za-z][A-Za-z0-9'_-]{0,23}(?: [A-Za-z0-9][A-Za-z0-9'_-]{0,23}){0,5})$"
    pre_cleaners: [strip_thinking_tokens, strip_special_tokens]
    post_cleaners:
        - type: truncate_tail
          config:
              max_chars: 96

# TerminalBench: strict collection, permissive ranking
ranking_environment_extractor:
    type: composite
    extractors:
        - type: tag_based
        - type: raw
    pre_cleaners: [strip_thinking_tokens, strip_special_tokens]
```

### Loading Contexts

Use `load_context()` to load an environment config into a `MethodContext`:

```python
from qval import load_context, SignalType

context = load_context("chain_sum", SignalType.STATE_VALUE)
# context.task_description -> "The agent solves arithmetic problems..."
# context.reward_description -> "1.0 if the normalized answer..."
# context.signal_type -> SignalType.STATE_VALUE
# context.evaluator_extractor -> CleanedExtractor(NumericExtractor(), ...)
```

Environment context YAML files may set `prompting_scheme` to control the actor
output format. Valid schemes: `answer_tags` (default), `react_tags`, `react_classic`.
The system prompt is composed from three layers: per-env role description,
per-scheme formatting instructions, and per-env×per-scheme examples (all defined
in `qval.system_prompts`). Collection and GT rollout prediction resolve
system prompts with this precedence:

1. explicit pipeline `system_prompt` override
2. `system_prompt_file` from the environment context YAML (if present)
3. composed three-layer prompt (when adapter is registered in `ENV_PROMPTS`)
4. raw adapter prompt (for unregistered adapters)
5. dataset metadata fallback (prediction GT path only)

#### Jericho Per-Game Prompts

For the `jericho` adapter, the three-layer system composes a rich per-game
prompt when `env_name` is available (e.g., `jericho:zork1`). The prompt is
assembled from 7 fragments defined in `qval.jericho_prompts`:

| Fragment | Scope | Content |
|----------|-------|---------|
| Role & Goal | Universal | Brief intro and score-maximization goal |
| Game Description | Per-game | Genre, setting, and premise |
| Command Structure | Universal | Three command patterns (VERB, VERB OBJ, VERB OBJ PREP OBJ) |
| Available Verbs | Per-game | Categorized verbs from the game's grammar templates |
| Prepositions | Universal | Core prepositions for connecting verbs and objects |
| Scoring | Per-game | Max score and scoring mechanics |
| Game Mechanics | Universal | Inventory, containers, darkness, persistence, parser tips |

Per-game verb references are auto-generated from `jericho.game_info` grammar
templates. Game descriptions and scoring details are hand-authored in
`qval.jericho_prompts.game_data` for all 56 bundled games. Unknown
games fall back to a generic description.

When `env_name` is not provided (e.g., in tests), the system falls back to the
generic `ENV_PROMPTS["jericho"]` string.

A custom directory can be specified:

```python
context = load_context(
    "my_env", SignalType.Q_VALUE, contexts_dir="path/to/environments"
)
```

### Error Handling

- **Missing file**: `FileNotFoundError` with a message listing available contexts in the directory.
- **Unknown keys**: `ValueError` identifying the typo (e.g., `reword_description` instead of `reward_description`).
- **Missing extractor type**: `ValueError` if the extractor spec omits the `type` field.

### Adding New Environment Configs

To add config for a new environment, create a YAML file named `{context_name}.yaml` in the `shared/configs/environments/` directory (house style: `<env>_<variant>`, e.g. `alfworld_40ms_react_tags.yaml`). Reference it by `context_name` (the filename without `.yaml`) — from a collection config, and from the `prediction.context_name` field of a prediction config. Use a filesystem-safe context name even when the underlying `llenvs` adapter id contains special characters.
