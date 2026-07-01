# Truncation System

LLM backends have finite context windows. When trajectories are long or observations are verbose, prompts can exceed the model's `max_model_len`, causing crashes. The truncation system ensures prompts fit within the context window while preserving the most useful content.

## Token Estimation

Token counts are estimated using a 3-tier approach:

1. **Backend tokenizer** (exact) -- used when the backend exposes a tokenizer (vLLM, HuggingFace).
2. **Family-matched reference tokenizer** -- selects a tokenizer from the same model family (e.g., DeepSeek tokenizer for DeepSeek models, Qwen3-0.6B for Qwen models). Falls back to Qwen3-0.6B for unknown families with a 15% safety margin to compensate for possible vocab-size mismatch. Reference tokenizers are loaded on demand, may be downloaded from Hugging Face on first use, and are cached after that.
3. **Character heuristic** (chars / 3.5) -- fallback when no tokenizer can be loaded.

Whenever `make_prompt_budget()` resolves the estimator, it logs which path was selected. If a family-matched reference tokenizer cannot be loaded, the code logs a warning and falls back to the default reference tokenizer when available; otherwise it falls back to the character heuristic. This path does not fail closed.

The prompt token budget is: `max_model_len - max_tokens` (context window minus generation budget).

## Parameters

All floor parameters define the **minimum** number of characters a text component may be truncated **to**. If a floor is `None`, that component is never truncated (it stays full-length regardless of budget pressure).

### Environment-level (defined in `shared/configs/environments/*.yaml`)

| Parameter | Applies to | Description |
|---|---|---|
| `min_action_chars` | History actions | Floor for past action text in trajectory history |
| `min_observation_chars` | History observations | Floor for past observation text in trajectory history |
| `min_current_observation_chars` | Current state | Floor for the current (live) observation the model acts on |
| `min_next_observation_chars` | Next-state observation | Floor for the next-state observation shown to eval methods |
| `min_ranking_candidate_action_chars` | Ranking candidate actions | Floor for each candidate action in ranking prompts |
| `min_ranking_next_observation_chars` | Ranking candidate next-states | Floor for each candidate's resulting observation in ranking prompts |

### Config-level

| Parameter | Source | Description |
|---|---|---|
| `max_history_turns` | `CollectionConfig`, `PredictionConfig` | Fixed turn limit applied before budget-aware truncation |
| `max_model_len` | `BackendConfig` | Maximum sequence length for the model (vLLM) |
| `max_tokens` | `BackendSamplingConfig` | Maximum generation tokens (subtracted from `max_model_len` for prompt budget) |

### Per-method overrides (`EvalMethodConfig`)

Each eval method can override the environment-level defaults and `max_history_turns`:

| Field | Overrides |
|---|---|
| `max_history_turns` | `PredictionConfig.max_history_turns` |
| `min_action_chars` | env context `min_action_chars` |
| `min_observation_chars` | env context `min_observation_chars` |
| `min_current_observation_chars` | env context `min_current_observation_chars` |
| `min_next_observation_chars` | env context `min_next_observation_chars` |
| `max_feedback_chars` | Hard SDPO teacher feedback cap; TerminalBench shell output uses head+tail truncation |

When set on `EvalMethodConfig`, the method-level value takes precedence. When `None` (the default), the environment context value is used.

## Pipeline Stages

### 1. Trajectory Collection (actor prompts)

**Script:** `collect_dataset.py` | **Mechanism:** `PromptBudget` via `TrajectoryRunner`

Parameters used: `min_action_chars`, `min_observation_chars`, `min_current_observation_chars`, `max_history_turns`.

The `PromptBudget` is created from the collection backend's config via `make_prompt_budget()` and passed to `TrajectoryRunner`. The runner computes non-history token costs (system prompt, task, current state) and delegates history construction to `build_budget_aware_history()`.

### 2. Ranking Point Collection (sampling alternative actions)

**Script:** `collect_dataset.py` | **Mechanism:** `PromptBudget` in `_build_sampling_messages()`

Parameters used: same as trajectory collection.

A `PromptBudget` is created from the ranking backend's config and applied when building messages for LLM-based ranking sampling. History is truncated via `prompt_budget.build_history()`, and the current state observation is truncated as a last resort.

### 3. GT MC Rollouts (prediction phase)

**Script:** `predict.py` | **Mechanism:** `PromptBudget` via `TrajectoryRunner` + `_build_gt_history_fn()`

Parameters used: `min_action_chars`, `min_observation_chars`, `min_current_observation_chars`, `max_history_turns`.

The GT estimation backend gets its own `PromptBudget`. A separate `gt_history_fn` applies content truncation via `content_truncated_history()` using `min_action_chars` and `min_observation_chars`.

### 4. LLMDirectMethod (prediction)

**Script:** `predict.py` | **Mechanism:** `_apply_budget_truncation()` in `llm_direct.py`

Multi-phase truncation:

1. **Phase 1 -- History:** Truncates history observations then actions using `budget_aware_truncation()` with `min_observation_chars` / `min_action_chars` floors. Drops oldest entries first, then partially truncates remaining entries.
2. **Phase 2 -- Next-state:** If still over budget, truncates the next-state observation using `min_next_observation_chars` floor.
3. **Phase 3 -- Current state:** Last resort. Truncates the current observation using `min_current_observation_chars` floor.

### 5. LLMRankingMethod (prediction)

**Script:** `predict.py` | **Mechanism:** `_apply_budget_truncation()` in `llm_ranking.py`

Multi-phase truncation (4 phases):

1. **Phase 1 -- History:** Same as LLMDirectMethod.
2. **Phase 2 -- Candidate next-states:** Proportionally truncates each candidate's next-state observation using `min_ranking_next_observation_chars` floor. Iterates up to 3 passes for convergence.
3. **Phase 2b -- Candidate actions:** Proportionally truncates each candidate's action text using `min_ranking_candidate_action_chars` floor. Same iterative approach.
4. **Phase 3 -- Current state:** Same as LLMDirectMethod.

### 5b. SDPORankingMethod (prediction)

**Script:** `predict.py` | **Mechanism:** `_apply_budget_truncation()` in `sdpo_ranking.py`

SDPO first applies any fixed `max_history_turns` setting. For TerminalBench
only, `max_feedback_chars` then hard-caps each candidate's immediate shell
response before prompt assembly, preserving the head and tail and replacing the
middle with a truncation marker. This cap is independent of model-context token
budgeting, so very long shell transcripts are bounded even if the token
estimator would otherwise allow them.

After that hard cap, SDPO applies the same budget-aware phases as direct
prediction: history first, candidate feedback/next-state second, and current
state last.

### 6. LLMCodeGenMethod (prediction)

No truncation applied. Code generation prompts are templates with environment descriptions and example data; the generated code then runs on full observations/actions.

## Budget-Aware Truncation Algorithm

`budget_aware_truncation()` in `token_budget.py` operates on a list of `(observation, action)` pairs:

1. Estimates token cost of each pair (including per-entry overhead).
2. If total exceeds available budget, drops the **oldest** entries first.
3. If still over budget, **partially truncates** remaining entries -- observations first, then actions -- down to their respective floors.
4. Returns the truncated pairs in order.

## Environment-Specific Values

| Environment | `min_action` | `min_obs` | `min_cur_obs` | `min_next_obs` | `min_rank_action` | `min_rank_next_obs` |
|---|---|---|---|---|---|---|
| Frozen Lake | -- | -- | -- | -- | -- | -- |
| ALFWorld | 64 | 256 | 512 | 256 | -- | -- |
| Jericho | 32 | 256 | 512 | 256 | -- | -- |
| Craftax | 32 | 256 | 512 | 256 | -- | -- |
| WebShop | 64 | 512 | 1024 | 512 | -- | -- |
| Terminal Bench (30ms) | 256 | 256 | 512 | 256 | 128 | 128 |
| Terminal Bench | 512 | 512 | 1024 | 512 | 256 | 256 |

`--` means `None` (no truncation).
