# SDPO-GT With Next Expert Step And Rollout Summary

This document describes an SDPO-GT variant for OpenApps, ALFWorld,
TerminalBench, and FrozenLake that gives the feedback-conditioned teacher the
next expert action after each candidate action, plus a structured outcome summary
derived from the full stored expert rollout.

It should be treated as a separate method from the current `sdpo_ranking`.
The current method remains the paper-closer baseline:

1. **`sdpo_ranking`**: teacher sees only the candidate's immediate `next_state`.
2. **`sdpo_short_expert_continuation`**: teacher sees the immediate `next_state`
   plus the next expert command and structured rollout outcome fields recovered
   from cached GT rollouts. The config type keeps its historical name, but the
   official `sdpo-gt_qv` prompt is now the compact next-step-plus-summary
   version.

The second variant is an oracle/privileged-information ablation. It is useful
for testing whether ALFWorld SDPO is poor because one-step textual feedback is
too weak, but it is less faithful to the original SDPO paper than the
next-state-only version.

## Motivation

In ALFWorld, a one-step response can be under-informative. For example, after a
candidate command such as `take peppershaker 2 from cabinet 1`, the immediate
environment response may only say that the object was picked up. That tells the
teacher the command was valid, but it may not fully expose why the command was
high-value: the decisive evidence is often that the resulting state admits an
expert next step and a short path to success.

For TerminalBench, full shell continuations are often too long and noisy. The
compact SDPO-GT variant therefore exposes only the next expert shell command and
the stored rollout's verifier outcome summary. The student prompt is unchanged.
The candidate action text being scored is also unchanged.

OpenApps has a similar leakage risk: a multi-step browser trace can reveal the
target app workflow instead of simply indicating whether the candidate landed in
a useful browser state. FrozenLake is simpler, but a full move sequence would
mostly restate the path. For both environments, the compact next-step-plus-summary
prompt is the cleaner first SDPO-GT variant.

The score remains:

```text
score(s, a) = mean_t [log p_teacher(a_t | state, feedback, expert_evidence) - log p_student(a_t | state)]
```

The important SDPO scoring invariant still holds:

- student and teacher score the exact same candidate action string
- only the prompt context differs
- higher score means the expert-evidence-conditioned teacher assigns more
  probability to the same command tokens relative to the student

`expert_evidence` is prompt evidence only. The method does not directly replace
the SDPO score with the stored GT return.

## ALFWorld Teacher Prompt

The student prompt remains the existing ALFWorld SDPO prompt:

```text
Choose the next ALFWorld command for the following decision point. Use the household objective, current observation, inventory/location state, and admissible commands when they are shown.

[optional state-action history block]

**Current State:**
{STATE_TEXT}

Respond with exactly one valid ALFWorld command and no explanation. Prefer commands that make immediate progress toward the stated household goal while preserving required preconditions. The assistant continuation being scored is that command itself.
```

The SDPO-GT variant changes only the teacher prompt. The teacher still sees the
same current state and immediate candidate response, but it also receives the
next expert command after the candidate result and a structured outcome summary
derived from the full stored expert rollout:

```text
Choose again from the same ALFWorld decision point after seeing the immediate environment response produced by one candidate command and the next expert command from the resulting state, plus a structured outcome summary from the stored expert rollout.

[optional state-action history block]

**Current State:**
{STATE_TEXT}

**Immediate ALFWorld Response After The Candidate Command:**
{FEEDBACK_TEXT_OR_FALLBACK}

**Expert Rollout Evidence After The Candidate Command:**
The expert command below occurs after the candidate command has already been executed in the ALFWorld state.
Use it only as evidence about the original candidate command; it is not a replacement command to output.
Best next expert command after candidate result: {EXPERT_NEXT_COMMAND_OR_UNKNOWN}
Structured stored expert rollout summary:
Stored expert rollout reached goal: {yes|no success observed|unknown}
Expert steps from candidate result to success: {N}
Remaining distance proxy: {N} expert step(s) to task completion
Final observed task reward: {REWARD}
Stored expert rollout outcome: {success|no success observed|unknown}

Use the immediate response, next expert command, and structured expert rollout summary to judge the original candidate command from the current ALFWorld state. Prefer commands that are valid, satisfy preconditions, move the right object or location toward the household goal, reduce expert distance to success, and preserve useful future options. Treat a short successful stored expert rollout as strong positive evidence; treat no observed success as weaker evidence, not proof that the state is unsolvable. Do not reward a command merely because the response or next expert command makes it easy to reconstruct. The assistant continuation being scored is the original candidate command itself, so do not rewrite or replace it.
```

If success is not observed, the prompt replaces the success-step lines with:

```text
Expert steps from candidate result observed: {N}
Remaining distance proxy: task completion was not observed in the stored expert rollout
```

The expert evidence block should be omitted or replaced with
`[No expert continuation was available for this candidate.]` when no matching GT
rollout is found.

## TerminalBench Teacher Prompt

TerminalBench uses the same SDPO-GT scoring invariant, but the evidence block is
more conservative than ALFWorld's. Shell continuations are often longer, noisier,
and more diagnostic than household command chains. The teacher sees the next
expert command plus compact verifier/outcome fields, not a full terminal
transcript.

The implementation uses the existing `sdpo_short_expert_continuation` config type
for compatibility, but `sdpo-gt_qv` is the next-step-plus-summary prompt. The
same rollout-store lookup shape is reused:

```text
(trajectory_index, step_index, candidate_index) -> ExpertContinuation
```

For TerminalBench, `ExpertContinuation` should be extended or specialized to
store compact shell evidence:

- `Best next expert command after candidate result`: the first command after the
  candidate command.
- `Stored expert rollout reached verifier success`: `yes` when the stored
  rollout reaches positive TerminalBench correctness/verifier reward,
  `no success observed` when reward values are present but no success is
  observed, and `unknown` when no usable reward value is available.
- `Expert steps from candidate result to success`: the number of expert shell
  commands after the candidate result until the first positive verifier reward.
- `Expert steps from candidate result observed`: used when success is not
  observed.
- `Final observed verifier reward`: the last usable correctness/verifier reward
  found in the stored rollout.
- `Stored expert rollout outcome`: `success`, `no success observed`, or
  `unknown`.

When multiple stored rollouts exist for one candidate, the TerminalBench version
should choose the highest-return rollout for the next expert command and compute
summary fields from that rollout. A later ablation could expose aggregate success
counts across rollouts, but the first version should keep the prompt small and
comparable to ALFWorld SDPO-GT.

Suggested TerminalBench teacher prompt:

```text
Choose again from the same TerminalBench shell state after seeing the immediate shell response produced by one candidate command, the next expert command from the resulting state, and a structured verifier outcome summary from the stored expert rollout.

[optional state-action history block]

**Current State:**
{STATE_TEXT}

**Immediate Shell Response After The Candidate Command:**
{FEEDBACK_TEXT_OR_FALLBACK}

**Expert Shell Rollout Evidence After The Candidate Command:**
The expert command below occurs after the candidate command has already been executed in the persistent TerminalBench shell session.
Use it only as evidence about the original candidate command; it is not a replacement command to output.
Best next expert command after candidate result: {EXPERT_NEXT_COMMAND_OR_UNKNOWN}
Structured stored expert rollout summary:
Stored expert rollout reached verifier success: {yes|no success observed|unknown}
Expert shell steps from candidate result to success: {N}
Remaining distance proxy: {N} expert shell step(s) to verifier success
Final observed verifier reward: {REWARD}
Stored expert rollout outcome: {success|no success observed|unknown}

Use the immediate shell response, next expert command, and structured verifier outcome summary to judge the original candidate command from the current TerminalBench shell state. Prefer commands that make durable file or configuration progress, produce useful diagnostics or validation evidence, preserve needed shell state, reduce expert distance to verifier success, and avoid redundant probes or noisy dead ends. Treat a short successful stored expert rollout as strong positive evidence; treat no observed success as weaker evidence, not proof that the state is unsolvable. Do not reward a command merely because its output or the next expert command makes it easy to reconstruct. The assistant continuation being scored is the original candidate command itself, so do not rewrite or replace it.
```

If success is not observed, the prompt should replace the success-step lines
with:

```text
Expert shell steps from candidate result observed: {N}
Remaining distance proxy: verifier success was not observed in the stored expert rollout
```

TerminalBench prompt-budget truncation should happen in this order:

1. history
2. immediate shell response
3. current state

The full stored terminal trajectory should remain outside the prompt. It should
be used only to derive the compact continuation and structured verifier outcome
summary.

## OpenApps Teacher Prompt

Initial prompt shape:

```text
Choose again from the same OpenApps browser state after seeing the immediate page response produced by one candidate browser action, the next expert browser action from the resulting state, and a structured outcome summary from the stored expert rollout.

[optional state-action history block]

**Current State:**
{STATE_TEXT}

**Immediate OpenApps Response After The Candidate Browser Action:**
{FEEDBACK_TEXT_OR_FALLBACK}

**Expert Browser Rollout Evidence After The Candidate Browser Action:**
The expert browser action below occurs after the candidate browser action has already been executed in the OpenApps browser state.
Use it only as evidence about the original candidate browser action; it is not a replacement action to output.
Best next expert browser action after candidate result: {EXPERT_NEXT_ACTION_OR_UNKNOWN}
Structured stored expert rollout summary:
Stored expert rollout reached task success: {yes|no success observed|unknown}
Expert browser steps from candidate result to success: {N}
Remaining distance proxy: {N} expert browser step(s) to task success
Final observed task reward: {REWARD}
Stored expert rollout outcome: {success|no success observed|unknown}

Use the immediate page response, next expert browser action, and structured rollout summary to judge the original candidate browser action from the current OpenApps state. Prefer actions that navigate to the correct synthetic app, interact with the intended visible element, enter exact instruction values, resolve or avoid validation errors, reduce expert distance to task success, and preserve useful future browser state. Treat a short successful stored expert rollout as strong positive evidence; treat no observed success as weaker evidence, not proof that the state is unsolvable. Do not reward a browser action merely because the page response or next expert action makes it easy to reconstruct. The assistant continuation being scored is the original candidate browser action itself, so do not rewrite or replace it.
```

Prompt quality analysis:

- The useful privileged signal is whether the candidate led to a browser state
  from which an expert can finish soon.
- Showing a full browser action trace would leak the app workflow and can make
  the teacher score recoverability instead of candidate quality.
- The next expert action is enough to reveal the local direction of recovery or
  progress, while the structured summary provides the long-horizon value signal.

Adjustment: keep the initial prompt shape. The final prompt should remain
next-action-plus-summary only, with no displayed multi-step browser trace.

Current OpenApps GT artifacts provide flat ranking-candidate MC values in:

```text
catalogs/qval_benchmark/data/predictions/open_apps/GT_94pt_eps025-ms45-recovery-som-filtered-zeros-ranking_scripted/gt/ranking_gt_qv_mc_max_<timestamp>.json
catalogs/qval_benchmark/data/predictions/open_apps/GT_94pt_eps025-ms45-recovery-som-filtered-zeros-ranking_scripted/gt/ranking_gt_sv_mc_max_<timestamp>.json
```

These files do not contain persisted rollout shards or next expert browser
actions. When an OpenApps SDPO-GT config points at one of these JSON files, the
teacher evidence should therefore use the discounted ranking GT value as the
structured summary, infer distance-to-success from the discount factor when
possible, and leave the next expert browser action as `unknown`.

If success is not observed, the prompt should replace the success-step lines
with:

```text
Expert browser steps from candidate result observed: {N}
Remaining distance proxy: task success was not observed in the stored expert rollout
```

## FrozenLake Teacher Prompt

Initial prompt shape:

```text
Choose again from the same FrozenLake grid position after seeing the immediate environment response produced by one candidate move, the next expert move from the resulting grid state, and a structured outcome summary from the stored expert rollout.

[optional state-action history block]

**Current State:**
{STATE_TEXT}

**Immediate FrozenLake Response After The Candidate Move:**
{FEEDBACK_TEXT_OR_FALLBACK}

**Expert Rollout Evidence After The Candidate Move:**
The expert move below occurs after the candidate move has already been executed in the FrozenLake grid state.
Use it only as evidence about the original candidate move; it is not a replacement move to output.
Best next expert move after candidate result: {EXPERT_NEXT_MOVE_OR_UNKNOWN}
Structured stored expert rollout summary:
Stored expert rollout reached goal: {yes|no success observed|unknown}
Expert moves from candidate result to success: {N}
Remaining distance proxy: {N} expert move(s) to goal
Final observed environment reward: {REWARD}
Stored expert rollout outcome: {success|no success observed|unknown}

Use the immediate grid response, next expert move, and structured rollout summary to judge the original candidate move from the current FrozenLake grid state. Prefer moves that keep the agent on safe ice, avoid holes and off-grid no-ops, reduce expert distance to the goal, preserve future safe moves, and use the remaining step budget efficiently. Treat a short successful stored expert rollout as strong positive evidence; treat no observed success as weaker evidence, not proof that the state is unsolvable. Do not reward a move merely because the response or next expert move makes it easy to reconstruct. The assistant continuation being scored is the original candidate move itself, so do not rewrite or replace it.
```

Prompt quality analysis:

- FrozenLake has a small action space, so exposing a full expert path would add
  little beyond the structured distance-to-goal summary.
- The next expert move is useful because it disambiguates whether the candidate
  landed on a viable path, entered a recovery state, or moved toward a dead end.
- The immediate response still matters for holes, terminal success, off-grid
  no-ops, and backtracking.

Adjustment: keep the initial prompt shape. The prompt is already compact and
should not display the full path unless a separate stronger oracle ablation is
needed.

If success is not observed, the prompt should replace the success-step lines
with:

```text
Expert moves from candidate result observed: {N}
Remaining distance proxy: goal was not observed in the stored expert rollout
```

## Expert Continuation Source

For the current ALFWorld data, the relevant cached GT artifacts are under:

```text
catalogs/qval_benchmark/data/predictions/alfworld/GT_200pt_all-types-40ms_scripted/
```

The flattened numeric GT files in `gt/*.json` are not sufficient for the prompt
augmentation. They contain Q-values and metadata, but not the expert action
sequence.

The useful data lives in the rollout store:

```text
catalogs/qval_benchmark/data/predictions/alfworld/GT_200pt_all-types-40ms_scripted/rollouts/ranking_candidates/state_action/{dataset_fingerprint}/{generation_hash}/
```

That store contains `manifest.json` plus `shards/shard_*.pkl`. Each shard maps a
ranking candidate key to one stored rollout:

```text
{trajectory_index}:{step_index}:{candidate_index}
```

Example keys:

```text
0:1:0
0:1:1
0:1:2
0:1:3
```

Each value is a tuple of `StoredTrajectory` objects. For the scripted GT store
used here, there is usually one rollout per candidate. The rollout stores:

- `initial_state`: the state where the candidate command is attempted
- `transitions`: the candidate command followed by scripted expert commands

The first transition action is the candidate command. The following transitions
are the stored expert rollout from the candidate result. The prompt displays
only the first expert action after the candidate result, but the structured
outcome summary is computed from the full stored trajectory. Some ALFWorld
next-state observations were generated with `expose_expert_plan_in_obs: true`,
so they may also include markers such as:

```text
[expert_plan_next: go to drawer 1]
```

The implementation should prefer `transitions[1]` for the compact next-step
evidence and use `[expert_plan_next: ...]` only as a fallback for the
`Best next expert command after candidate result` line.

## Structured Outcome Extraction

For each stored trajectory, the implementation derives the structured fields
from `StoredTrajectory.transitions`:

- `Best next expert command after candidate result`: the first expert action
  after the candidate,
  i.e. `transitions[1]`, formatted with the same action display helper used for
  candidates. If no expert action is available, it falls back to
  `[expert_plan_next: ...]` from the candidate transition's `next_state`.
- `Stored expert rollout reached goal`: `yes` when any transition in the stored
  trajectory has positive task reward, `no success observed` when reward values
  are present but no positive reward is observed, and `unknown` when no usable
  reward value is found.
- `Expert steps from candidate result to success`: the index of the first
  positive-reward transition. Because `transitions[0]` is the candidate command,
  a success on `transitions[1]` is reported as `1` expert step from the candidate
  result to success.
- `Expert steps from candidate result observed`: `len(transitions) - 1`, used
  when success is not observed.
- `Remaining distance proxy`: the success step count when success is observed;
  otherwise a statement that completion was not observed in the stored expert
  rollout.
- `Final observed task reward`: the last reward value encountered while scanning
  the trajectory.
- `Stored expert rollout outcome`: `success`, `no success observed`, or
  `unknown`, matching the reached-goal status. This is intentionally named as a
  stored rollout outcome because it is computed from the full stored trajectory,
  not only the single displayed next expert command.

Reward extraction prefers ALFWorld/OpenApps `task_completion`, then
TerminalBench-style `correctness`, then FrozenLake-style `gym_reward`. For
compatibility with other stored reward bundles, it also checks matching signals
in `rewards.signals` and finally falls back to `rewards.total`.

## Implementation Sketch

The existing implementation is in:

```text
src/qval/methods/sdpo_ranking.py
```

In the shipped implementation this reuses `type: sdpo_ranking`; a distinct method
name (`sdpo-gt_qv`, vs the paper-closer `sdpo_qv`) keeps the two sets of results
separate without a separate method type. The config shape:

```yaml
eval_methods:
  - name: sdpo-gt_qv
    type: sdpo_ranking
    signal_type: q_value
    prompt_preset: v2
    policy_assumption: optimal
    backend_name: qwen35_9_thinking_text_or
    expert_rollout_store: catalogs/qval_benchmark/data/predictions/alfworld/GT_200pt_all-types-40ms_scripted/rollouts/ranking_candidates/state_action/{dataset_fingerprint}/{generation_hash}
```

Existing configs may still set `max_expert_continuation_steps` from the earlier
multi-command design. The current `sdpo-gt_qv` prompt displays only the first
expert command after the candidate result, so that field is kept only for
backward config compatibility.

The method should:

1. Load the rollout store manifest and shards once at construction time.
2. Build a lookup:

```text
(trajectory_index, step_index, candidate_index) -> ExpertContinuation
```

3. For each ranking candidate, extract:
   - immediate response: existing `candidate.next_state`
   - best next expert action: first expert action after the candidate action
   - structured outcome fields: reached-goal status, steps to success or
     observed expert steps, final observed task reward, and terminal outcome
4. Append the next expert command and structured outcome block only to the
   teacher prompt.
5. Score the same candidate action text under the unchanged student prompt and
   the augmented teacher prompt.
6. Save predictions with metadata that identifies the privileged context:

```json
{
  "score_formula": "mean_token_logprob_delta_teacher_with_next_expert_step_summary_minus_student",
  "teacher_feedback": "next_state_plus_next_expert_step_and_rollout_summary",
  "expert_rollout_store": "..."
}
```

For eval artifacts, the official saved method name and file basename prefix is:

```text
sdpo-gt_qv
```

This means the saved JSON should contain `"method_name": "sdpo-gt_qv"` and the
file should be named like `eval_sdpo-gt_qv_<timestamp>.json` inside the `eval/`
folder, matching the standard eval artifact prefix convention.

The rollout lookup must use the ranking point order actually passed to
`score_batch`. The `candidate_index` is the candidate's index within
`RankingPoint.candidates`, matching the order used by ranking GT rollout keys.

## Truncation

The existing SDPO prompt-budget logic truncates history, current state, and
next-state feedback. The SDPO-GT block is intentionally compact: one next expert
command plus a structured rollout summary. It should not need a separate raw
continuation truncation target.

If prompts still exceed budget, truncate in this order:

1. history
2. immediate next-state feedback
3. current state

The raw expert continuation should not be expanded to full trajectories in
`sdpo-gt_qv`. The full stored trajectory is used only for the compact structured
outcome summary. A full-command-continuation oracle can be a separate ablation if
needed.

## Interpretation

This method asks:

> How much more likely does the model find the candidate action after it sees
> the immediate response, the next expert command from the resulting state, and
> explicit structured evidence about whether that expert rollout reached task
> completion or verifier success?

That is different from the default SDPO question:

> How much more likely does the model find the candidate action after it sees
> only the immediate environment response?

The structured outcome summary is meant to test whether current SDPO is limited
by weak one-step teacher information while avoiding the noise and leakage of a
multi-command expert transcript. If this variant improves substantially, that
supports the hypothesis that the teacher needs compact long-horizon value
evidence. If it still does not improve, that points more strongly at the
teacher-student logprob-delta scoring mechanism itself.
