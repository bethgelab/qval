# Extending QVal

QVal is a testbed, not a fixed set of experiments: you can evaluate **any**
dense-signal method, on **any** model, in **any** environment. This guide covers
the three extension points:

- [A. Your own method](#a-your-own-method)
- [B. A different model](#b-a-different-model)
- [C. A new environment and dataset](#c-a-new-environment-and-dataset)

It assumes you've read [`getting_started.md`](getting_started.md) (the run
loop) and refers to [`registry.md`](registry.md) and
[`configuration.md`](configuration.md) for full field references.

Everything is anchored on the registry catalog
(`catalogs/qval_benchmark/catalog.py`). Adding a model, environment, or method
means appending a spec there and then regenerating configs
(`python scripts/generate_configs.py`).

---

## A. Your own method

A method is anything that turns a `(state, action, next_state)` step into a
scalar. In code it is a subclass of `DenseSignalMethod`
(`src/qval/dense_signal.py`):

```python
from qval.dense_signal import DenseSignalMethod
from qval.types import EvaluationPoint, MethodContext


class MyMethod(DenseSignalMethod):
    def __init__(self, context: MethodContext, *, my_param: float = 1.0) -> None:
        super().__init__(context)
        self.my_param = my_param

    def evaluate(self, point: EvaluationPoint) -> float:
        # point.state, point.action, point.next_state, point.history, …
        # self.context carries the task/reward descriptions, signal_type,
        # policy_assumption, discount_factor, example trajectories, etc.
        return float(...)
```

What you implement:

- **`evaluate(point) -> float`** is required. `EvaluationPoint` carries the step
  plus its `history`, `trajectory_index`, `step_index`, and (when relevant) the
  precomputed action representations and replay information.
- **`evaluate_batch(points) -> list[float]`** is optional but worth overriding
  for anything that batches well (LLM inference, GPU scoring). The default
  applies `evaluate` point-by-point.
- The base class supplies `__call__` (the bare
  `(state, action, next_state) -> float` interface) for free.
- **Fail soft:** on a per-point failure, return `math.nan` rather than raising.
  NaN pairs are filtered out before the correlation is computed, and a fully
  failed method is reported rather than crashing the run.

The method reads everything it needs from `self.context` (a `MethodContext`):
the task and reward descriptions, the target `signal_type` (state-value vs.
Q-value vs. advantage), the `policy_assumption`, the discount factor, and any
example trajectories. The same method class is reused for every signal type —
branch on `self.context.signal_type` if your method behaves differently for,
say, Q-value vs. state-value.

**Ranking methods** are a variant: instead of scoring one action they rank a set
of candidate actions at a state. They operate on `RankingPoint`s and expose
`rank_batch` / `score_batch`. If your method is comparative by nature, model it
as a ranking method (set `is_ranking=True` on its `MethodSpec`).

### Wiring it in

1. **Construction.** `src/qval/method_factory.py` is the single place
   that turns an `EvalMethodConfig` into a constructed method instance (the
   prediction pipeline and the test-time-scaling experiment both go through it).
   Add a branch for your method's `type`, following the existing cases
   (`LLMDirectMethod`, `LLMCodeGenMethod`, the `vle`/`vip`/`liv` builders, …).
   Individual-prediction methods are built by `build_regular_method`; ranking
   methods by `build_ranking_method` (register the `type` in
   `RANKING_METHOD_TYPES`), and image-consuming methods in `VISION_METHOD_TYPES`.
2. **Catalog entry.** Append a `MethodSpec` to `METHODS` in `catalog.py`:

   ```python
   MethodSpec(
       "my-method",        # base; on-disk method_name becomes my-method_<qv|sv>
       "my_method_type",   # EvalMethodConfig.type — must match the factory branch
       "MyFamily",         # family (grouping label)
       "my-method",        # display label
       (QV, SV),           # which signals it produces
       (TEXT,),            # which modalities it applies to
       extra_fields=(("my_param", 1.0),),   # rendered verbatim into the config
   )
   ```

   `extra_fields` are environment-agnostic config fields. For fields that are
   genuinely environment-specific (e.g. per-environment prompts), use
   `EnvironmentSpec.method_params` instead — see how the `vle-*` methods pull
   their negative goals / baseline prompt from the environment.
3. **Regenerate** (`python scripts/generate_configs.py`) and run the pipeline.
   Your method now appears in the generated prediction configs for every
   applicable `(env, actor, modality)`.

Look at `src/qval/methods/llm_direct.py` (an LLM prompted for a number)
and `src/qval/methods/vle/method.py` (a frozen embedding scorer) as two
ends of the spectrum.

---

## B. A different model

A model is an **actor** in the catalog plus a **backend** that serves it.

1. **Backend.** Add an entry to `shared/configs/backends.yaml` (hand-maintained — never
   generated):

   ```yaml
   backends:
     my_model_or:
       type: openrouter            # or vllm / huggingface / …
       model: "org/My-Model"
       enable_thinking: false
       sampling:
         temperature: 0.7
         max_tokens: 2048
   ```

2. **Actor.** Add a `ModelSpec` to `MODELS` in `catalog.py`, referencing the
   backend by name:

   ```python
   ModelSpec(
       "my-model",                 # actor token (appears in output paths)
       "My Model",                 # display label
       backend_base="my_model_or",
       backend_codegen="my_model_codegen_or",  # optional; falls back to base
       backend_vision="my_model_vision_or",    # optional; falls back to base
       supports_vision=False,
   )
   ```

`backend_for(codegen=…, vision=…)` resolves which backend a given usage takes,
falling back to `backend_base`. Capability flags drive what the actor is used
for:

- `supports_vision`, `supports_thinking`, `supports_multi_image` — modality and
  decoding capabilities.
- `is_embedding_backbone` — a CLIP/SigLIP-style backbone used *only* by the
  `vle-*` embedding methods.
- `is_pretrained` — the `self` pseudo-actor used by the pinned pre-trained
  methods (`vip`, `liv-*`), which bring their own backbone.
- `is_gt_only` — produces ground truth (a scripted policy, or a strong agent
  used for GT rollouts), never evaluated as a method.
- `include_in_results` — set `False` to keep an actor runnable but out of the
  results tables.

**Applicability** (the rules in `src/qval/registry/applicability.py`)
then decides which methods attach to your actor: ordinary prompt/code LLM
methods run on LLM actors; the `vle-*` methods run on `clip`/`siglip`
backbones; the pinned pre-trained methods run on `self`. Vision predictions are
Q-value-only for plain LLM methods. Regenerate configs and the new actor's
prediction configs appear under each applicable environment.

---

## C. A new environment and dataset

Onboarding an environment is three pieces: an environment-context YAML, an
`EnvironmentSpec`, and a collected `Dataset`.

### 1. Environment context

Write a context YAML under `shared/configs/environments/`. It tells QVal which `llenvs`
adapter to use, how to construct the environment, how to describe the task and
reward to a method, and how to parse model output into actions. A trimmed
example:

```yaml
adapter: alfworld                       # the llenvs adapter
env_name: alfworld:eval_out_of_distribution
prompting_scheme: react_tags
max_steps: 40
make_kwargs:                            # passed to the adapter
  task_types: [1]
  include_admissible_commands: true
reward_signal_name: task_completion     # the sparse reward signal to track
task_description: >-
  The agent acts in ALFWorld, a multi-turn household environment …
reward_description: >-
  The native reward is sparse and outcome-based: 1.0 on success, 0.0 otherwise.
environment_extractor:                  # model text → an environment action
  type: composite
  pre_cleaners: [strip_thinking_tokens, strip_special_tokens]
  extractors:
    - type: tag_based
      tag_name: action
    - type: raw
evaluator_extractor:                    # method text → a numeric signal
  type: numeric
  pre_cleaners: [strip_thinking_tokens, strip_special_tokens]
```

Useful extras: `signal_interpretation` (an environment-specific gloss on what a
value means, surfaced to methods), the `min_*_chars` truncation thresholds for
prompt budgeting, and `ranking_environment_extractor` for ranking candidates.
Full field documentation is in [`configuration.md`](configuration.md). The
underlying environments come from `llenvs` — see its docs (`~/dev/llenvs/docs`)
for the available adapters and their `make_kwargs`.

### 2. Environment spec

Add an `EnvironmentSpec` to `ENVIRONMENTS` in `catalog.py`:

```python
EnvironmentSpec(
    env="my_env",                       # on-disk token
    display="MyEnv",
    dataset_id="my-slice",              # hyphenated id, appears in dir names
    dataset_path="shared/data/datasets/my_env/my_slice.pkl",
    supports_vision=False,
    gt_actors=("scripted",),            # who produces ground truth
    text_actors=_LLM_ACTORS_TEXT,       # which actors are evaluated (text)
    text_points=100,
    gt_points=200,
    context_name="my_env_react_tags",   # the YAML above, without .yaml
                                        # (house style: <env>_<variant>)
    gt_policy_name="my_env_optimal",    # a scripted policy for GT rollouts …
)
```

Key choices:

- **Point counts** (`text_points`, `vision_points`, `vision_nonllm_points`,
  `gt_points`) set the `<N>pt` token and how many evaluation points each slice
  uses.
- **Ground-truth source.** A scripted-policy environment names a
  `gt_policy_name` (see `src/qval/optimal_policies`); an environment whose
  ground truth comes from a strong agent instead lists that agent in `gt_actors`
  and rolls out with its backend. `gt_aggregations` selects the Monte-Carlo
  aggregation(s) (`max`, `mean`).
- **`supports_sv` / `supports_vision`** gate which signal types and modalities
  are generated for the environment.
- **`method_params`** carries any per-method, environment-specific config fields.

### 3. Collect the dataset

Write a collection config (under `shared/configs/collection/my_env/`) that
rolls a policy through the environment and samples evaluation points, then run:

```bash
python scripts/pipeline/collect_dataset.py --config <my_env-collection.yaml>
```

This writes the `Dataset` pickle to the config's `dataset_path` — the same path
the `EnvironmentSpec` points at. A `Dataset` holds the evaluation points
(state-action pairs with their histories), the per-trajectory returns, optional
ranking candidates, and metadata. Once it exists, `predict.py` and
`evaluate.py` work exactly as in the getting-started guide.

### Onboarding checklist

- [ ] `shared/configs/environments/<my_env>_<variant>.yaml` written (adapter, descriptions,
      extractors).
- [ ] `EnvironmentSpec` added to `catalog.py` (dataset id/path, points, GT source,
      context name).
- [ ] Collection config written and `collect_dataset.py` run → `Dataset` pickle
      at `dataset_path`.
- [ ] `python scripts/generate_configs.py` → prediction/GT configs appear under
      `catalogs/qval_benchmark/configs/prediction/<my_env>/`.
- [ ] `predict.py` then `evaluate.py` → a `summary_*.json` with correlations.
