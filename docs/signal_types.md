# Signal Types

qval evaluates dense signal functions of three types. The `SignalType` enum specifies which type is being evaluated, and the ground truth estimator computes the corresponding ground truth.

## STATE_VALUE — V(s)

The value of being in state `s`: the expected cumulative reward from state `s` onward.

**Ground truth estimation:** Sample N rollouts from state `s`, aggregate the cumulative returns.

$$V(s) = \text{agg}(G_1, G_2, \ldots, G_N)$$

where $G_i = \sum_t r_t^{(i)}$ is the total cumulative return (sum of all rewards) of rollout $i$, and $\text{agg}$ is the aggregation function (mean or max, configured via `EstimationConfig.aggregation`).

**Assumptions:**
- Discount factor λ = 1.0.
- Works with both sparse terminal rewards (where intermediate rewards are zero) and dense per-step rewards.

## Q_VALUE — Q(s, a)

The value of taking action `a` in state `s`: the expected cumulative reward when starting from `s`, taking `a`, then following the rollout policy.

**Ground truth estimation:** Force action `a` at state `s`, then sample N rollouts from the resulting next state. Aggregate the cumulative returns.

$$Q(s, a) = \text{agg}(G_1, G_2, \ldots, G_N)$$

where each rollout begins by taking action `a` in state `s`, and $G_i$ is the total cumulative return.

## ADVANTAGE — A(s, a)

How much better (or worse) action `a` is compared to the average action in state `s`. Computed as the difference between the value after taking the action and the value of the current state.

**Ground truth estimation:**

$$A(s, a) = \text{agg}(V(s')) - \text{agg}(V(s))$$

where `s'` is the next state after taking action `a`. Both V(s') and V(s) are estimated via MC rollouts (N rollouts each), with the aggregation function applied independently to each set of returns.

This simplification holds because:
- R(s, a) = 0 for non-terminal states (sparse reward assumption); for environments with per-step rewards, the full formula $A(s, a) = R(s, a) + \lambda V(s') - V(s)$ should be used.
- λ = 1.0 (no discounting).

## Aggregation Methods

The `EstimationConfig.aggregation` field controls how rollout returns are combined:

- **MEAN** — `sum(returns) / len(returns)`. Estimates the expected value under the rollout policy. Standard MC estimation.
- **MAX** — `max(returns)`. Approximates the value under a best-of-N policy. Useful when the evaluated method targets an optimal policy rather than the actor's average behavior.

Multiple estimation configs can be specified to compute both ground truths in a single run, correlating method predictions against each independently.

## DenseSignalFunction Protocol

All methods being evaluated implement the same interface regardless of signal type:

```python
class DenseSignalFunction(Protocol):
    def __call__(self, state: Any, action: Any, next_state: Any) -> float: ...
```

The method receives the full `(state, action, next_state)` triple. Which components it uses is up to the method — a V(s) method may only look at `state`, while a Q(s,a) method also uses `action`.

## DenseSignalMethod ABC

For methods that need context or setup, subclass `DenseSignalMethod` and implement `evaluate()`:

```python
from qval import DenseSignalMethod, MethodContext, EvaluationPoint


class MyMethod(DenseSignalMethod):
    def __init__(self, context: MethodContext) -> None:
        super().__init__(context)

    def evaluate(self, point: EvaluationPoint) -> float:
        # Access self.context.signal_type, task_description, etc.
        return 0.5
```

`DenseSignalMethod` satisfies both `DenseSignalFunction` (via `__call__`) and `BatchDenseSignalFunction` (via `evaluate_batch`).

## Built-in LLM Methods

### LLMDirectMethod

Prompts an LLM per evaluation point to produce a numeric signal value.

```python
from qval import LLMDirectMethod, MethodContext, SignalType
from qval.prompt_presets import get_preset

context = MethodContext(
    signal_type=SignalType.Q_VALUE,
    task_description="A maze navigation task where the agent must reach a goal.",
    reward_description="1.0 if the agent reaches the goal, 0.0 otherwise.",
    example_trajectories=trajectory_results,  # optional
)
method = LLMDirectMethod(
    backend=my_backend, context=context, prompt_preset=get_preset("v2")
)

# Evaluate on points
values = method.evaluate_batch(evaluation_points)
```

### LLMCodeGenMethod

Prompts an LLM once to generate a Python signal function, compiles it, and executes it per point. The generated function's signature varies by signal type:

- **STATE_VALUE / POTENTIAL**: `signal_function(state: str) -> float`
- **Q_VALUE / ADVANTAGE / SHAPED_REWARD** (default): `signal_function(state: str, action: str, next_state: str) -> float`
- **Q_VALUE / ADVANTAGE / SHAPED_REWARD** with `include_next_state: false`: `signal_function(state: str, action: str) -> float`

```python
from qval import LLMCodeGenMethod, MethodContext, SignalType
from qval.prompt_presets import get_preset

context = MethodContext(
    signal_type=SignalType.STATE_VALUE,
    task_description="A number guessing game.",
)
method = LLMCodeGenMethod(
    backend=my_backend, context=context, prompt_preset=get_preset("v2")
)

# Optionally trigger generation early
method.generate()
print(method.generated_code)  # inspect the generated code

# Evaluate on points
values = method.evaluate_batch(evaluation_points)
```

The generated code runs in a restricted sandbox: a curated set of pure builtins (including common exception classes so `try/except` works) and the standard library modules `collections`, `itertools`, `json`, `math`, `re`, `statistics`, and `string`. The whitelist is maintained as footgun prevention, not a security boundary. Introspection builtins (`locals`, `globals`, `vars`, `dir`, `eval`, `exec`) are deliberately not exposed and raise `NameError` at runtime; the codegen prompt informs the LLM of this so generated code does not attempt the common `return total, locals()` shortcut for assembling result dictionaries. Runtime errors in the generated function return NaN as a fallback, as do extraction failures in `LLMDirectMethod`. NaN values are automatically filtered out during correlation computation.

## Correlation

Ground truth and predicted values are compared via correlation metrics. Pairs where either value is NaN are automatically filtered before computation; `num_points` in results reflects the post-filtering count. At least 2 non-NaN points are required.

Available metrics:

- **Pearson** (`CorrelationMethod.PEARSON`) — linear correlation.
- **Spearman** (`CorrelationMethod.SPEARMAN`) — rank correlation, robust to monotonic transformations.
- **Kendall's Tau** (`CorrelationMethod.KENDALL_TAU`) — rank correlation based on concordant/discordant pairs.
- **Sign agreement** (`CorrelationMethod.SIGN_AGREEMENT`) — fraction of pairs where the predicted and ground-truth signals share the same sign, with a binomial-test p-value.

Pearson, Spearman, and Kendall's Tau return a coefficient in [-1, 1]; sign agreement returns a fraction in [0, 1]. All return a p-value.
