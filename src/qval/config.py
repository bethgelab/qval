"""Configuration dataclasses for qval."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from qval.types import (
    AggregationMethod,
    CorrelationMethod,
    PolicyAssumption,
    SignalType,
)

_log = logging.getLogger(__name__)


def _parse_enum(value: str, enum_cls: type, field_name: str) -> Any:
    """Parse a string into an enum member (case-insensitive)."""
    try:
        return enum_cls[value.upper()]  # type: ignore[index]
    except KeyError:
        valid = [e.name.lower() for e in enum_cls]  # type: ignore[operator]
        raise ValueError(
            f"Invalid {field_name}: {value!r}. Valid values: {valid}"
        ) from None


def _check_unknown_keys(data: dict, cls: type, context: str = "") -> None:
    """Raise ValueError if data contains keys not in cls's dataclass fields."""
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        label = f" in {context}" if context else ""
        raise ValueError(
            f"Unknown keys{label}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {sorted(valid)}"
        )


def _parse_positive_int(value: Any, field_name: str) -> int:
    """Parse a strictly positive integer config value."""
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class EstimationConfig:
    """Configuration for a ground truth estimation method.

    Supports MC and MCTS estimation with MEAN or MAX aggregation.

    Attributes:
        name: Human-readable name for this estimation (used in results).
        assumption: How to interpret method output for GT comparison.
            Determines what GT to compute and how to compare.
        method: Estimation algorithm ("mc" or "mcts"). Ignored for
            SHAPED_REWARD assumption (no rollouts needed).
        aggregation: How to aggregate rollout returns (MEAN or MAX).
        num_rollouts: Number of rollouts for MC, or simulations per MCTS step.
        mcts_iterations: Number of MCTS search tree iterations.
        mcts_expansion_width: Number of children per expansion (MCTS).
        mcts_exploration_constant: UCB1 exploration constant (MCTS).
        backend_name: Name of backend entry to use for rollouts.
        policy_name: Name of scripted policy entry to use for rollouts.
    """

    name: str = "mc_mean"
    assumption: SignalType = SignalType.STATE_VALUE
    method: str = "mc"
    aggregation: AggregationMethod = AggregationMethod.MEAN
    num_rollouts: int = 32
    mcts_iterations: int = 100
    mcts_expansion_width: int = 5
    mcts_exploration_constant: float = 1.41
    backend_name: str | None = None
    policy_name: str | None = None


@dataclass(frozen=True)
class BackendSamplingConfig:
    """Sampling parameters for a backend.

    These are per-call settings passed at generation time and do not affect
    how the backend or model is initialized.

    Attributes:
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling cutoff.
        max_tokens: Maximum tokens per generation.
        thinking_budget: Max tokens allowed in a <think> block.
        thinking_budget_soft_ratio: Soft ratio for thinking budget (0.0-1.0).
        early_stopping_suffix: When True and thinking_budget is set, pass
            DEFAULT_EARLY_STOPPING_SUFFIX as thinking_budget_suffix on
            SamplingParams.
        reasoning_effort: OpenRouter reasoning effort level. Valid values are
            "xhigh", "high", "medium", "low", "minimal", and "none".
            Mutually exclusive with thinking_budget for OpenRouter backends.
        reasoning_exclude: OpenRouter reasoning.exclude flag. When True,
            OpenRouter may still spend reasoning tokens but will not return
            reasoning text in the response.
        second_elicitation: When True, enable a no-thinking follow-up call on
            MAX_TOKENS that asks the model to follow the formatting
            instructions specified above exactly.
        second_elicitation_max_tokens: Token budget for the follow-up call.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int | None = 2048
    thinking_budget: int | None = None
    thinking_budget_soft_ratio: float | None = None
    early_stopping_suffix: bool = True
    reasoning_effort: str | None = None
    reasoning_exclude: bool | None = None
    second_elicitation: bool = False
    second_elicitation_max_tokens: int = 256


@dataclass(frozen=True)
class OpenRouterProviderPreferences:
    """OpenRouter request-body ``provider`` routing preferences.

    Fields map 1:1 to OpenRouter's provider-routing schema
    (openrouter.ai/docs/features/provider-routing). Forwarded verbatim on
    every chat-completion request via the OpenAI SDK's ``extra_body`` kwarg.

    List-valued fields use ``tuple[str, ...]`` (not ``list[str]``) so
    BackendConfig stays hashable — it's used as a dict key in predict.py's
    backend cache.

    Only valid when the enclosing BackendConfig has ``type == "openrouter"``;
    the YAML parser rejects other combinations at load time. ``None`` means
    "unset" (omitted from the request); an explicit empty tuple means
    "empty value" and is forwarded as an empty JSON array.

    Attributes:
        order: Provider preference order; OpenRouter tries them in sequence.
        only: Whitelist — restricts routing to these providers.
        ignore: Blacklist — excludes these providers.
        allow_fallbacks: If ``False``, fail the request rather than routing
            to a provider outside ``order``/``only``.
        require_parameters: If ``True``, only route to providers supporting
            every parameter in the request.
        data_collection: ``"allow"`` or ``"deny"`` — forbids routing to
            providers that log prompts.
        quantizations: Filter providers by quantization variant (e.g.
            ``("fp8", "bf16")``).
        sort: Routing priority: ``"price"``, ``"throughput"``, or
            ``"latency"``.
    """

    order: tuple[str, ...] | None = None
    only: tuple[str, ...] | None = None
    ignore: tuple[str, ...] | None = None
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    data_collection: str | None = None
    quantizations: tuple[str, ...] | None = None
    sort: str | None = None

    def to_request_dict(self) -> dict[str, Any]:
        """Build the JSON-ready dict for OpenRouter's ``provider`` body field.

        Skips fields whose value is ``None`` so an all-unset instance
        returns ``{}``. Tuples are converted back to lists for JSON
        serialization.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for a named backend.

    Most fields are used when constructing the backend instance; the backend
    cache (``script_utils.backend_cache_key``) reuses one instance across
    configs that differ only in per-request fields. ``sampling`` is always
    per-request. ``enable_thinking`` is per-request for every backend type
    except in-process ``vllm``/``huggingface`` (which bake it into the
    instance via ``chat_template_kwargs``); for all other types — including
    ``vllm_singularity`` — two configs differing only in ``enable_thinking``
    (or sampling) share a single backend instance.

    Attributes:
        type: Backend type (openai, huggingface, vllm, vllm_singularity,
            anthropic, openrouter, litellm, codex, claude_code).
        model: Model name or path. For ``litellm`` use litellm's
            ``provider/model`` format, e.g. ``"gemini/gemini-2.5-flash"`` or
            ``"litellm_proxy/<model>"`` for a LiteLLM proxy/gateway.
        backend_url: URL for OpenAI-compatible backends.
        api_base: Custom endpoint URL for the ``litellm`` backend (e.g. a
            LiteLLM proxy/gateway). ``None`` lets litellm pick the
            provider's default endpoint (or read ``LITELLM_PROXY_API_BASE``
            for ``litellm_proxy/`` models). API keys are never configured
            here — litellm reads the provider's native env var
            (``LITELLM_PROXY_API_KEY``, ``GEMINI_API_KEY``, ...). Only
            valid when ``type == "litellm"``; the YAML parser raises
            otherwise.
        device: Device for HuggingFace backend.
        dtype: dtype for HuggingFace/vLLM/vllm_singularity backends.
        gpu_memory_utilization: Fraction of GPU memory for
            vLLM/vllm_singularity.
        max_model_len: Maximum sequence length for vLLM/vllm_singularity.
        tensor_parallel_size: Number of GPUs for vLLM/vllm_singularity
            tensor parallelism.
        vllm_kwargs: Extra keyword arguments passed to the in-process vLLM
            ``LLM`` constructor (``vllm`` backend only — has no effect on
            ``vllm_singularity`` where vllm runs inside a container; use
            ``singularity_extra_vllm_args`` there). Stored as tuple-of-tuples
            to keep the frozen dataclass hashable (predict.py's backend
            cache uses ``BackendConfig`` as a dict key via
            ``dataclasses.replace``).
        config_overrides: Extra ``-c key=value`` overrides for the Codex
            CLI backend (ignored by other types). Stored as tuple-of-tuples
            for hashability, mirroring ``vllm_kwargs``.
        claude_code_kwargs: Extra keyword arguments forwarded to the
            ``ClaudeCodeBackend`` constructor (``claude_code`` backend only —
            ignored by other types). Useful keys include ``effort`` (low/
            medium/high/xhigh/max), ``bare`` (skip OAuth and use
            ``ANTHROPIC_API_KEY`` only), ``tools`` (comma-separated tool
            allowlist; defaults to empty so the backend has no tool access),
            ``permission_mode``, ``max_budget_usd``, ``fallback_model``, and
            ``extra_args`` (raw CLI flags). Stored as tuple-of-tuples for
            hashability; list-valued kwargs (e.g. ``extra_args``) are stored
            as inner tuples and reversed back to lists at construction time.
        singularity_sif: Path to the Singularity ``.sif`` image for the
            ``vllm_singularity`` backend. ``None`` falls back to the
            ``LLENVS_SIF`` env var (typically set by sourcing
            ``llenvs/bin/_cluster.sh``). Ignored by other backend types.
        singularity_binds: Extra ``--bind`` specs forwarded to
            ``singularity exec`` for the ``vllm_singularity`` backend.
            Empty tuple falls back to ``LLENVS_BINDS`` (space-separated).
            Stored as a tuple of strings for hashability. Ignored by other
            backend types.
        singularity_extra_vllm_args: Extra CLI flags passed to the in-
            container ``vllm serve`` (e.g. ``("--swap-space", "4")``) for
            the ``vllm_singularity`` backend. Stored as a tuple of strings
            for hashability. Ignored by other backend types.
        singularity_startup_timeout: Seconds to wait for ``vllm serve``'s
            ``/health`` endpoint to come up for the ``vllm_singularity``
            backend. Large VLMs with torch.compile warmup can need 1800+.
            Ignored by other backend types.
        singularity_cuda_visible_devices: Value for
            ``CUDA_VISIBLE_DEVICES`` inside the container for the
            ``vllm_singularity`` backend (e.g. ``"0,1"``) — pins
            ``vllm serve`` to a subset of the job's GPUs so the rest stay
            free for other workloads. ``None`` inherits whatever
            ``CUDA_VISIBLE_DEVICES`` the parent process sees. Ignored by
            other backend types.
        enable_thinking: Whether to enable model thinking tokens. Passed as a
            chat template kwarg at generation time for HuggingFace/vLLM backends.
        max_concurrency: Maximum concurrent requests for API backends
            (openrouter, litellm). Defaults to 64.
        rate_limit_wait: Seconds to wait before retrying after a 429 rate-limit
            error (openrouter, litellm). 0 disables (uses SDK default backoff).
        rate_limit_max_retries: Maximum number of rate-limit retries before
            giving up (openrouter, litellm). Only applies when
            rate_limit_wait > 0.
        connect_timeout: TCP connect timeout in seconds for API backends.
            ``None`` uses SDK defaults (5s). Increase for unreliable networks.
            For ``litellm`` this maps onto litellm's single per-request
            timeout (connect + read).
        api_max_retries: Maximum SDK-level retries for transient errors
            (timeouts, connection errors, 5xx). ``None`` uses SDK default (2).
            For ``litellm`` this maps onto ``num_retries`` (default off).
        singularity_sif: Path to the Singularity ``.sif`` for the
            ``vllm_singularity`` backend. Empty string means "fall back to
            ``$LLENVS_SIF``".
        singularity_binds: Extra ``--bind`` specs for the
            ``vllm_singularity`` backend. Empty tuple means "fall back to
            ``$LLENVS_BINDS``".
        singularity_extra_vllm_args: Extra CLI flags passed to
            ``vllm serve`` inside the container (e.g. ``("--swap-space",
            "4")``).
        singularity_startup_timeout: Seconds to wait for ``vllm serve`` to
            become healthy. Defaults to 900 (15 min) — large VLMs with
            torch.compile can take a while cold.
        singularity_cuda_visible_devices: If set, pin the container's
            ``vllm serve`` process to this subset of GPUs (e.g. ``"0,1"``).
            Lets other workloads use the remaining GPUs on the same node.
        quota_retry_policy: How to handle ``QuotaExhaustedError`` from the
            backend (e.g., Codex 5h window). ``"sleep_and_retry"`` transparently
            sleeps on a fixed schedule (5 min + 5 x 1 h) and retries before
            giving up. ``"abort"`` lets the error propagate on first occurrence
            so the pipeline can exit gracefully immediately.
        openrouter_provider: Request-level provider routing for OpenRouter
            backends. Forwarded on every chat-completion call via the OpenAI
            SDK's ``extra_body``. Only valid when ``type == "openrouter"``;
            the YAML parser raises if set on any other backend type.
        sampling: Per-call sampling parameters (temperature, top_p, etc.).
    """

    type: str
    model: str
    backend_url: str = "http://localhost:8000/v1"
    api_base: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    vllm_kwargs: tuple[tuple[str, Any], ...] = ()
    config_overrides: tuple[tuple[str, Any], ...] = ()
    claude_code_kwargs: tuple[tuple[str, Any], ...] = ()
    singularity_sif: str | None = None
    singularity_binds: tuple[str, ...] = ()
    singularity_extra_vllm_args: tuple[str, ...] = ()
    singularity_startup_timeout: float = 900.0
    singularity_cuda_visible_devices: str | None = None
    enable_thinking: bool = True
    max_concurrency: int = 64
    rate_limit_wait: float = 0.0
    rate_limit_max_retries: int = 2
    connect_timeout: float | None = None
    api_max_retries: int | None = None
    quota_retry_policy: str = "sleep_and_retry"
    openrouter_provider: OpenRouterProviderPreferences | None = None
    sampling: BackendSamplingConfig = field(default_factory=BackendSamplingConfig)

    def __post_init__(self) -> None:
        if self.quota_retry_policy not in ("sleep_and_retry", "abort"):
            raise ValueError(
                "BackendConfig.quota_retry_policy must be 'sleep_and_retry' "
                f"or 'abort', got {self.quota_retry_policy!r}"
            )


@dataclass(frozen=True)
class CollectionConfig:
    """Configuration for dataset collection (scripts/pipeline/collect_dataset.py).

    Attributes:
        harbor_state_capture: Harbor-only state capture mode. ``"replay"``
            keeps replay-based restore. ``"snapshot_exact"`` captures exact
            runtime checkpoints during live collection.
        ranking_actions: Target maximum candidate actions per ranking point
            (k). ``None`` disables ranking point collection entirely.
        ranking_sampling_mode: ``"llm"`` samples extra actions from the
            actor backend; ``"manual"`` uses a per-environment registered
            action sampler.
        ranking_backend_names: Optional backend list used only for LLM
            ranking sampling. When set, collection backends are torn down
            before ranking sampling and these backends are created just for
            ranking.
        ranking_sampling_strategy: Strategy for auxiliary LLM-sampled
            actions. ``"avoid_seen"`` asks each candidate slot not to
            repeat already accepted actions. ``"independent"`` keeps the
            actor-style prompt path.
        ranking_sampler_name: Name of the manual action sampler (required
            when ``ranking_sampling_mode`` is ``"manual"``).
        ranking_sampling: Sampling parameters for the extra LLM-sampled
            actions (only used when ``ranking_sampling_mode`` is ``"llm"``).
        restore_concurrency: Maximum concurrent restore-based ranking
            validations for non-pure environments. ``1`` preserves
            sequential behavior.
        discard_runtime_probe_risky_trajectories: When ``True``, Harbor
            trajectories flagged by runtime probing are discarded before
            evaluation-point extraction and ranking collection.
    """

    adapter: str | None = None
    env_name: str | None = None
    backend_names: tuple[str, ...] | None = None
    max_steps: int | None = None
    seed: int = 42
    make_kwargs: dict[str, Any] | None = None
    step_penalty: float | None = None
    turn_info: bool | None = None
    context_name: str | None = None
    contexts_dir: str = "shared/configs/environments/"
    system_prompt: str | None = None
    trajectories_path: str | None = None
    harbor_state_capture: str = "replay"
    ranking_actions: int | None = None
    ranking_sampling_mode: str = "llm"
    ranking_backend_names: tuple[str, ...] | None = None
    ranking_sampling_strategy: str = "avoid_seen"
    ranking_sampler_name: str | None = None
    ranking_sampling: BackendSamplingConfig | None = None
    restore_concurrency: int = 16
    discard_runtime_probe_risky_trajectories: bool = True
    max_history_turns: int | None = None
    runtime_probing: bool = False
    debug: bool = False
    # Scripted-policy collection (alternative to LLM/VLM backends).
    # When ``scripted_policy_name`` is set, collection uses
    # ``ScriptedBackend`` wrapping the named registered policy (with
    # epsilon-uniform-random mixing), and ``backend_names`` is ignored
    # for the actor.
    scripted_policy_name: str | None = None
    scripted_policy_epsilon: float = 0.0
    scripted_policy_seed: int | None = None


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for prediction (scripts/pipeline/predict.py)."""

    adapter: str | None = None
    gt_backend_name: str | None = None
    eval_backend_name: str | None = None
    max_history_turns: int | None = None
    context_name: str | None = None
    contexts_dir: str = "shared/configs/environments/"
    system_prompt: str | None = None
    make_kwargs: dict[str, Any] | None = None
    cli_backend_tmux_prompt_hardening: bool = False


@dataclass(frozen=True)
class MCRolloutPersistenceConfig:
    """Configuration for persistent MC rollout storage.

    Shard granularity is not configurable — it is derived per rollout group
    from ``batch_size // num_rollouts`` (the number of points that fill one
    lockstep parallel batch). This keeps durability aligned with the natural
    parallel unit: a shard flushes once every batch of rollouts completes.

    ``trust_sibling_hash`` is a recovery escape hatch: when an additive
    schema change to ``BackendConfig`` invalidates the rollout cache key
    (the generation hash) but the underlying rollouts are still
    semantically valid, this flag tells the store to adopt an orphan
    sibling directory under the same ``dataset_fingerprint`` parent
    instead of starting from scratch. Adoption succeeds only when exactly
    one sibling exists; multiple siblings raise. Off by default — only
    enable when you know the rollouts are reusable for the new spec.
    """

    dir: str | None = None
    resume: bool = True
    trust_sibling_hash: bool = False


_VALID_SAMPLING_STRATEGIES = {"first", "random", "uniform"}
_DISCARD_FIELDS = ("early_turns_to_discard", "late_turns_to_discard")


@dataclass(frozen=True)
class EvaluationPointsConfig:
    """Configuration for collecting evaluation points.

    Attributes:
        num_trajectories: Number of trajectories to collect points from.
        max_points_per_trajectory: Maximum points to sample per trajectory.
        task_indices: Specific task indices to use (None = use first num_trajectories).
        sampling_strategy: How to select points when max_points_per_trajectory
            limits the count. ``"first"`` takes the first N transitions,
            ``"random"`` samples N deterministically (seeded by trajectory index),
            ``"uniform"`` picks evenly spaced indices across the trajectory.
        shuffle_tasks: When True and no explicit task_indices are provided,
            shuffle all available task indices and take the first
            num_trajectories. Ensures representative sampling across task
            types in multi-type environments.
        early_turns_to_discard: Number of initial transitions to exclude from
            sampling. Applied per trajectory before the sampling strategy.
        late_turns_to_discard: Number of final transitions to exclude from
            sampling. Applied per trajectory before the sampling strategy.
    """

    num_trajectories: int = 10
    max_points_per_trajectory: int | None = None
    task_indices: tuple[int, ...] | None = None
    sampling_strategy: str = "random"
    shuffle_tasks: bool = True
    max_trajectories_to_keep: int | None = None
    early_turns_to_discard: int = 1
    late_turns_to_discard: int = 1


@dataclass(frozen=True)
class ReplayValidationConfig:
    """Configuration for replay validation during Harbor data collection.

    Attributes:
        enabled: Whether to validate replay consistency.
        target_points: Desired number of validated points.
        max_discards: Maximum allowed discarded points before erroring.
    """

    enabled: bool = False
    target_points: int = 300
    max_discards: int = 50


@dataclass(frozen=True)
class ResumeRankingSubsetConfig:
    """Subset selection for ranking-only resume runs.

    Attributes:
        count: Number of stored evaluation points to sample.
        seed: Optional RNG seed. ``None`` falls back to ``collection.seed``.
    """

    count: int
    seed: int | None = None


_VALID_ESTIMATION_METHODS = {"mc", "mcts"}

_VALID_BACKEND_TYPES = {
    "openai",
    "huggingface",
    "vllm",
    "vllm_singularity",
    "anthropic",
    "openrouter",
    "litellm",
    "huggingface_vle",
    "vip",
    "liv",
    "codex",
    "claude_code",
}

_VALID_BACKEND_SAMPLING_KEYS = {f.name for f in fields(BackendSamplingConfig)}

_VALID_OPENROUTER_DATA_COLLECTION = {"allow", "deny"}
_VALID_OPENROUTER_SORT = {"price", "throughput", "latency"}
_VALID_OPENROUTER_REASONING_EFFORT = {
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
}


def _parse_estimation_config(data: dict) -> EstimationConfig:
    """Parse a single estimation config dict into an EstimationConfig."""
    d = dict(data)
    if "rollout_temperature" in d or "rollout_max_tokens" in d:
        raise ValueError(
            "Use backends.<name>.sampling for rollout sampling settings; "
            "rollout_temperature/rollout_max_tokens are no longer supported."
        )
    _check_unknown_keys(d, EstimationConfig, "estimations[]")
    if "assumption" in d:
        d["assumption"] = _parse_enum(d["assumption"], SignalType, "assumption")
    if "method" in d:
        method = str(d["method"]).lower()
        if method not in _VALID_ESTIMATION_METHODS:
            raise ValueError(
                f"Invalid method: {d['method']!r}. "
                f"Valid values: {sorted(_VALID_ESTIMATION_METHODS)}"
            )
        d["method"] = method
    if "aggregation" in d:
        d["aggregation"] = _parse_enum(
            d["aggregation"], AggregationMethod, "aggregation"
        )
    if d.get("policy_name") is not None:
        policy_name = str(d["policy_name"]).strip()
        if not policy_name:
            raise ValueError("estimations[].policy_name must not be empty")
        d["policy_name"] = policy_name
    if d.get("policy_name") is not None and d.get("backend_name") is not None:
        raise ValueError("estimations[] must not set both policy_name and backend_name")
    return EstimationConfig(**d)


_VALID_EVAL_METHOD_TYPES = {
    "llm_direct",
    "llm_direct_gvl",
    "llm_verifier",
    "llm_codegen",
    "llm_eureka",
    "llm_ranking",
    "sdpo_ranking",
    "sdpo_short_expert_continuation",
    "delta_belief_ranking",
    "baseline_random",
    "vlm_rm",
    "vlm_sor",
    "vip",
    "liv_img",
    "liv_txt",
}

_VLE_METHOD_TYPES = {"vlm_rm", "vlm_sor"}
_VALID_VLE_IMAGE_SOURCES = {"state", "next_state"}
_VALID_VLE_MULTI_IMAGE_STRATEGIES = {"last", "first", "mean"}

_VIP_METHOD_TYPES = {"vip"}
_VALID_VIP_GOAL_SOURCES = {"trajectory_end", "env_image"}
_VALID_VIP_IMAGE_STRATEGIES = {"last", "first"}
_VIP_ONLY_FIELDS = (
    "vip_goal_source",
    "vip_env_goal_image_path",
    "vip_multi_image_strategy",
)

_LIV_METHOD_TYPES = {"liv_img", "liv_txt"}
_LIV_IMG_METHOD_TYPE = "liv_img"
_LIV_TXT_METHOD_TYPE = "liv_txt"
_VALID_LIV_SIMILARITIES = {"cosine", "l2"}
_VALID_LIV_GOAL_SOURCES = {"trajectory_end", "env_image"}
_VALID_LIV_IMAGE_STRATEGIES = {"last", "first"}
_LIV_ONLY_FIELDS = (
    "liv_similarity",
    "liv_goal_source",
    "liv_env_goal_image_path",
    "liv_goal_text",
    "liv_goal_per_point",
    "liv_multi_image_strategy",
)
_VLE_ONLY_FIELDS = (
    "vle_baseline_prompt",
    "vle_alpha",
    "vle_negative_goals",
    "vle_temperature",
    "vle_beta",
    "vle_image_source",
    "vle_multi_image_strategy",
    "vle_goal_text",
    "vle_goal_per_point",
)

_VALID_RANKING_SAMPLING_MODES = {"llm", "manual"}
_VALID_RANKING_SAMPLING_STRATEGIES = {"avoid_seen", "independent"}
_VALID_HARBOR_STATE_CAPTURE_MODES = {"replay", "snapshot_exact"}


@dataclass(frozen=True)
class EvalMethodConfig:
    """Configuration for an evaluation method instance.

        Allows running multiple instances of the same method type with
        different settings (e.g., different signal types, policy assumptions).

        Attributes:
            name: Human-readable name for this method instance (used in
                output filenames like ``eval_{name}.json``).
            type: Method implementation: ``"llm_direct"``, ``"llm_direct_gvl"``,
                ``"llm_verifier"``, ``"llm_codegen"``, ``"llm_eureka"``,
                ``"llm_ranking"``, ``"sdpo_ranking"``,
                ``"delta_belief_ranking"`` (ranking methods require
                ``signal_type=q_value``), ``"baseline_random"``, ``"vlm_rm"``,
                ``"vlm_sor"``, ``"vip"``, ``"liv_img"``, or ``"liv_txt"``.
            signal_type: Type of dense signal this method evaluates. Each
                method declares its own signal type, enabling mixed-signal-type
                prediction runs.
            prompt_preset: Name of the prompt preset to use (e.g., ``"v0"``,
                ``"v1"``). Required for LLM-based methods — every experiment
                must explicitly declare which prompts it uses. Optional for
                ``baseline_random``; defaults to ``"v0"`` for metadata uniformity
                when omitted.
            policy_assumption: Policy assumption for prompts.
            backend_name: Name of backend entry to use.
            num_samples: Number of independent code samples/candidates to
                generate. Supported by ``llm_codegen`` and ``llm_eureka``.
            search_iterations: Number of iterative search rounds for
                ``llm_eureka``.
            judge_num_points: Number of sampled datapoints shown to the judge
                prompt for ``llm_eureka``.
            num_verifications: Number of repeated prediction passes to average.
                Supported by ``llm_direct``, ``llm_direct_gvl``, and
                ``llm_verifier``.
            prompt_batch_size: Number of evaluation points to pack into a
                single prompt. Used by ``llm_direct`` and ``llm_verifier``.
                ``llm_direct_gvl`` does not accept this field because it uses
                a dedicated one-target-per-prompt trajectory-context format.
            prompt_batch_mode: Prompt protocol for batched ``llm_direct``
                prompts: ``"packed"`` or ``"sequential"``.
            prompt_grouping: Prompt grouping mode for ``llm_verifier``.
                ``llm_direct_gvl`` accepts only the fixed value
                ``"trajectory"`` when explicitly provided.
            criteria: Built-in verifier criterion IDs for ``llm_verifier``.
            disable_prompt_truncation: If True, skip budget-aware prompt
                truncation for this method even when ``min_*_chars`` defaults
                are inherited from the environment context. Required for
                ``llm_direct`` with ``prompt_batch_mode="sequential"``, which
                does not support token-budget truncation.
            max_feedback_chars: Hard character cap for the feedback/next-state
                text shown to ``sdpo_ranking`` teachers. Currently used for
                TerminalBench shell output with head+tail truncation.
            seed: PRNG seed for ``baseline_random``. Two runs with the same
                seed produce identical prediction sequences. Rejected for any
                other method type.
    """

    name: str
    type: str
    signal_type: SignalType
    prompt_preset: str | None = None
    policy_assumption: PolicyAssumption = PolicyAssumption.OPTIMAL
    backend_name: str | None = None
    num_samples: int = 1
    search_iterations: int = 3
    judge_num_points: int = 8
    num_verifications: int = 1
    prompt_batch_size: int = 1
    prompt_batch_mode: str = "packed"
    prompt_grouping: str | None = None
    criteria: tuple[str, ...] | None = None
    disclose_step_penalty: bool = False
    disclose_discount_factor: bool = False
    include_next_state: bool = True
    include_current_thoughts: bool = False
    include_history_thoughts: bool = False
    include_state_text_when_images: bool | None = None
    include_images: bool | None = None
    # Truncation overrides — None means use environment context default
    max_history_turns: int | None = None
    min_action_chars: int | None = None
    min_observation_chars: int | None = None
    min_current_observation_chars: int | None = None
    min_next_observation_chars: int | None = None
    disable_prompt_truncation: bool = False
    max_feedback_chars: int | None = None
    expert_rollout_store: str | None = None
    max_expert_continuation_steps: int = 5
    # VLE fields (type in {"vlm_rm", "vlm_sor"}). See qval.methods.vle.
    vle_baseline_prompt: str | None = None
    vle_alpha: float | None = None
    vle_negative_goals: tuple[str, ...] | None = None
    vle_temperature: float | None = None
    vle_beta: float | None = None
    vle_image_source: str | None = None
    vle_multi_image_strategy: str | None = None
    vle_goal_text: str | None = None
    vle_goal_per_point: bool = False
    # VIP fields (type == "vip"). See qval.methods.vip.
    vip_goal_source: str | None = None
    vip_env_goal_image_path: str | None = None
    vip_multi_image_strategy: str | None = None
    # LIV fields (type in {"liv_img", "liv_txt"}). See qval.methods.liv.
    liv_similarity: str | None = None
    liv_goal_source: str | None = None
    liv_env_goal_image_path: str | None = None
    liv_goal_text: str | None = None
    liv_goal_per_point: bool = False
    liv_multi_image_strategy: str | None = None
    # Baseline-specific: PRNG seed for ``baseline_random``. Ignored by all
    # other method types.
    seed: int | None = None


def _parse_eval_method_config(data: dict) -> EvalMethodConfig:
    """Parse a single eval method config dict into an EvalMethodConfig."""
    d = dict(data)
    _check_unknown_keys(d, EvalMethodConfig, "eval_methods[]")
    if "type" in d:
        method_type = str(d["type"]).lower()
        if method_type not in _VALID_EVAL_METHOD_TYPES:
            raise ValueError(
                f"Invalid eval method type: {d['type']!r}. "
                f"Valid values: {sorted(_VALID_EVAL_METHOD_TYPES)}"
            )
        d["type"] = method_type
    else:
        raise ValueError("eval_methods[] entries must include 'type'")
    if "name" not in d:
        raise ValueError("eval_methods[] entries must include 'name'")
    is_baseline_random = d["type"] == "baseline_random"
    prompt_optional_method = (
        is_baseline_random
        or d["type"] in _VLE_METHOD_TYPES
        or d["type"] in _VIP_METHOD_TYPES
        or d["type"] in _LIV_METHOD_TYPES
    )
    if "prompt_preset" not in d:
        if is_baseline_random:
            d["prompt_preset"] = "v0"
        elif prompt_optional_method:
            d["prompt_preset"] = None
        else:
            raise ValueError("eval_methods[] entries must include 'prompt_preset'")
    if "signal_type" not in d:
        raise ValueError("eval_methods[] entries must include 'signal_type'")
    d["signal_type"] = _parse_enum(d["signal_type"], SignalType, "signal_type")
    # Validate: ranking methods require q_value
    if (
        d["type"] in {
            "llm_ranking",
            "sdpo_ranking",
            "sdpo_short_expert_continuation",
            "delta_belief_ranking",
        }
        and d["signal_type"] != SignalType.Q_VALUE
    ):
        raise ValueError(
            f"{d['type']} requires signal_type=q_value, "
            f"got {d['signal_type'].name.lower()!r}"
        )
    # Validate the preset name is registered (skipped for type=='vle' when absent)
    if d.get("prompt_preset") is not None:
        from qval.prompt_presets import get_preset

        try:
            get_preset(d["prompt_preset"])
        except KeyError as e:
            raise ValueError(str(e)) from None
    if "seed" in d and d["seed"] is not None:
        if not is_baseline_random:
            raise ValueError(
                f"eval_methods[].seed is only supported for type=baseline_random, "
                f"got type={d['type']!r}"
            )
        d["seed"] = int(d["seed"])
    if "policy_assumption" in d:
        d["policy_assumption"] = _parse_enum(
            d["policy_assumption"], PolicyAssumption, "policy_assumption"
        )
    if "num_samples" in d:
        d["num_samples"] = int(d["num_samples"])
        if d["num_samples"] < 1:
            raise ValueError("eval_methods[].num_samples must be >= 1")
        if d["num_samples"] > 1 and d["type"] not in ("llm_codegen", "llm_eureka"):
            raise ValueError(
                f"num_samples > 1 is only supported for llm_codegen or llm_eureka, "
                f"got type={d['type']!r}"
            )
    if "search_iterations" in d:
        d["search_iterations"] = int(d["search_iterations"])
        if d["search_iterations"] < 1:
            raise ValueError("eval_methods[].search_iterations must be >= 1")
    if "judge_num_points" in d:
        d["judge_num_points"] = int(d["judge_num_points"])
        if d["judge_num_points"] < 1:
            raise ValueError("eval_methods[].judge_num_points must be >= 1")
    if "num_verifications" in d:
        d["num_verifications"] = int(d["num_verifications"])
        if d["num_verifications"] < 1:
            raise ValueError("eval_methods[].num_verifications must be >= 1")
    if "prompt_batch_size" in d:
        d["prompt_batch_size"] = int(d["prompt_batch_size"])
        if d["prompt_batch_size"] < 1:
            raise ValueError("eval_methods[].prompt_batch_size must be >= 1")
    if "prompt_batch_mode" in d and d["prompt_batch_mode"] is not None:
        d["prompt_batch_mode"] = str(d["prompt_batch_mode"]).lower()
    if "prompt_grouping" in d and d["prompt_grouping"] is not None:
        d["prompt_grouping"] = str(d["prompt_grouping"]).lower()
    if "criteria" in d and d["criteria"] is not None:
        if not isinstance(d["criteria"], (list, tuple)):
            raise ValueError("eval_methods[].criteria must be a list of criterion IDs")
        d["criteria"] = tuple(str(item) for item in d["criteria"])
        if len(d["criteria"]) == 0:
            raise ValueError("eval_methods[].criteria must not be empty")
    if "disclose_step_penalty" in d:
        d["disclose_step_penalty"] = bool(d["disclose_step_penalty"])
    if "disclose_discount_factor" in d:
        d["disclose_discount_factor"] = bool(d["disclose_discount_factor"])
    if "include_next_state" in d:
        d["include_next_state"] = bool(d["include_next_state"])
    if "include_current_thoughts" in d:
        d["include_current_thoughts"] = bool(d["include_current_thoughts"])
    if "include_history_thoughts" in d:
        d["include_history_thoughts"] = bool(d["include_history_thoughts"])
    if "disable_prompt_truncation" in d:
        d["disable_prompt_truncation"] = bool(d["disable_prompt_truncation"])
    if "max_feedback_chars" in d and d["max_feedback_chars"] is not None:
        d["max_feedback_chars"] = _parse_positive_int(
            d["max_feedback_chars"],
            "eval_methods[].max_feedback_chars",
        )
        if d["type"] != "sdpo_ranking":
            raise ValueError(
                "max_feedback_chars is only supported for sdpo_ranking"
            )
    if "max_history_turns" in d and d["max_history_turns"] is not None:
        d["max_history_turns"] = int(d["max_history_turns"])
    if "expert_rollout_store" in d and d["expert_rollout_store"] is not None:
        d["expert_rollout_store"] = str(d["expert_rollout_store"])
    if "max_expert_continuation_steps" in d:
        d["max_expert_continuation_steps"] = int(
            d["max_expert_continuation_steps"]
        )
        if d["max_expert_continuation_steps"] < 0:
            raise ValueError(
                "eval_methods[].max_expert_continuation_steps must be >= 0"
            )
    if d["type"] == "sdpo_short_expert_continuation":
        if not d.get("expert_rollout_store"):
            raise ValueError(
                "sdpo_short_expert_continuation requires expert_rollout_store"
            )
    else:
        if d.get("expert_rollout_store") is not None:
            raise ValueError(
                "expert_rollout_store is only supported for "
                "sdpo_short_expert_continuation"
            )
        if "max_expert_continuation_steps" in d:
            raise ValueError(
                "max_expert_continuation_steps is only supported for "
                "sdpo_short_expert_continuation"
            )

    if d["type"] != "llm_eureka":
        if "search_iterations" in d:
            raise ValueError(
                f"search_iterations is only supported for llm_eureka, got type={d['type']!r}"
            )
        if "judge_num_points" in d:
            raise ValueError(
                f"judge_num_points is only supported for llm_eureka, got type={d['type']!r}"
            )

    if d["type"] == "llm_direct_gvl":
        if "max_history_turns" not in d:
            d["max_history_turns"] = 0
        if d["max_history_turns"] not in (None, 0):
            raise ValueError("llm_direct_gvl requires max_history_turns=0")
        if d.get("include_current_thoughts", False):
            raise ValueError("llm_direct_gvl does not support include_current_thoughts")
        if d.get("include_history_thoughts", False):
            raise ValueError("llm_direct_gvl does not support include_history_thoughts")
        if "prompt_batch_size" in d:
            raise ValueError(
                "llm_direct_gvl does not accept prompt_batch_size; "
                "it always packs the full trajectory into one prompt"
            )
        if "prompt_batch_mode" in d and d["prompt_batch_mode"] != "packed":
            raise ValueError("llm_direct_gvl requires prompt_batch_mode='packed'")
        if "prompt_grouping" in d and d["prompt_grouping"] not in (None, "trajectory"):
            raise ValueError("llm_direct_gvl requires prompt_grouping='trajectory'")

    if d["type"] == "llm_direct":
        if "prompt_batch_mode" not in d or d["prompt_batch_mode"] is None:
            d["prompt_batch_mode"] = "packed"
        if d["prompt_batch_mode"] not in {"packed", "sequential"}:
            raise ValueError(
                "llm_direct prompt_batch_mode must be one of {'packed', 'sequential'}"
            )
    else:
        if d.get("prompt_batch_mode") is not None:
            raise ValueError("prompt_batch_mode is only supported for llm_direct")

    if d["type"] == "llm_verifier":
        from qval.verifier_criteria import list_verifier_criteria

        if d["signal_type"] not in (SignalType.STATE_VALUE, SignalType.Q_VALUE):
            raise ValueError(
                "llm_verifier supports only signal_type in {'state_value', 'q_value'}"
            )
        if "prompt_grouping" not in d or d["prompt_grouping"] is None:
            d["prompt_grouping"] = "random"
        if d["prompt_grouping"] not in {"contiguous", "random", "trajectory"}:
            raise ValueError(
                "llm_verifier prompt_grouping must be one of "
                "{'contiguous', 'random', 'trajectory'}"
            )
        if d.get("criteria") is not None:
            available = set(list_verifier_criteria())
            unknown = [
                criterion for criterion in d["criteria"] if criterion not in available
            ]
            if unknown:
                raise ValueError(
                    f"Unknown verifier criterion {unknown[0]!r}. "
                    f"Available: {sorted(available)}"
                )
    else:
        if d.get("criteria") is not None:
            raise ValueError(
                f"criteria is only supported for llm_verifier, got type={d['type']!r}"
            )
        if d.get("prompt_grouping") is not None:
            raise ValueError("prompt_grouping is only configurable for llm_verifier")

    if d["type"] not in {"llm_direct", "llm_direct_gvl", "llm_verifier"}:
        if d.get("num_verifications", 1) != 1:
            raise ValueError(
                f"num_verifications is only supported for llm_direct, "
                f"llm_direct_gvl, and llm_verifier, got type={d['type']!r}"
            )

    _validate_vle_method_fields(d)
    _validate_vip_method_fields(d)
    _validate_liv_method_fields(d)
    return EvalMethodConfig(**d)

def _validate_vle_method_fields(d: dict) -> None:
    """Validate VLE-specific fields. Mutates ``d`` in place for coercions."""
    method_type = d["type"]
    is_vle = method_type in _VLE_METHOD_TYPES

    def _is_meaningful(key: str, val: object) -> bool:
        if val is None:
            return False
        if key == "vle_goal_per_point":
            return bool(val)
        return True

    vle_present = {k: d[k] for k in _VLE_ONLY_FIELDS if _is_meaningful(k, d.get(k))}

    if not is_vle:
        if vle_present:
            raise ValueError(
                f"VLE-specific fields {sorted(vle_present)} are only valid "
                f"when type is one of {sorted(_VLE_METHOD_TYPES)}, "
                f"got type={method_type!r}"
            )
        return

    if d["signal_type"] not in (SignalType.STATE_VALUE, SignalType.Q_VALUE):
        raise ValueError(
            f"{method_type} supports only signal_type in "
            "{'state_value', 'q_value'}"
        )
    if d.get("backend_name") is None:
        raise ValueError(
            f"{method_type} requires backend_name (a huggingface_vle backend entry)"
        )

    if "vle_negative_goals" in d and d["vle_negative_goals"] is not None:
        goals = d["vle_negative_goals"]
        if isinstance(goals, str) or not isinstance(goals, (list, tuple)):
            raise ValueError(
                "eval_methods[].vle_negative_goals must be a list of strings"
            )
        goals = tuple(str(g) for g in goals)
        if not goals:
            raise ValueError("eval_methods[].vle_negative_goals must be non-empty")
        d["vle_negative_goals"] = goals
    if "vle_alpha" in d and d["vle_alpha"] is not None:
        d["vle_alpha"] = float(d["vle_alpha"])
    if "vle_temperature" in d and d["vle_temperature"] is not None:
        d["vle_temperature"] = float(d["vle_temperature"])
    if "vle_beta" in d and d["vle_beta"] is not None:
        d["vle_beta"] = float(d["vle_beta"])
    if "vle_image_source" in d and d["vle_image_source"] is not None:
        src = str(d["vle_image_source"]).lower()
        if src not in _VALID_VLE_IMAGE_SOURCES:
            raise ValueError(
                f"Invalid vle_image_source: {d['vle_image_source']!r}. "
                f"Valid: {sorted(_VALID_VLE_IMAGE_SOURCES)}"
            )
        d["vle_image_source"] = src
    if "vle_multi_image_strategy" in d and d["vle_multi_image_strategy"] is not None:
        strat = str(d["vle_multi_image_strategy"]).lower()
        if strat not in _VALID_VLE_MULTI_IMAGE_STRATEGIES:
            raise ValueError(
                f"Invalid vle_multi_image_strategy: "
                f"{d['vle_multi_image_strategy']!r}. "
                f"Valid: {sorted(_VALID_VLE_MULTI_IMAGE_STRATEGIES)}"
            )
        d["vle_multi_image_strategy"] = strat
    if "vle_goal_text" in d and d["vle_goal_text"] is not None:
        text = str(d["vle_goal_text"]).strip()
        if not text:
            raise ValueError("eval_methods[].vle_goal_text must not be empty if set")
        d["vle_goal_text"] = text
    if "vle_goal_per_point" in d and d["vle_goal_per_point"] is not None:
        d["vle_goal_per_point"] = bool(d["vle_goal_per_point"])
    if d.get("vle_goal_text") is not None and d.get("vle_goal_per_point"):
        raise ValueError(
            "eval_methods[]: vle_goal_text and vle_goal_per_point are "
            "mutually exclusive"
        )

    if method_type == "vlm_rm":
        forbidden = [
            f for f in ("vle_negative_goals", "vle_temperature", "vle_beta")
            if d.get(f) is not None
        ]
        if forbidden:
            raise ValueError(
                f"vlm_rm does not accept fields {forbidden}; these are for vlm_sor"
            )
        has_baseline = d.get("vle_baseline_prompt") is not None
        has_alpha = d.get("vle_alpha") is not None
        if has_baseline ^ has_alpha:
            raise ValueError(
                "vlm_rm goal-baseline regularization requires both "
                "vle_baseline_prompt and vle_alpha; leave both unset for raw cosine"
            )
    else:
        forbidden = [
            f for f in ("vle_baseline_prompt", "vle_alpha")
            if d.get(f) is not None
        ]
        if forbidden:
            raise ValueError(
                f"vlm_sor does not accept fields {forbidden}; these are for vlm_rm"
            )
        if d.get("vle_negative_goals") is None:
            raise ValueError("vlm_sor requires vle_negative_goals")
        if d.get("vle_temperature") is None:
            raise ValueError("vlm_sor requires vle_temperature")


def _validate_vip_method_fields(d: dict) -> None:
    """Validate VIP-specific fields. Mutates ``d`` in place for coercions."""
    method_type = d["type"]
    is_vip = method_type in _VIP_METHOD_TYPES
    vip_present = {k: d[k] for k in _VIP_ONLY_FIELDS if d.get(k) is not None}
    if not is_vip:
        if vip_present:
            raise ValueError(
                f"VIP-specific fields {sorted(vip_present)} are only valid "
                f"when type == 'vip', got type={method_type!r}"
            )
        return
    if d["signal_type"] not in (
        SignalType.STATE_VALUE,
        SignalType.Q_VALUE,
        SignalType.SHAPED_REWARD,
    ):
        raise ValueError(
            "vip supports only signal_type in "
            "{'state_value', 'q_value', 'shaped_reward'}"
        )
    if d.get("backend_name") is None:
        raise ValueError("vip requires backend_name (a vip backend entry)")
    goal_source = d.get("vip_goal_source")
    if goal_source is None:
        raise ValueError("vip requires vip_goal_source")
    goal_source = str(goal_source).lower()
    if goal_source not in _VALID_VIP_GOAL_SOURCES:
        raise ValueError(
            f"Invalid vip_goal_source: {d['vip_goal_source']!r}. "
            f"Valid: {sorted(_VALID_VIP_GOAL_SOURCES)}"
        )
    d["vip_goal_source"] = goal_source
    env_img_path = d.get("vip_env_goal_image_path")
    if goal_source == "env_image":
        if not env_img_path:
            raise ValueError(
                "vip_goal_source='env_image' requires vip_env_goal_image_path"
            )
        if not Path(str(env_img_path)).exists():
            raise ValueError(f"vip_env_goal_image_path does not exist: {env_img_path}")
    elif env_img_path is not None:
        raise ValueError(
            "vip_env_goal_image_path is only valid with vip_goal_source='env_image'"
        )
    if "vip_multi_image_strategy" in d and d["vip_multi_image_strategy"] is not None:
        strat = str(d["vip_multi_image_strategy"]).lower()
        if strat not in _VALID_VIP_IMAGE_STRATEGIES:
            raise ValueError(
                f"Invalid vip_multi_image_strategy: "
                f"{d['vip_multi_image_strategy']!r}. "
                f"Valid: {sorted(_VALID_VIP_IMAGE_STRATEGIES)}"
            )
        d["vip_multi_image_strategy"] = strat


def _validate_liv_method_fields(d: dict) -> None:
    """Validate LIV-specific fields. Mutates ``d`` in place for coercions."""
    method_type = d["type"]
    is_liv = method_type in _LIV_METHOD_TYPES
    liv_present = {k: d[k] for k in _LIV_ONLY_FIELDS if _liv_meaningful(k, d.get(k))}
    if not is_liv:
        if liv_present:
            raise ValueError(
                f"LIV-specific fields {sorted(liv_present)} are only valid "
                f"when type in {sorted(_LIV_METHOD_TYPES)}, got type={method_type!r}"
            )
        return
    if d["signal_type"] not in (
        SignalType.STATE_VALUE,
        SignalType.Q_VALUE,
        SignalType.SHAPED_REWARD,
    ):
        raise ValueError(
            f"{method_type} supports only signal_type in "
            "{'state_value', 'q_value', 'shaped_reward'}"
        )
    if d.get("backend_name") is None:
        raise ValueError(f"{method_type} requires backend_name (a liv backend entry)")

    if "liv_similarity" in d and d["liv_similarity"] is not None:
        sim = str(d["liv_similarity"]).lower()
        if sim not in _VALID_LIV_SIMILARITIES:
            raise ValueError(
                f"Invalid liv_similarity: {d['liv_similarity']!r}. "
                f"Valid: {sorted(_VALID_LIV_SIMILARITIES)}"
            )
        d["liv_similarity"] = sim
    if "liv_multi_image_strategy" in d and d["liv_multi_image_strategy"] is not None:
        strat = str(d["liv_multi_image_strategy"]).lower()
        if strat not in _VALID_LIV_IMAGE_STRATEGIES:
            raise ValueError(
                f"Invalid liv_multi_image_strategy: "
                f"{d['liv_multi_image_strategy']!r}. "
                f"Valid: {sorted(_VALID_LIV_IMAGE_STRATEGIES)}"
            )
        d["liv_multi_image_strategy"] = strat
    if "liv_goal_text" in d and d["liv_goal_text"] is not None:
        text = str(d["liv_goal_text"]).strip()
        if not text:
            raise ValueError("liv_goal_text must be non-empty when set")
        d["liv_goal_text"] = text
    if "liv_goal_per_point" in d and d["liv_goal_per_point"] is not None:
        d["liv_goal_per_point"] = bool(d["liv_goal_per_point"])

    if method_type == _LIV_IMG_METHOD_TYPE:
        forbidden = [
            f for f in ("liv_goal_text", "liv_goal_per_point") if d.get(f)
        ]
        if forbidden:
            raise ValueError(
                f"liv_img does not accept fields {forbidden}; these are for liv_txt"
            )
        goal_source = d.get("liv_goal_source")
        if goal_source is None:
            raise ValueError("liv_img requires liv_goal_source")
        goal_source = str(goal_source).lower()
        if goal_source not in _VALID_LIV_GOAL_SOURCES:
            raise ValueError(
                f"Invalid liv_goal_source: {d['liv_goal_source']!r}. "
                f"Valid: {sorted(_VALID_LIV_GOAL_SOURCES)}"
            )
        d["liv_goal_source"] = goal_source
        env_img = d.get("liv_env_goal_image_path")
        if goal_source == "env_image":
            if not env_img:
                raise ValueError(
                    "liv_goal_source='env_image' requires liv_env_goal_image_path"
                )
            if not Path(str(env_img)).exists():
                raise ValueError(
                    f"liv_env_goal_image_path does not exist: {env_img}"
                )
        elif env_img is not None:
            raise ValueError(
                "liv_env_goal_image_path is only valid with liv_goal_source='env_image'"
            )
    else:
        forbidden = [
            f for f in ("liv_goal_source", "liv_env_goal_image_path")
            if d.get(f) is not None
        ]
        if forbidden:
            raise ValueError(
                f"liv_txt does not accept fields {forbidden}; these are for liv_img"
            )
        if d.get("liv_goal_text") and d.get("liv_goal_per_point"):
            raise ValueError(
                "liv_txt: liv_goal_text and liv_goal_per_point are mutually exclusive"
            )


def _liv_meaningful(key: str, val: object) -> bool:
    if val is None:
        return False
    if key == "liv_goal_per_point":
        return bool(val)
    return True


def _parse_backend_sampling_config(
    data: dict,
    context: str,
    *,
    backend_type: str | None = None,
) -> BackendSamplingConfig:
    d = dict(data)
    unknown = set(d) - _VALID_BACKEND_SAMPLING_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in {context}.sampling: {', '.join(sorted(unknown))}. "
            f"Valid keys: {sorted(_VALID_BACKEND_SAMPLING_KEYS)}"
        )
    if "top_k" in d and d["top_k"] is not None:
        d["top_k"] = int(d["top_k"])
    if "max_tokens" in d and d["max_tokens"] is not None:
        d["max_tokens"] = int(d["max_tokens"])
    if "reasoning_effort" in d and d["reasoning_effort"] is not None:
        d["reasoning_effort"] = str(d["reasoning_effort"]).lower()
        if d["reasoning_effort"] not in _VALID_OPENROUTER_REASONING_EFFORT:
            raise ValueError(
                f"Invalid {context}.sampling.reasoning_effort: "
                f"{d['reasoning_effort']!r}. Valid values: "
                f"{sorted(_VALID_OPENROUTER_REASONING_EFFORT)}"
            )
    if (
        backend_type is not None
        and backend_type not in ("openrouter", "litellm")
        and (
            d.get("reasoning_effort") is not None
            or d.get("reasoning_exclude") is not None
        )
    ):
        raise ValueError(
            f"{context}.sampling.reasoning_effort/reasoning_exclude are only "
            f"valid for openrouter or litellm backends (got {backend_type!r})"
        )
    if backend_type == "litellm" and d.get("reasoning_exclude") is not None:
        raise ValueError(
            f"{context}.sampling.reasoning_exclude is only valid for "
            "openrouter backends; litellm has no equivalent of OpenRouter's "
            "reasoning.exclude flag"
        )
    if (
        backend_type in ("openrouter", "litellm")
        and d.get("thinking_budget") is not None
        and d.get("reasoning_effort") is not None
    ):
        raise ValueError(
            f"{context}.sampling cannot set both thinking_budget and "
            f"reasoning_effort for an {backend_type} backend; both map onto "
            "the same provider-side reasoning control (a token budget or an "
            "effort level, not both)"
        )
    return BackendSamplingConfig(**d)


def _parse_openrouter_provider_preferences(
    data: dict, context: str
) -> OpenRouterProviderPreferences:
    d = dict(data)
    _check_unknown_keys(d, OpenRouterProviderPreferences, context)

    for list_key in ("order", "only", "ignore", "quantizations"):
        if list_key in d and d[list_key] is not None:
            raw = d[list_key]
            if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
                raise ValueError(
                    f"{context}.{list_key} must be a list, got {type(raw).__name__}"
                )
            d[list_key] = tuple(str(x) for x in raw)

    if d.get("data_collection") is not None:
        val = str(d["data_collection"]).lower()
        if val not in _VALID_OPENROUTER_DATA_COLLECTION:
            raise ValueError(
                f"{context}.data_collection must be one of "
                f"{sorted(_VALID_OPENROUTER_DATA_COLLECTION)}, got "
                f"{d['data_collection']!r}"
            )
        d["data_collection"] = val

    if d.get("sort") is not None:
        val = str(d["sort"]).lower()
        if val not in _VALID_OPENROUTER_SORT:
            raise ValueError(
                f"{context}.sort must be one of "
                f"{sorted(_VALID_OPENROUTER_SORT)}, got {d['sort']!r}"
            )
        d["sort"] = val

    return OpenRouterProviderPreferences(**d)


def _parse_backend_config(name: str, data: dict) -> BackendConfig:
    d = dict(data)
    _check_unknown_keys(d, BackendConfig, f"backends.{name}")
    if "type" not in d:
        raise ValueError(f"backends.{name} must include 'type'")
    if "model" not in d:
        raise ValueError(f"backends.{name} must include 'model'")
    backend_type = str(d["type"]).lower()
    if backend_type not in _VALID_BACKEND_TYPES:
        raise ValueError(
            f"Invalid backend type: {d['type']!r}. "
            f"Valid values: {sorted(_VALID_BACKEND_TYPES)}"
        )
    d["type"] = backend_type
    if "vllm_kwargs" in d:
        raw_vllm = d["vllm_kwargs"]
        if isinstance(raw_vllm, dict):
            d["vllm_kwargs"] = tuple(sorted(raw_vllm.items()))
        elif raw_vllm is None:
            d["vllm_kwargs"] = ()
    for k in ("singularity_binds", "singularity_extra_vllm_args"):
        if k in d:
            raw = d[k]
            if raw is None:
                d[k] = ()
            elif isinstance(raw, (list, tuple)):
                d[k] = tuple(str(x) for x in raw)
            else:
                raise ValueError(
                    f"backends.{name}.{k} must be a list of strings"
                )
    if "config_overrides" in d:
        raw_overrides = d["config_overrides"]
        if isinstance(raw_overrides, dict):
            d["config_overrides"] = tuple(sorted(raw_overrides.items()))
        elif raw_overrides is None:
            d["config_overrides"] = ()
    if "claude_code_kwargs" in d:
        raw_cc = d["claude_code_kwargs"]
        if isinstance(raw_cc, dict):
            d["claude_code_kwargs"] = tuple(
                (key, tuple(value) if isinstance(value, list) else value)
                for key, value in sorted(raw_cc.items())
            )
        elif raw_cc is None:
            d["claude_code_kwargs"] = ()
    for key in ("singularity_binds", "singularity_extra_vllm_args"):
        if key in d:
            raw = d[key]
            if raw is None:
                d[key] = ()
            elif isinstance(raw, (list, tuple)):
                d[key] = tuple(str(x) for x in raw)
    if d.get("api_base") is not None and backend_type != "litellm":
        raise ValueError(
            f"backends.{name}.api_base is only valid when type == 'litellm' "
            f"(got {backend_type!r}); openai backends use backend_url"
        )
    if "openrouter_provider" in d:
        raw_provider = d["openrouter_provider"]
        if raw_provider is None:
            # Explicit null is treated as unset on any backend type.
            d["openrouter_provider"] = None
        else:
            if backend_type != "openrouter":
                raise ValueError(
                    f"backends.{name}.openrouter_provider is only valid when "
                    f"type == 'openrouter' (got {backend_type!r})"
                )
            d["openrouter_provider"] = _parse_openrouter_provider_preferences(
                raw_provider, f"backends.{name}.openrouter_provider"
            )
    if "sampling" in d:
        d["sampling"] = _parse_backend_sampling_config(
            d["sampling"], f"backends.{name}", backend_type=backend_type
        )
    return BackendConfig(**d)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration.

    Attributes:
        signal_type: Type of dense signal to evaluate.
        correlation_methods: Which correlation metrics to compute.
        estimations: Ground truth estimation configs. Each produces an
            independent set of ground truth values, and method predictions
            are correlated against each one.
        points_config: Evaluation point collection parameters.
        batch_size: Maximum concurrent rollouts/trajectories per lockstep batch.
            Used for MC estimation and trajectory collection. None means no limit.
    """

    signal_type: SignalType = SignalType.STATE_VALUE
    correlation_methods: tuple[CorrelationMethod, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
        CorrelationMethod.KENDALL_TAU,
    )
    estimations: tuple[EstimationConfig, ...] = (EstimationConfig(),)
    points_config: EvaluationPointsConfig = field(
        default_factory=EvaluationPointsConfig
    )
    batch_size: int = 256
    include_actor_thinking: bool = False
    discount_factor: float = 1.0
    eval_methods: tuple[EvalMethodConfig, ...] | None = None
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    default_backend: str | None = None
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Construct a BenchmarkConfig from a plain dictionary.

        Enum fields accept lowercase strings (e.g., ``"q_value"`` for
        ``SignalType.Q_VALUE``). The ``estimations`` field accepts a list
        of estimation config dicts. Unknown keys raise ``ValueError``
        to catch typos early.
        """
        _check_unknown_keys(data, cls)
        kwargs: dict[str, Any] = {}

        if "signal_type" in data:
            kwargs["signal_type"] = _parse_enum(
                data["signal_type"], SignalType, "signal_type"
            )

        if "correlation_methods" in data:
            kwargs["correlation_methods"] = tuple(
                _parse_enum(m, CorrelationMethod, "correlation_methods")
                for m in data["correlation_methods"]
            )

        if "estimations" in data:
            if data["estimations"] is not None:
                kwargs["estimations"] = tuple(
                    _parse_estimation_config(item) for item in data["estimations"]
                )
            else:
                kwargs["estimations"] = ()

        if "points_config" in data:
            pts_data = dict(data["points_config"])
            _check_unknown_keys(pts_data, EvaluationPointsConfig, "points_config")
            if "task_indices" in pts_data and pts_data["task_indices"] is not None:
                pts_data["task_indices"] = tuple(pts_data["task_indices"])
            if "sampling_strategy" in pts_data:
                strategy = str(pts_data["sampling_strategy"]).lower()
                if strategy not in _VALID_SAMPLING_STRATEGIES:
                    raise ValueError(
                        f"Invalid sampling_strategy: {pts_data['sampling_strategy']!r}. "
                        f"Valid values: {sorted(_VALID_SAMPLING_STRATEGIES)}"
                    )
                pts_data["sampling_strategy"] = strategy
            for field in _DISCARD_FIELDS:
                if field in pts_data and pts_data[field] is not None:
                    if not isinstance(pts_data[field], int) or pts_data[field] < 0:
                        raise ValueError(
                            f"{field} must be a non-negative integer, "
                            f"got {pts_data[field]!r}"
                        )
            kwargs["points_config"] = EvaluationPointsConfig(**pts_data)

        if "batch_size" in data:
            kwargs["batch_size"] = data["batch_size"]

        if "include_actor_thinking" in data:
            kwargs["include_actor_thinking"] = data["include_actor_thinking"]

        if "discount_factor" in data:
            kwargs["discount_factor"] = float(data["discount_factor"])

        if "eval_methods" in data:
            if data["eval_methods"] is not None:
                kwargs["eval_methods"] = tuple(
                    _parse_eval_method_config(item) for item in data["eval_methods"]
                )
            else:
                kwargs["eval_methods"] = None

        if "backends" in data:
            backends_raw = data["backends"] or {}
            if not isinstance(backends_raw, dict):
                raise ValueError("backends must be a dict of backend configs")
            parsed_backends: dict[str, BackendConfig] = {}
            for name, backend_data in backends_raw.items():
                if not isinstance(backend_data, dict):
                    raise ValueError(f"backends.{name} must be a dict")
                parsed_backends[name] = _parse_backend_config(name, backend_data)
            kwargs["backends"] = parsed_backends

        if "default_backend" in data:
            kwargs["default_backend"] = data["default_backend"]

        if "collection" in data:
            coll_data = dict(data["collection"])
            _check_unknown_keys(coll_data, CollectionConfig, "collection")
            if "backend_names" in coll_data and coll_data["backend_names"] is not None:
                coll_data["backend_names"] = tuple(coll_data["backend_names"])
            if (
                "ranking_backend_names" in coll_data
                and coll_data["ranking_backend_names"] is not None
            ):
                coll_data["ranking_backend_names"] = tuple(
                    coll_data["ranking_backend_names"]
                )
            if "harbor_state_capture" in coll_data:
                harbor_state_capture = str(coll_data["harbor_state_capture"]).lower()
                if harbor_state_capture not in _VALID_HARBOR_STATE_CAPTURE_MODES:
                    raise ValueError(
                        "Invalid harbor_state_capture: "
                        f"{coll_data['harbor_state_capture']!r}. "
                        f"Valid values: {sorted(_VALID_HARBOR_STATE_CAPTURE_MODES)}"
                    )
                coll_data["harbor_state_capture"] = harbor_state_capture
            if (
                "ranking_sampling" in coll_data
                and coll_data["ranking_sampling"] is not None
            ):
                coll_data["ranking_sampling"] = _parse_backend_sampling_config(
                    coll_data["ranking_sampling"], "collection.ranking_sampling"
                )
            if "ranking_sampling_mode" in coll_data:
                mode = str(coll_data["ranking_sampling_mode"]).lower()
                if mode not in _VALID_RANKING_SAMPLING_MODES:
                    raise ValueError(
                        f"Invalid ranking_sampling_mode: {coll_data['ranking_sampling_mode']!r}. "
                        f"Valid values: {sorted(_VALID_RANKING_SAMPLING_MODES)}"
                    )
                coll_data["ranking_sampling_mode"] = mode
            if "ranking_sampling_strategy" in coll_data:
                strategy = str(coll_data["ranking_sampling_strategy"]).lower()
                if strategy not in _VALID_RANKING_SAMPLING_STRATEGIES:
                    raise ValueError(
                        "Invalid ranking_sampling_strategy: "
                        f"{coll_data['ranking_sampling_strategy']!r}. "
                        f"Valid values: {sorted(_VALID_RANKING_SAMPLING_STRATEGIES)}"
                    )
                coll_data["ranking_sampling_strategy"] = strategy
            if "restore_concurrency" in coll_data:
                coll_data["restore_concurrency"] = _parse_positive_int(
                    coll_data["restore_concurrency"],
                    "restore_concurrency",
                )
            kwargs["collection"] = CollectionConfig(**coll_data)

        if "prediction" in data:
            pred_data = dict(data["prediction"])
            _check_unknown_keys(pred_data, PredictionConfig, "prediction")
            kwargs["prediction"] = PredictionConfig(**pred_data)

        config = cls(**kwargs)
        if (
            config.default_backend is not None
            and config.default_backend not in config.backends
        ):
            raise ValueError(
                f"default_backend {config.default_backend!r} not found in backends"
            )
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load a BenchmarkConfig from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed BenchmarkConfig.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML contains invalid keys or enum values.
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls.from_dict(data)


@dataclass(frozen=True)
class BackendsRegistryConfig:
    """Shared backend registry configuration."""

    backends: dict[str, BackendConfig]
    default_backend: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackendsRegistryConfig:
        _check_unknown_keys(data, cls)
        backends_raw = data.get("backends")
        if backends_raw is None:
            raise ValueError("backends registry must include 'backends'")
        if not isinstance(backends_raw, dict):
            raise ValueError("backends must be a dict of backend configs")
        parsed_backends: dict[str, BackendConfig] = {}
        for name, backend_data in backends_raw.items():
            if not isinstance(backend_data, dict):
                raise ValueError(f"backends.{name} must be a dict")
            parsed_backends[name] = _parse_backend_config(name, backend_data)
        default_backend = data.get("default_backend")
        config = cls(backends=parsed_backends, default_backend=default_backend)
        if (
            config.default_backend is not None
            and config.default_backend not in config.backends
        ):
            raise ValueError(
                f"default_backend {config.default_backend!r} not found in backends"
            )
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> BackendsRegistryConfig:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


@dataclass(frozen=True)
class PipelineCollectionConfig:
    """Configuration for scripts/pipeline/collect_dataset.py."""

    backends_config_path: str = "shared/configs/backends.yaml"
    environment: str | None = None
    dataset_path: str | None = None
    resume: bool = True
    resume_ranking_from: str | None = None
    resume_ranking_subset: ResumeRankingSubsetConfig | None = None
    correlation_methods: tuple[CorrelationMethod, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
        CorrelationMethod.KENDALL_TAU,
    )
    points_config: EvaluationPointsConfig = field(
        default_factory=EvaluationPointsConfig
    )
    batch_size: int = 256
    include_actor_thinking: bool = False
    discount_factor: float = 1.0
    max_aborted_points: float = 0.1
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    replay_validation: ReplayValidationConfig = field(
        default_factory=ReplayValidationConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineCollectionConfig:
        _check_unknown_keys(data, cls)
        if "backends" in data or "default_backend" in data:
            raise ValueError(
                "Collection config must not define backends; use backends_config_path"
            )
        if "signal_type" in data:
            raise ValueError(
                "Collection config must not define signal_type; "
                "signal type belongs in prediction/evaluation configs"
            )
        kwargs: dict[str, Any] = {}
        if "correlation_methods" in data:
            kwargs["correlation_methods"] = tuple(
                _parse_enum(m, CorrelationMethod, "correlation_methods")
                for m in data["correlation_methods"]
            )
        if "points_config" in data:
            pts_data = dict(data["points_config"])
            _check_unknown_keys(pts_data, EvaluationPointsConfig, "points_config")
            if "task_indices" in pts_data and pts_data["task_indices"] is not None:
                pts_data["task_indices"] = tuple(pts_data["task_indices"])
            if "sampling_strategy" in pts_data:
                strategy = str(pts_data["sampling_strategy"]).lower()
                if strategy not in _VALID_SAMPLING_STRATEGIES:
                    raise ValueError(
                        f"Invalid sampling_strategy: {pts_data['sampling_strategy']!r}. "
                        f"Valid values: {sorted(_VALID_SAMPLING_STRATEGIES)}"
                    )
                pts_data["sampling_strategy"] = strategy
            for field in _DISCARD_FIELDS:
                if field in pts_data and pts_data[field] is not None:
                    if not isinstance(pts_data[field], int) or pts_data[field] < 0:
                        raise ValueError(
                            f"{field} must be a non-negative integer, "
                            f"got {pts_data[field]!r}"
                        )
            kwargs["points_config"] = EvaluationPointsConfig(**pts_data)
        if "collection" in data:
            coll_data = dict(data["collection"])
            _check_unknown_keys(coll_data, CollectionConfig, "collection")
            if "backend_names" in coll_data and coll_data["backend_names"] is not None:
                coll_data["backend_names"] = tuple(coll_data["backend_names"])
            if (
                "ranking_backend_names" in coll_data
                and coll_data["ranking_backend_names"] is not None
            ):
                coll_data["ranking_backend_names"] = tuple(
                    coll_data["ranking_backend_names"]
                )
            if (
                "ranking_sampling" in coll_data
                and coll_data["ranking_sampling"] is not None
            ):
                coll_data["ranking_sampling"] = _parse_backend_sampling_config(
                    coll_data["ranking_sampling"], "collection.ranking_sampling"
                )
            if "ranking_sampling_mode" in coll_data:
                mode = str(coll_data["ranking_sampling_mode"]).lower()
                if mode not in _VALID_RANKING_SAMPLING_MODES:
                    raise ValueError(
                        f"Invalid ranking_sampling_mode: {coll_data['ranking_sampling_mode']!r}. "
                        f"Valid values: {sorted(_VALID_RANKING_SAMPLING_MODES)}"
                    )
                coll_data["ranking_sampling_mode"] = mode
            if "ranking_sampling_strategy" in coll_data:
                strategy = str(coll_data["ranking_sampling_strategy"]).lower()
                if strategy not in _VALID_RANKING_SAMPLING_STRATEGIES:
                    raise ValueError(
                        "Invalid ranking_sampling_strategy: "
                        f"{coll_data['ranking_sampling_strategy']!r}. "
                        f"Valid values: {sorted(_VALID_RANKING_SAMPLING_STRATEGIES)}"
                    )
                coll_data["ranking_sampling_strategy"] = strategy
            if "restore_concurrency" in coll_data:
                coll_data["restore_concurrency"] = _parse_positive_int(
                    coll_data["restore_concurrency"],
                    "restore_concurrency",
                )
            kwargs["collection"] = CollectionConfig(**coll_data)
        if "batch_size" in data:
            kwargs["batch_size"] = data["batch_size"]
        if "include_actor_thinking" in data:
            kwargs["include_actor_thinking"] = data["include_actor_thinking"]
        if "discount_factor" in data:
            kwargs["discount_factor"] = float(data["discount_factor"])
        if "max_aborted_points" in data:
            kwargs["max_aborted_points"] = float(data["max_aborted_points"])
        if "dataset_path" in data:
            kwargs["dataset_path"] = data["dataset_path"]
        if "resume" in data:
            kwargs["resume"] = bool(data["resume"])
        if "resume_ranking_from" in data:
            kwargs["resume_ranking_from"] = data["resume_ranking_from"]
        if "resume_ranking_subset" in data:
            subset_data = data["resume_ranking_subset"]
            if subset_data is not None:
                if not isinstance(subset_data, dict):
                    raise ValueError("resume_ranking_subset must be a dict")
                _check_unknown_keys(
                    subset_data,
                    ResumeRankingSubsetConfig,
                    "resume_ranking_subset",
                )
                if "count" not in subset_data:
                    raise ValueError("resume_ranking_subset.count is required")
                count = int(subset_data["count"])
                if count <= 0:
                    raise ValueError("resume_ranking_subset.count must be positive")
                kwargs["resume_ranking_subset"] = ResumeRankingSubsetConfig(
                    count=count,
                    seed=(
                        None
                        if subset_data.get("seed") is None
                        else int(subset_data["seed"])
                    ),
                )
        if "environment" in data:
            kwargs["environment"] = data["environment"]
        if "backends_config_path" in data:
            kwargs["backends_config_path"] = data["backends_config_path"]
        if "replay_validation" in data:
            rv_data = dict(data["replay_validation"])
            _check_unknown_keys(rv_data, ReplayValidationConfig, "replay_validation")
            kwargs["replay_validation"] = ReplayValidationConfig(**rv_data)
        if (
            kwargs.get("resume_ranking_subset") is not None
            and kwargs.get("resume_ranking_from") is None
        ):
            raise ValueError("resume_ranking_subset requires resume_ranking_from")
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineCollectionConfig:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


@dataclass(frozen=True)
class PipelinePredictionConfig:
    """Configuration for scripts/pipeline/predict.py."""

    backends_config_path: str = "shared/configs/backends.yaml"
    dataset_path: str | None = None
    predictions_dir: str | None = None
    batch_size: int = 256
    include_actor_thinking: bool = False
    discount_factor: float = 1.0
    step_penalty: float | None = None
    estimations: tuple[EstimationConfig, ...] = ()
    eval_methods: tuple[EvalMethodConfig, ...] | None = None
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    harbor_replay_cache: bool = False
    max_aborted_points: float = 0.1
    max_eval_points: int | None = None
    skip_ranking_predictions: bool = False
    mc_rollout_persistence: MCRolloutPersistenceConfig = field(
        default_factory=MCRolloutPersistenceConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelinePredictionConfig:
        _check_unknown_keys(data, cls)
        if "backends" in data or "default_backend" in data:
            raise ValueError(
                "Prediction config must not define backends; use backends_config_path"
            )
        if "signal_type" in data:
            raise ValueError(
                "Prediction config must not define signal_type; "
                "signal type belongs in eval_methods[] entries"
            )
        kwargs: dict[str, Any] = {}
        if "estimations" in data:
            if data["estimations"] is not None:
                kwargs["estimations"] = tuple(
                    _parse_estimation_config(item) for item in data["estimations"]
                )
            else:
                kwargs["estimations"] = ()
        if "eval_methods" in data:
            if data["eval_methods"] is not None:
                kwargs["eval_methods"] = tuple(
                    _parse_eval_method_config(item) for item in data["eval_methods"]
                )
            else:
                kwargs["eval_methods"] = None
        if "prediction" in data:
            pred_data = dict(data["prediction"])
            _check_unknown_keys(pred_data, PredictionConfig, "prediction")
            kwargs["prediction"] = PredictionConfig(**pred_data)
        if "mc_rollout_persistence" in data:
            raw_persistence = data["mc_rollout_persistence"]
            if raw_persistence is None:
                kwargs["mc_rollout_persistence"] = MCRolloutPersistenceConfig()
            else:
                persistence_data = dict(raw_persistence)
                if "enabled" in persistence_data:
                    _log.warning(
                        "mc_rollout_persistence.enabled is deprecated and "
                        "ignored; persistence is always on. Remove 'enabled' "
                        "from your config."
                    )
                    del persistence_data["enabled"]
                _check_unknown_keys(
                    persistence_data,
                    MCRolloutPersistenceConfig,
                    "mc_rollout_persistence",
                )
                kwargs["mc_rollout_persistence"] = MCRolloutPersistenceConfig(
                    **persistence_data
                )
        if "batch_size" in data:
            kwargs["batch_size"] = data["batch_size"]
        if "include_actor_thinking" in data:
            kwargs["include_actor_thinking"] = data["include_actor_thinking"]
        if "discount_factor" in data:
            kwargs["discount_factor"] = float(data["discount_factor"])
        if "step_penalty" in data:
            sp = data["step_penalty"]
            kwargs["step_penalty"] = float(sp) if sp is not None else None
        if "dataset_path" in data:
            kwargs["dataset_path"] = data["dataset_path"]
        if "predictions_dir" in data:
            kwargs["predictions_dir"] = data["predictions_dir"]
        if "backends_config_path" in data:
            kwargs["backends_config_path"] = data["backends_config_path"]
        if "harbor_replay_cache" in data:
            kwargs["harbor_replay_cache"] = bool(data["harbor_replay_cache"])
        if "max_aborted_points" in data:
            kwargs["max_aborted_points"] = float(data["max_aborted_points"])
        if "max_eval_points" in data:
            mep = data["max_eval_points"]
            kwargs["max_eval_points"] = int(mep) if mep is not None else None
        if "skip_ranking_predictions" in data:
            kwargs["skip_ranking_predictions"] = bool(data["skip_ranking_predictions"])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelinePredictionConfig:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


@dataclass(frozen=True)
class PredictionSourceConfig:
    """Prediction source configuration for evaluation.

    ``method_name_prefix`` is applied after ``include_methods`` /
    ``exclude_methods`` filtering, so filters reference the stored
    (un-prefixed) ``method_name``. Useful for disambiguating artifacts that
    share a method name across runs (e.g., the same GT method computed under
    different backends).
    """

    path: str | None = None
    paths: tuple[str, ...] | None = None
    type: str | None = None
    include_methods: tuple[str, ...] | None = None
    exclude_methods: tuple[str, ...] | None = None
    method_name_prefix: str | None = None


@dataclass(frozen=True)
class ComparisonConfig:
    """Explicit GT/eval comparison pairing for evaluation.

    ``eval`` can be a single method name or a tuple of method names
    (1:N pairing). In YAML, a list of strings is accepted.
    """

    gt: str
    eval: str | tuple[str, ...]


@dataclass(frozen=True)
class GtComparisonConfig:
    """Explicit GT-vs-GT comparison pairing for evaluation.

    Both sides must be ``"gt"``-type predictions sharing the same
    ``assumption``. ``rhs`` can be a single method name or a tuple of
    method names (1:N pairing). In YAML, a list of strings is accepted.
    """

    lhs: str
    rhs: str | tuple[str, ...]


@dataclass(frozen=True)
class PipelineEvaluationConfig:
    """Configuration for scripts/pipeline/evaluate.py."""

    dataset_path: str | None = None
    output_dir: str | None = None
    prediction_sources: tuple[PredictionSourceConfig, ...] = ()
    comparisons: tuple[ComparisonConfig, ...] | None = None
    gt_vs_gt_comparisons: tuple[GtComparisonConfig, ...] | None = None
    correlation_methods: tuple[CorrelationMethod, ...] | None = None
    signal_type: SignalType | None = None
    exclude_zero_points: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineEvaluationConfig:
        _check_unknown_keys(data, cls)
        kwargs: dict[str, Any] = {}
        if "dataset_path" in data:
            kwargs["dataset_path"] = data["dataset_path"]
        if "output_dir" in data:
            kwargs["output_dir"] = data["output_dir"]
        if "correlation_methods" in data:
            kwargs["correlation_methods"] = tuple(
                _parse_enum(m, CorrelationMethod, "correlation_methods")
                for m in data["correlation_methods"]
            )
        if "signal_type" in data:
            kwargs["signal_type"] = _parse_enum(
                data["signal_type"], SignalType, "signal_type"
            )
        if "exclude_zero_points" in data:
            kwargs["exclude_zero_points"] = bool(data["exclude_zero_points"])
        if "prediction_sources" in data:
            sources: list[PredictionSourceConfig] = []
            for raw in data["prediction_sources"]:
                src = dict(raw)
                _check_unknown_keys(src, PredictionSourceConfig, "prediction_sources[]")
                if "paths" in src and src["paths"] is not None:
                    src["paths"] = tuple(src["paths"])
                if "include_methods" in src and src["include_methods"] is not None:
                    src["include_methods"] = tuple(src["include_methods"])
                if "exclude_methods" in src and src["exclude_methods"] is not None:
                    src["exclude_methods"] = tuple(src["exclude_methods"])
                if "type" in src and src["type"] is not None:
                    src["type"] = str(src["type"]).lower()
                sources.append(PredictionSourceConfig(**src))
            kwargs["prediction_sources"] = tuple(sources)
        if "comparisons" in data:
            if data["comparisons"] is None:
                kwargs["comparisons"] = None
            else:
                comparisons: list[ComparisonConfig] = []
                for raw in data["comparisons"]:
                    item = dict(raw)
                    _check_unknown_keys(item, ComparisonConfig, "comparisons[]")
                    # Normalize eval: str → tuple, list → tuple
                    if "eval" in item:
                        eval_val = item["eval"]
                        if isinstance(eval_val, str):
                            item["eval"] = (eval_val,)
                        elif isinstance(eval_val, (list, tuple)):
                            item["eval"] = tuple(eval_val)
                    comparisons.append(ComparisonConfig(**item))
                kwargs["comparisons"] = tuple(comparisons)
        if "gt_vs_gt_comparisons" in data:
            if data["gt_vs_gt_comparisons"] is None:
                kwargs["gt_vs_gt_comparisons"] = None
            else:
                gt_pairs: list[GtComparisonConfig] = []
                for raw in data["gt_vs_gt_comparisons"]:
                    item = dict(raw)
                    _check_unknown_keys(
                        item, GtComparisonConfig, "gt_vs_gt_comparisons[]"
                    )
                    if "rhs" in item:
                        rhs_val = item["rhs"]
                        if isinstance(rhs_val, str):
                            item["rhs"] = (rhs_val,)
                        elif isinstance(rhs_val, (list, tuple)):
                            item["rhs"] = tuple(rhs_val)
                    gt_pairs.append(GtComparisonConfig(**item))
                kwargs["gt_vs_gt_comparisons"] = tuple(gt_pairs)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineEvaluationConfig:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


@dataclass(frozen=True)
class PipelinePassAtKConfig:
    """Configuration for scripts/pipeline/compute_pass_at_k.py.

    Attributes:
        backends_config_path: Path to the backends registry YAML.
        output_path: Path for the JSON output file.
        batch_size: Maximum concurrent trajectories per batch.
        context_name: Environment context YAML name.
        contexts_dir: Directory containing environment context YAMLs.
        backend_name: Single backend name to evaluate.
        seed: Random seed for task shuffling.
        system_prompt: Override system prompt.
        make_kwargs: Extra kwargs passed to the environment adapter.
        max_history_turns: Truncate history to last N turns.
        debug: Enable verbose debug logging.
        samples_per_task: Number of independent runs per task (N).
        k_values: Which k values to compute Pass@k for.
        num_tasks: Number of tasks to evaluate (None = all available).
        task_indices: Explicit task indices (overrides num_tasks/shuffle).
        shuffle_tasks: Whether to shuffle task order before selection.
        success_threshold: Minimum cumulative ``reward_signal_name`` return
            for a trajectory to count as successful. Defaults to ``1.0`` —
            matches binary-reward environments bit-for-bit. Raise for
            dense-reward environments (Craftax, Jericho) where clean Pass@k
            semantics require a game-specific threshold.
        cli_backend_tmux_prompt_hardening: When ``True``, CLI-backed Harbor
            runs (``codex`` or ``claude_code``) using
            ``text_exec_mode: tmux_session`` append extra system-prompt
            guidance reminding the model that it is not in a normal CLI tool
            session and must emit one non-interactive shell command for the
            external tmux runner. Mirrors the prediction-step flag.
    """

    backends_config_path: str = "shared/configs/backends.yaml"
    output_path: str | None = None
    batch_size: int = 256
    context_name: str | None = None
    contexts_dir: str = "shared/configs/environments/"
    backend_name: str | None = None
    seed: int = 42
    system_prompt: str | None = None
    make_kwargs: dict[str, Any] | None = None
    max_history_turns: int | None = None
    debug: bool = False
    samples_per_task: int = 20
    k_values: tuple[int, ...] = (1, 5, 10, 20)
    num_tasks: int | None = None
    task_indices: tuple[int, ...] | None = None
    shuffle_tasks: bool = True
    max_aborted_points: float = 0.1
    success_threshold: float = 1.0
    cli_backend_tmux_prompt_hardening: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelinePassAtKConfig:
        _check_unknown_keys(data, cls)
        kwargs: dict[str, Any] = {}
        for key in (
            "backends_config_path",
            "output_path",
            "batch_size",
            "context_name",
            "contexts_dir",
            "backend_name",
            "seed",
            "system_prompt",
            "make_kwargs",
            "max_history_turns",
            "debug",
            "shuffle_tasks",
            "num_tasks",
            "max_aborted_points",
            "success_threshold",
            "cli_backend_tmux_prompt_hardening",
        ):
            if key in data:
                kwargs[key] = data[key]
        if "success_threshold" in kwargs:
            kwargs["success_threshold"] = float(kwargs["success_threshold"])
        if "num_tasks" in kwargs and kwargs["num_tasks"] is not None:
            nt = int(kwargs["num_tasks"])
            if nt < 1:
                raise ValueError("num_tasks must be >= 1")
            kwargs["num_tasks"] = nt
        if "samples_per_task" in data:
            spt = int(data["samples_per_task"])
            if spt < 1:
                raise ValueError("samples_per_task must be >= 1")
            kwargs["samples_per_task"] = spt
        if "k_values" in data:
            k_list = list(data["k_values"])
            if not k_list:
                raise ValueError("k_values must not be empty")
            for k in k_list:
                if int(k) < 1:
                    raise ValueError(f"All k_values entries must be >= 1, got {k}")
            kwargs["k_values"] = tuple(int(k) for k in k_list)
        if "task_indices" in data:
            ti = data["task_indices"]
            kwargs["task_indices"] = tuple(ti) if ti is not None else None
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelinePassAtKConfig:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


@dataclass(frozen=True)
class TestTimeScalingActorConfig:
    """Guided actor configuration for test-time scaling experiments."""

    k: int = 1
    scorer: EvalMethodConfig | None = None


_TEST_TIME_SCALING_ACTION_SIGNAL_TYPES = {
    SignalType.Q_VALUE,
    SignalType.ADVANTAGE,
    SignalType.SHAPED_REWARD,
}


def _parse_test_time_scaling_actor_config(
    data: dict[str, Any] | None,
) -> TestTimeScalingActorConfig:
    raw = {} if data is None else dict(data)
    _check_unknown_keys(raw, TestTimeScalingActorConfig, "guided_actor")
    kwargs: dict[str, Any] = {}
    if "k" in raw:
        k = int(raw["k"])
        if k < 1:
            raise ValueError("guided_actor.k must be >= 1")
        kwargs["k"] = k
    if "scorer" in raw:
        scorer_raw = raw["scorer"]
        if scorer_raw is not None:
            # Any benchmarked method may guide the actor: individual-prediction
            # methods are scored per candidate, ranking methods rank candidates
            # directly. ``_parse_eval_method_config`` already validates the type.
            scorer = _parse_eval_method_config(dict(scorer_raw))
            if scorer.signal_type not in _TEST_TIME_SCALING_ACTION_SIGNAL_TYPES:
                raise ValueError(
                    "Test-time scaling scorer must use an action-conditioned "
                    "signal_type: q_value, advantage, or shaped_reward"
                )
            if scorer.backend_name is None and scorer.type != "baseline_random":
                raise ValueError(
                    "guided_actor.scorer.backend_name is required so the "
                    "evaluator backend is explicit and separate from the actor"
                )
            if scorer.type == "llm_direct_gvl":
                raise ValueError(
                    "llm_direct_gvl cannot be used as a test-time-scaling scorer: "
                    "it labels values over a collected trajectory set "
                    "(Dataset.trajectory_results), which does not exist at test "
                    "time. Supporting it would require feeding the live actor "
                    "trajectory-so-far as trajectory_results."
                )
            kwargs["scorer"] = scorer
    return TestTimeScalingActorConfig(**kwargs)


@dataclass(frozen=True)
class TestTimeScalingSelfConsistencyConfig:
    """Self-consistency baseline configuration for test-time scaling.

    The baseline reuses ``guided_actor.k`` samples per step (matched compute)
    and selects an action by consensus instead of by a learned scorer:

    * ``discrete`` — vote over each candidate's resolved action (small,
      enumerable action spaces).
    * ``open_ended`` — a selector LLM (``backend_name``) picks the most
      consistent action among the candidates' extracted actions
      (Universal Self-Consistency).
    """

    mode: str = "discrete"
    backend_name: str | None = None
    max_candidate_chars: int = 4000


_TEST_TIME_SCALING_SELF_CONSISTENCY_MODES = {"discrete", "open_ended"}


def _parse_test_time_scaling_self_consistency_config(
    data: dict[str, Any] | None,
) -> TestTimeScalingSelfConsistencyConfig:
    raw = {} if data is None else dict(data)
    _check_unknown_keys(
        raw, TestTimeScalingSelfConsistencyConfig, "self_consistency"
    )
    kwargs: dict[str, Any] = {}
    mode = str(raw.get("mode", "discrete"))
    if mode not in _TEST_TIME_SCALING_SELF_CONSISTENCY_MODES:
        raise ValueError(
            "self_consistency.mode must be 'discrete' or 'open_ended', "
            f"got {mode!r}"
        )
    kwargs["mode"] = mode
    backend_name = raw.get("backend_name")
    if mode == "open_ended" and backend_name is None:
        raise ValueError(
            "self_consistency.backend_name is required when mode='open_ended' "
            "(the selector LLM that picks the most consistent action)"
        )
    if backend_name is not None:
        kwargs["backend_name"] = str(backend_name)
    if "max_candidate_chars" in raw:
        mcc = int(raw["max_candidate_chars"])
        if mcc < 1:
            raise ValueError("self_consistency.max_candidate_chars must be >= 1")
        kwargs["max_candidate_chars"] = mcc
    return TestTimeScalingSelfConsistencyConfig(**kwargs)


@dataclass(frozen=True)
class PipelineTestTimeScalingConfig:
    """Configuration for scripts/pipeline/test_time_scaling.py."""

    backends_config_path: str = "shared/configs/backends.yaml"
    output_path: str | None = None
    batch_size: int = 256
    context_name: str | None = None
    contexts_dir: str = "shared/configs/environments/"
    actor_backend_name: str | None = None
    guided_actor: TestTimeScalingActorConfig = field(
        default_factory=TestTimeScalingActorConfig
    )
    self_consistency: TestTimeScalingSelfConsistencyConfig | None = None
    seed: int = 42
    system_prompt: str | None = None
    make_kwargs: dict[str, Any] | None = None
    max_history_turns: int | None = None
    debug: bool = False
    samples_per_task: int = 1
    num_tasks: int | None = None
    task_indices: tuple[int, ...] | None = None
    shuffle_tasks: bool = True
    max_aborted_points: float = 0.1
    success_threshold: float = 1.0
    cli_backend_tmux_prompt_hardening: bool = False
    max_concurrent_trajectories: int = 32
    restore_concurrency: int = 16
    resume: bool = True
    harbor_replay_cache: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineTestTimeScalingConfig:
        _check_unknown_keys(data, cls)
        kwargs: dict[str, Any] = {}
        for key in (
            "backends_config_path",
            "output_path",
            "batch_size",
            "context_name",
            "contexts_dir",
            "actor_backend_name",
            "seed",
            "system_prompt",
            "make_kwargs",
            "max_history_turns",
            "debug",
            "shuffle_tasks",
            "num_tasks",
            "max_aborted_points",
            "success_threshold",
            "cli_backend_tmux_prompt_hardening",
        ):
            if key in data:
                kwargs[key] = data[key]
        if "guided_actor" in data:
            kwargs["guided_actor"] = _parse_test_time_scaling_actor_config(
                data["guided_actor"]
            )
        if "self_consistency" in data and data["self_consistency"] is not None:
            kwargs["self_consistency"] = (
                _parse_test_time_scaling_self_consistency_config(
                    data["self_consistency"]
                )
            )
        if "success_threshold" in kwargs:
            kwargs["success_threshold"] = float(kwargs["success_threshold"])
        if "max_aborted_points" in kwargs:
            kwargs["max_aborted_points"] = float(kwargs["max_aborted_points"])
        if "num_tasks" in kwargs and kwargs["num_tasks"] is not None:
            nt = int(kwargs["num_tasks"])
            if nt < 1:
                raise ValueError("num_tasks must be >= 1")
            kwargs["num_tasks"] = nt
        if "samples_per_task" in data:
            spt = int(data["samples_per_task"])
            if spt < 1:
                raise ValueError("samples_per_task must be >= 1")
            kwargs["samples_per_task"] = spt
        if "task_indices" in data:
            ti = data["task_indices"]
            kwargs["task_indices"] = tuple(ti) if ti is not None else None
        if "max_concurrent_trajectories" in data:
            kwargs["max_concurrent_trajectories"] = _parse_positive_int(
                data["max_concurrent_trajectories"], "max_concurrent_trajectories"
            )
        if "restore_concurrency" in data:
            kwargs["restore_concurrency"] = _parse_positive_int(
                data["restore_concurrency"], "restore_concurrency"
            )
        if "resume" in data:
            kwargs["resume"] = bool(data["resume"])
        if "harbor_replay_cache" in data:
            kwargs["harbor_replay_cache"] = bool(data["harbor_replay_cache"])
        if kwargs.get("self_consistency") is not None:
            guided = kwargs.get("guided_actor") or TestTimeScalingActorConfig()
            if guided.k <= 1:
                raise ValueError(
                    "self_consistency requires guided_actor.k > 1 so there are "
                    "multiple samples to reach consensus over (matched compute)"
                )
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineTestTimeScalingConfig:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
