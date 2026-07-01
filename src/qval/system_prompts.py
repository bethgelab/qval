"""Three-layer actor system prompt composition.

Layer 1 — **ENV_PROMPTS**: per-environment role descriptions (scheme-agnostic).
Layer 2 — **SCHEME_INSTRUCTIONS**: per-scheme formatting rules (env-agnostic).
Layer 3 — **SCHEME_EXAMPLES**: per-environment × per-scheme concrete examples.

Final prompt = layer1 + layer2 + layer3, joined by blank lines.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Layer 1: per-env role prompts (no formatting, no examples)
# ---------------------------------------------------------------------------

ENV_PROMPTS: dict[str, str] = {
    "reasoning_gym": (
        "You are solving a reasoning task. Think through the problem carefully. "
        "You are rewarded for providing the correct answer."
    ),
    "alfworld": (
        "You are an agent interacting with a text-based household environment. "
        "On each turn you receive a description of your surroundings and a list "
        "of admissible commands. Your goal is to complete the stated objective. "
        "You are rewarded for successfully completing the task."
    ),
    "jericho": (
        "You are playing a classic text adventure game. Read the game's "
        "descriptions carefully and respond with a command to interact with "
        "the world. Common commands include: go [direction], take [object], "
        "open [object], examine [object], inventory, look. Your goal is to "
        "maximize your score by exploring, solving puzzles, and collecting "
        "treasures."
    ),
    "webshop": (
        "You are shopping on a simulated e-commerce website to find and "
        "purchase a product matching a given instruction — including all "
        "specified attributes (color, size, material, price range, etc.). "
        "Reward is assigned at the end based on how closely the purchased "
        "product matches the instruction.\n\n"
        "## Observation format\n"
        "Each turn you see the shopping goal inside <goal>…</goal> and the "
        "current page inside <page>…</page>. Clickable elements in the page "
        "appear as [button] <text> [button_]; product listings use the "
        "product's ASIN (e.g. B09KLQLLT2) as their clickable text.\n\n"
        "## Actions — exactly two verbs\n"
        "  search[<keywords>]\n"
        "    Free-form keyword search (BM25 over ~1.18M products). A search "
        "bar exists only on the homepage and the \"Back to Search\" page — a "
        "search[...] issued anywhere else wastes the turn. After a prior "
        "search, click[Back to Search] before searching again. Keep keywords "
        "concise and grounded in the instruction's attributes.\n\n"
        "  click[<element>]\n"
        "    Click a visible element on the current page. The argument must "
        "match the text of a [button] <text> [button_] on the current page "
        "(case-insensitive). You can click: product ASINs from search "
        "results; option values (e.g. blue, small) on item pages; navigation "
        "(\"Back to Search\", \"Next >\", \"< Prev\"); detail tabs "
        "(\"Description\", \"Features\", \"Reviews\", \"Attributes\"); and "
        "\"Buy Now\" to purchase. Selecting a size/color option on an item "
        "page sets that attribute on the purchase — click the options "
        "matching the instruction before click[Buy Now]. click[Buy Now] is "
        "terminal: it commits the current item and selected options and ends "
        "the episode.\n\n"
        "## Validity\n"
        "If your action has an unknown verb, empty keywords, or a click "
        "target that isn't on the current page, the turn is wasted: the "
        "page stays the same and you lose one of your remaining steps. "
        "Before clicking, verify the target appears inside [button] "
        "<text> [button_] in the current <page>."
    ),
    # {command_timeout_desc} and {max_steps_clause} are filled by
    # build_actor_system_prompt().
    "harbor": (
        "You are an AI agent with access to a Linux terminal inside a "
        "minimal container. Internet access is available. No user "
        "accounts, API tokens, or credentials are configured — do not "
        "attempt to log in to external services. "
        "You already have full permissions inside the container — do "
        "not use su or sudo. "
        "Execute commands to complete the task described below. "
        "Work step by step, checking the output of each command "
        "before proceeding. {max_steps_clause} "
        "Commands that run longer than {command_timeout_desc} are "
        "automatically cancelled. If a command is cancelled, adapt "
        "your approach: break the work into smaller steps, add flags "
        "that limit scope, or try a different method. "
        "Avoid commands that wait for interactive input or leave the "
        "shell waiting. Always use non-interactive flags (e.g., -y for "
        "package managers). Avoid bare cat with no file, bare python/python3 "
        "with no script or -c, ssh, su, interactive editors, or unfinished "
        "quotes/heredocs. Heredocs are allowed only when fully terminated in "
        "the same action. "
        "Never run exit, logout, exec ..., kill the shell process, or execute "
        "any command that terminates the current shell or tmux session. If you "
        "need to stop a program, stop that program rather than the shell. "
        "Destroying the shell session loses all progress and fails the task. "
        "Do not use shell-exit idioms such as `cmd; status=$?; cleanup; exit "
        "$status` or `cmd || exit 1`. If you need to preserve a command's "
        "failure status after cleanup, use a non-terminating check such as "
        "`cmd; status=$?; cleanup; test \"$status\" -ne 0`, or use an if/else "
        "that ends with `false` on unexpected success. The word `exit` is "
        "allowed inside scripts you create, but not as a top-level command in "
        "your action. "
        "When using grep, cat, or other commands that may produce long "
        "output, pipe through head/tail or limit the results to avoid "
        "flooding your context. "
        "Your work will be tested for correctness when you submit. "
        "When you have completed the task, submit by sending the "
        "SUBMIT command. SUBMIT must be the entire content of the "
        "action — do not combine it with any other commands or text "
        "in the same turn. Mixing SUBMIT with other commands ends "
        "the episode immediately without running those commands."
    ),
    "craftax": (
        "You are playing Craftax, an open-ended survival game. Explore the world, "
        "gather resources, craft tools and weapons, fight monsters, and try to "
        "survive as long as possible. You are rewarded for making progress: "
        "crafting new items, exploring new areas, defeating enemies, and keeping "
        "yourself alive. Taking damage or letting your vitals drop hurts your "
        "score."
    ),
    "open_apps": (
        "You are an AI agent interacting with a web application through a browser. "
        "On each turn you see the page's accessibility tree, which lists UI elements "
        "with their browser IDs (bid). Complete the given task by interacting with "
        "the UI elements.\n\n"
        "Available actions:\n"
        "  click('bid')              — click an element\n"
        "  fill('bid', 'text')       — clear an input field and type text\n"
        "  select_option('bid', 'option') — select a dropdown option\n"
        "  scroll(x, y)              — scroll the page by (x, y) pixels\n"
        "  press('key')              — press a keyboard key (e.g. 'Enter')\n"
        "  hover('bid')              — hover over an element\n"
        "  go_back()                 — navigate back\n"
        "  noop()                    — do nothing (wait)\n\n"
        "Use the bid shown in the accessibility tree for the element you want "
        "to interact with. You are rewarded for successfully completing the task."
    ),
}

CLI_BACKEND_TMUX_PROMPT_HARDENING = (
    "Important execution model for CLI-driven rollout backends: this is not "
    "a normal CLI tool session. You do not call tools directly, and you do "
    "not have access to the backend's native terminal/tool harness (e.g., "
    "Codex tool calls or Claude Code's built-in Bash/Edit tools). Your only "
    "control is the command text you place in the action field; an external "
    "runner copies that command into the task's persistent tmux shell and "
    "returns the output later as an observation. Do not write as if you can "
    "use apply_patch, editor tools, Python helpers outside the container, or "
    "multi-step interactive control. Produce exactly one complete, "
    "non-interactive shell command for the runner to execute."
)

# ---------------------------------------------------------------------------
# Layer 2: per-scheme formatting instructions (env-agnostic, no examples)
# ---------------------------------------------------------------------------

SCHEME_INSTRUCTIONS: dict[str, str] = {
    "answer_tags": (
        "Always provide your response inside <answer> tags. The response must be "
        "the only content inside the tags. Do not output anything after the "
        "closing </answer> tag."
    ),
    "react_tags": (
        "Before each action, explain your reasoning inside <thought> tags (1 sentence "
        "for straightforward decisions, up to 5 for complex ones), then provide your "
        "command inside <action> tags. Provide exactly one thought and one action per "
        "turn. The command must be the only content inside the <action> tags. Do not "
        "output anything after the closing </action> tag."
    ),
    "react_classic": (
        'Before each action, explain your reasoning on a line starting with '
        '"Thought:" (1 sentence for straightforward decisions, up to 5 for complex '
        'ones), then provide your command on a line starting with "Action:". '
        "Provide exactly one thought and one action per turn. Do not output "
        "anything after the action."
    ),
}

RANKING_SCHEME_INSTRUCTIONS: dict[str, str] = {
    "answer_tags": SCHEME_INSTRUCTIONS["answer_tags"],
    "react_tags": (
        "Provide exactly one command inside <action> tags. Do not include "
        "<thought> tags, explanations, or extra text. The command must be the "
        "only content inside the <action> tags. Do not output anything after "
        "the closing </action> tag."
    ),
    "react_classic": (
        'Provide exactly one command on a line starting with "Action:". '
        "Do not include a Thought line, explanations, or extra text. "
        "Do not output anything after the action."
    ),
}

# ---------------------------------------------------------------------------
# Layer 3: per-env × per-scheme examples
# ---------------------------------------------------------------------------

SCHEME_EXAMPLES: dict[str, dict[str, str]] = {
    # -- reasoning_gym --
    "reasoning_gym": {
        "answer_tags": "Example:\n<answer>42</answer>",
        "react_tags": (
            "Example:\n"
            "<thought>The problem asks for the sum of 20 and 22.</thought>\n"
            "<action>42</action>"
        ),
        "react_classic": (
            "Example:\n"
            "Thought: The problem asks for the sum of 20 and 22.\n"
            "Action: 42"
        ),
    },
    # -- alfworld --
    "alfworld": {
        "answer_tags": "Example:\n<answer>go to shelf 1</answer>",
        "react_tags": (
            "Example:\n"
            "<thought>The task says to find a book. Shelves are likely places "
            "to look.</thought>\n"
            "<action>go to shelf 1</action>"
        ),
        "react_classic": (
            "Example:\n"
            "Thought: The task says to find a book. Shelves are likely places "
            "to look.\n"
            "Action: go to shelf 1"
        ),
    },
    # -- jericho --
    "jericho": {
        "answer_tags": "Example:\n<answer>open mailbox</answer>",
        "react_tags": (
            "Example:\n"
            "<thought>There's a mailbox here. I should check if it contains "
            "anything useful.</thought>\n"
            "<action>open mailbox</action>"
        ),
        "react_classic": (
            "Example:\n"
            "Thought: There's a mailbox here. I should check if it contains "
            "anything useful.\n"
            "Action: open mailbox"
        ),
    },
    # -- webshop --
    "webshop": {
        "answer_tags": "Example:\n<answer>search[red shoes size 10]</answer>",
        "react_tags": (
            "Example:\n"
            "<thought>I need to find red shoes in size 10 matching the "
            "instruction.</thought>\n"
            "<action>search[red shoes size 10]</action>"
        ),
        "react_classic": (
            "Example:\n"
            "Thought: I need to find red shoes in size 10 matching the "
            "instruction.\n"
            "Action: search[red shoes size 10]"
        ),
    },
    # -- harbor --
    "harbor": {
        "answer_tags": (
            "Single-line example:\n"
            "<answer>ls -la /tmp</answer>\n\n"
            "Multi-line example:\n"
            "<answer>cat > /app/script.py << 'EOF'\n"
            "import sys\n"
            "print(\"hello\")\n"
            "EOF</answer>\n\n"
            "Submit example (when your work is complete):\n"
            "<answer>SUBMIT</answer>"
        ),
        "react_tags": (
            "Single-line example:\n"
            "<thought>I should inspect the working directory before changing "
            "files.</thought>\n"
            "<action>ls -la /app</action>\n\n"
            "Multi-line example:\n"
            "<thought>I need to write a small Python script into the task "
            "workspace.</thought>\n"
            "<action>cat > /app/script.py <<'EOF'\n"
            "import sys\n"
            "print(\"hello\")\n"
            "EOF</action>\n\n"
            "Submit example (when your work is complete):\n"
            "<thought>All tests pass and files are in place. Time to "
            "submit.</thought>\n"
            "<action>SUBMIT</action>"
        ),
        "react_classic": (
            "Single-line example:\n"
            "Thought: I should inspect the task directory before editing files.\n"
            "Action: ls -la /app\n\n"
            "Multi-line example:\n"
            "Thought: I need to write a small Python script into the task "
            "workspace.\n"
            "Action: cat > /app/script.py <<'EOF'\n"
            "import sys\n"
            "print(\"hello\")\n"
            "EOF\n\n"
            "Submit example (when your work is complete):\n"
            "Thought: All tests pass and files are in place. Time to submit.\n"
            "Action: SUBMIT"
        ),
    },
    # -- open_apps --
    "open_apps": {
        "answer_tags": (
            "Examples:\n"
            "<answer>click('23')</answer>\n\n"
            "<answer>fill('15', 'Call Mom')</answer>\n\n"
            "<answer>scroll(0, 300)</answer>"
        ),
        "react_tags": (
            "Example:\n"
            "<thought>I need to click on the OpenTodos link to navigate to "
            "the todo app.</thought>\n"
            "<action>click('23')</action>"
        ),
        "react_classic": (
            "Example:\n"
            "Thought: I need to click on the OpenTodos link to navigate to "
            "the todo app.\n"
            "Action: click('23')"
        ),
    },
    # -- craftax --
    "craftax": {
        "answer_tags": "Example:\n<answer>left</answer>",
        "react_tags": (
            "Example:\n"
            "<thought>There's a tree to my left that I can chop for "
            "wood.</thought>\n"
            "<action>left</action>"
        ),
        "react_classic": (
            "Example:\n"
            "Thought: There's a tree to my left that I can chop for wood.\n"
            "Action: left"
        ),
    },
}

RANKING_SCHEME_EXAMPLES: dict[str, dict[str, str]] = {
    "reasoning_gym": {
        "answer_tags": SCHEME_EXAMPLES["reasoning_gym"]["answer_tags"],
        "react_tags": "Example:\n<action>42</action>",
        "react_classic": "Example:\nAction: 42",
    },
    "alfworld": {
        "answer_tags": SCHEME_EXAMPLES["alfworld"]["answer_tags"],
        "react_tags": "Example:\n<action>go to shelf 1</action>",
        "react_classic": "Example:\nAction: go to shelf 1",
    },
    "jericho": {
        "answer_tags": SCHEME_EXAMPLES["jericho"]["answer_tags"],
        "react_tags": "Example:\n<action>open mailbox</action>",
        "react_classic": "Example:\nAction: open mailbox",
    },
    "webshop": {
        "answer_tags": SCHEME_EXAMPLES["webshop"]["answer_tags"],
        "react_tags": "Example:\n<action>search[red shoes size 10]</action>",
        "react_classic": "Example:\nAction: search[red shoes size 10]",
    },
    "harbor": {
        "answer_tags": SCHEME_EXAMPLES["harbor"]["answer_tags"],
        "react_tags": (
            "Single-line example:\n"
            "<action>ls -la /app</action>\n\n"
            "Multi-line example:\n"
            "<action>cat > /app/script.py <<'EOF'\n"
            "import sys\n"
            "print(\"hello\")\n"
            "EOF</action>\n\n"
            "Submit example (when your work is complete):\n"
            "<action>SUBMIT</action>"
        ),
        "react_classic": (
            "Single-line example:\n"
            "Action: ls -la /app\n\n"
            "Multi-line example:\n"
            "Action: cat > /app/script.py <<'EOF'\n"
            "import sys\n"
            "print(\"hello\")\n"
            "EOF\n\n"
            "Submit example (when your work is complete):\n"
            "Action: SUBMIT"
        ),
    },
    "craftax": {
        "answer_tags": SCHEME_EXAMPLES["craftax"]["answer_tags"],
        "react_tags": "Example:\n<action>left</action>",
        "react_classic": "Example:\nAction: left",
    },
}


_DEFAULT_COMMAND_TIMEOUT_SEC = 60

_MAX_STEPS_CLAUSE_GENERIC = (
    "You have a limited number of steps, so be efficient — plan ahead "
    "and avoid unnecessary commands."
)


def _format_timeout_desc(seconds: int) -> str:
    """Human-readable timeout description, e.g. ``'60 seconds'`` or ``'2 minutes'``."""
    if seconds >= 60 and seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{seconds} seconds"


def _format_max_steps_clause(max_steps: int | None) -> str:
    if max_steps is None:
        return _MAX_STEPS_CLAUSE_GENERIC
    return (
        f"You have a maximum of {int(max_steps)} turns to complete the "
        "task. If you do not complete it within those turns, you fail. "
        "Be efficient — plan ahead and avoid unnecessary commands."
    )


def _resolve_env_prompt(
    adapter: str,
    *,
    env_name: str | None = None,
    make_kwargs: dict[str, Any] | None = None,
) -> str:
    """Resolve the environment-specific prompt layer."""
    if adapter == "jericho" and env_name is not None:
        game_name = env_name.split(":", 1)[1] if ":" in env_name else env_name
        from qval.jericho_prompts import build_jericho_env_prompt

        env_prompt = build_jericho_env_prompt(game_name)
    else:
        env_prompt = ENV_PROMPTS[adapter]

    format_kwargs: dict[str, str] = {}
    if "{command_timeout_desc}" in env_prompt:
        timeout_sec = (
            (make_kwargs or {}).get("command_soft_timeout")
            or _DEFAULT_COMMAND_TIMEOUT_SEC
        )
        format_kwargs["command_timeout_desc"] = _format_timeout_desc(int(timeout_sec))
    if "{max_steps_clause}" in env_prompt:
        format_kwargs["max_steps_clause"] = _format_max_steps_clause(
            (make_kwargs or {}).get("max_steps")
        )
    if format_kwargs:
        env_prompt = env_prompt.format(**format_kwargs)
    return env_prompt


def _compose_system_prompt(
    adapter: str,
    scheme: str,
    *,
    env_name: str | None = None,
    make_kwargs: dict[str, Any] | None = None,
    scheme_instructions: dict[str, str],
    scheme_examples: dict[str, dict[str, str]],
) -> str:
    """Compose an environment prompt, scheme instructions, and examples."""
    env_prompt = _resolve_env_prompt(
        adapter,
        env_name=env_name,
        make_kwargs=make_kwargs,
    )

    parts = [env_prompt, scheme_instructions[scheme]]
    examples = scheme_examples.get(adapter, {}).get(scheme)
    if examples:
        parts.append(examples)
    return "\n\n".join(parts)


def build_actor_system_prompt(
    adapter: str,
    scheme: str = "answer_tags",
    *,
    env_name: str | None = None,
    make_kwargs: dict[str, Any] | None = None,
) -> str:
    """Compose the three-layer actor system prompt."""
    return _compose_system_prompt(
        adapter,
        scheme,
        env_name=env_name,
        make_kwargs=make_kwargs,
        scheme_instructions=SCHEME_INSTRUCTIONS,
        scheme_examples=SCHEME_EXAMPLES,
    )


def build_ranking_sampling_system_prompt(
    adapter: str,
    scheme: str = "answer_tags",
    *,
    env_name: str | None = None,
    make_kwargs: dict[str, Any] | None = None,
) -> str:
    """Compose the ranking-sampling system prompt.

    Ranking sampling should request a single executable action, not ReAct
    reasoning. This preserves the environment-specific role prompt while
    placing the formatting instructions last so they are closest to the
    model's generation point.

    Order: env prompt → example → formatting instructions.
    """
    env_prompt = _resolve_env_prompt(
        adapter, env_name=env_name, make_kwargs=make_kwargs,
    )
    parts = [env_prompt]
    examples = RANKING_SCHEME_EXAMPLES.get(adapter, {}).get(scheme)
    if examples:
        parts.append(examples)
    parts.append(RANKING_SCHEME_INSTRUCTIONS[scheme])
    return "\n\n".join(parts)
