"""Scripted expert policy for ALFWorld GT rollouts.

Reads the next handcoded command off a line the
``expose_expert_plan_in_obs=True`` AlfWorldAdapter prepends to the state
text.  Deterministic given the env state, so ``num_rollouts=1`` gives
exact optimal V / Q (modulo env stochasticity, which ALFWorld has none
of for the deterministic TextWorld backend).
"""

from __future__ import annotations

import re
from typing import Any

from qval.optimal_policies import register_policy

# Must match the marker injected by AlfWorldAdapter when
# expose_expert_plan_in_obs=True.  See llenvs/adapters/alfworld.py.
_EXPERT_PLAN_RE = re.compile(r"\[expert_plan_next:\s*(.+?)\]")

# Fallback command when the planner couldn't produce a plan for the
# current state (rare — usually means the env is already in a terminal
# or unreachable state).  ``look`` is a safe no-op that advances the
# step counter without changing the env.
_FALLBACK_COMMAND = "look"


def _alfworld_expert_fn(state: str, metadata: dict[str, Any]) -> str:
    del metadata
    match = _EXPERT_PLAN_RE.search(state)
    if match:
        return match.group(1).strip()
    return _FALLBACK_COMMAND


register_policy(
    "alfworld_expert",
    _alfworld_expert_fn,
    metadata={
        # No fixed action space for ALFWorld (admissible commands vary
        # per state).  The stochastic wrapper can't be used without an
        # ``actions`` callable, which isn't meaningful here — use the
        # deterministic policy directly.
    },
)
