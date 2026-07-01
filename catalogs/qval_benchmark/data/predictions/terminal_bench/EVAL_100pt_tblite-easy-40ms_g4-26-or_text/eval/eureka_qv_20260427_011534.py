import re

def signal_function(state: str, action: str, next_state: str):
    """
    Estimates the Q-value for taking 'action' in 'state' based on the 'next_state' outcome.
    Provides a dense signal with multiple components to reward progress, verification,
    and substantial output while penalizing errors, useless actions, and stagnation.
    """
    # 1. Pattern Definitions
    success_patterns = [
        r"passed", r"success", r"\[ok\]", r"0 errors", 
        r"check passed", r"completed", r"all tests passed"
    ]
    failure_patterns = [
        r"failed", r"error", r"not found", r"denied", 
        r"syntax error", r"segmentation fault", r"command not found", 
        r"no such file", r"permission denied"
    ]
    verifying_patterns = [
        r"test", r"verify", r"check", r"validate", 
        r"pytest", r"unittest", r"submit", r"run", r"execute"
    ]
    productive_patterns = [
        r"install", r"build", r"make", r"gcc", r"g\+\+", 
        r"python", r"cmake", r"mkdir", r"pip", r"chmod", 
        r"cp", r"mv", r"rm", r"nano", r"vi", r"vim", 
        r"cat", r"grep", r"find", r"apt", r"apt-get"
    ]
    useless_patterns = [
        r"^\s*ls(\s+.*)?$", r"^\s*pwd$", r"^\s*cd(\s+.*)?$"
    ]

    # 2. Immediate Terminal State Detection
    is_verifying = any(re.search(p, action, re.IGNORECASE) for p in verifying_patterns)
    is_success = any(re.search(p, next_state, re.IGNORECASE) for p in success_patterns)
    is_error = any(re.search(p, next_state, re.IGNORECASE) for p in failure_patterns)

    if is_verifying:
        if is_success:
            return 1.0, {"success_reward": 1.0}
        if is_error:
            return 0.0, {"failure_penalty": 0.0}

    # 3. Feature Detection for Non-Terminal Transitions
    is_productive = any(re.search(p, action, re.IGNORECASE) for p in productive_patterns)
    is_useless = any(re.search(p, action, re.IGNORECASE) for p in useless_patterns)
    is_stagnant = next_state.strip() == state.strip()
    # Substantial output if word count increases significantly
    is_substantial = len(next_state.split()) > len(state.split()) + 1

    # 4. Component-based Estimation
    # We use a base value and add/subtract bonuses/penalties.
    base_val = 0.4
    bonus_productive = 0.2 if is_productive else 0.0
    bonus_verifying = 0.2 if is_verifying else 0.0
    bonus_substantial = 0.1 if is_substantial else 0.0
    penalty_error = -0.4 if is_error else 0.0
    penalty_useless = -0.2 if is_useless else 0.0
    penalty_stagnant = -0.1 if is_stagnant else 0.0

    # 5. Aggregate and Clamp
    # Calculate raw sum and clamp to [0.0, 1.0]
    raw_total = (base_val + bonus_productive + bonus_verifying + bonus_substantial + 
                 penalty_error + penalty_useless + penalty_stagnant)
    clamped_total = max(0.0, min(1.0, raw_total))

    # Adjust base_val so the sum of components matches the clamped total exactly.
    # This ensures the reward signal remains consistent with the components provided.
    diff = clamped_total - raw_total
    adjusted_base = base_val + diff

    return clamped_total, {
        "base_val": adjusted_base,
        "bonus_productive": bonus_productive,
        "bonus_verifying": bonus_verifying,
        "bonus_substantial": bonus_substantial,
        "penalty_error": penalty_error,
        "penalty_useless": penalty_useless,
        "penalty_stagnant": penalty_stagnant
    }