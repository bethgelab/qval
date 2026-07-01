def signal_function(state: str, action: str, next_state: str) -> float:
    import math

    # Heuristic keywords for evaluating task progress and status
    success_keywords = ['pass', 'success', 'verified', 'correct', 'done', 'complete', 'accepted', 'solution', 'test passed']
    error_keywords = ['error', 'fail', 'exception', 'not found', 'permission denied', 'timeout', 'crash', 'undefined', 'fatal']
    progress_keywords = ['installing', 'building', 'processing', 'running', 'cloning', 'downloading', 'loading', 'generating']

    # Normalize strings for case-insensitive matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()

    # Initialize base Q-value
    score = 0.5

    # 1. Check for Completion (Highest Priority)
    # If the next state indicates the task is finished, Q-value is very high.
    for kw in success_keywords:
        if kw in next_state_lower:
            score = 0.98
            break

    # 2. Check for Errors (Low Priority if no success detected)
    # Errors significantly reduce the expected return.
    if score < 0.9:
        for kw in error_keywords:
            if kw in next_state_lower:
                score = 0.05
                break

    # 3. Check for Progress (Intermediate Priority)
    # Active processes suggest forward movement.
    if score < 0.9 and score > 0.2:
        for kw in progress_keywords:
            if kw in next_state_lower:
                score = 0.6
                break

    # 4. Check for Stagnation
    # If action was non-trivial but next_state is empty/whitespace, progress is low.
    if len(next_state.strip()) < 5 and len(action.strip()) > 2:
        if score > 0.5:
            score = 0.3

    # 5. Check for Premature Termination
    # Exiting before success is a negative signal.
    if 'exit' in action_lower or 'quit' in action_lower:
        if score < 0.9:
            score = 0.1

    # Clamp the score to valid probability range [0.0, 1.0]
    return max(0.0, min(1.0, score))