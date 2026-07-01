import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal-based 
    system administration/programming task. The reward is binary (1.0 for success, 
    0.0 otherwise) and provided at the end of the episode.
    """
    # Indicators of success or positive progress in the output (next_state)
    success_keywords = [
        r"successfully", r"completed", r"done", r"correct", r"pass", 
        r"created", r"updated", r"fixed", r"verified", r"accepted"
    ]
    
    # Indicators of failure or errors in the output (next_state)
    failure_keywords = [
        r"error", r"failed", r"denied", r"not found", r"invalid", 
        r"syntaxerror", r"exception", r"incorrect", r"refused", r"cannot"
    ]
    
    # Actions that typically signal submission or verification of the goal
    completion_actions = [
        r"submit", r"verify", r"test", r"check", r"done"
    ]

    score = 0.5  # Neutral starting point for the probability of success

    # 1. Analyze the output (next_state) for success indicators
    next_state_lower = next_state.lower()
    for kw in success_keywords:
        if re.search(kw, next_state_lower):
            score += 0.15
            break
            
    # 2. Analyze the output (next_state) for failure indicators
    for kw in failure_keywords:
        if re.search(kw, next_state_lower):
            score -= 0.25
            break

    # 3. Analyze the action taken
    action_lower = action.lower()
    for kw in completion_actions:
        if re.search(kw, action_lower):
            # If the agent is attempting to verify/submit and the output 
            # doesn't contain errors, it's a high-value state.
            if not any(re.search(fk, next_state_lower) for fk in failure_keywords):
                score += 0.2
            else:
                score -= 0.2
            break

    # 4. Meaningful state change
    # If the next_state is identical to the state (excluding prompt changes), 
    # it's likely a redundant or ineffective action.
    if state.strip() == next_state.strip():
        score -= 0.1

    # 5. Tool usage bonus
    # Using productive tools often correlates with progress toward a goal.
    productive_tools = [r"grep", r"sed", r"awk", r"vim", r"nano", r"gcc", r"python", r"pip", r"apt", r"chmod", r"chown"]
    for tool in productive_tools:
        if re.search(rf"\b{tool}\b", action_lower):
            score += 0.05
            break

    # Final clipping to ensure the value is within [0.0, 1.0]
    return max(0.0, min(1.0, score))