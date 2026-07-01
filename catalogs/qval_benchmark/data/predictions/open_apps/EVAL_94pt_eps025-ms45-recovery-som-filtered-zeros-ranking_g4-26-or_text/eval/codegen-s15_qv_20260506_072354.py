import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    The estimation is based on identifying signs of success, failure, and progress 
    within the accessibility tree text representation.
    """
    ns_lower = next_state.lower()
    a_lower = action.lower()
    s_lower = state.lower()

    # 1. Success Check (Terminal Reward)
    # If the next state contains specific language indicating task completion,
    # we assign a high Q-value reflecting the expected reward of 1.0.
    success_indicators = [
        "success", "sent", "added", "created", "saved", 
        "completed", "confirmed", "scheduled", "done"
    ]
    if any(indicator in ns_lower for indicator in success_indicators):
        return 1.0

    # 2. Failure Check
    # If the next state indicates an error, the expected cumulative reward is 0.0.
    error_indicators = [
        "error", "failed", "invalid", "not found", 
        "missing", "incorrect", "not allowed", "required"
    ]
    if any(error in ns_lower for error in error_indicators):
        return 0.0

    # 3. Stagnation or No-op Check
    # If no state change occurs or a noop action is taken, reward potential is minimal.
    if state == next_state or "noop" in a_lower:
        return 0.01

    # 4. Action-Specific Progress Heuristics
    
    # A. Handle 'fill' action
    # Regex matches patterns like fill('1', 'text') or fill(1, 'text')
    fill_match = re.search(r"fill\(\s*['\"]?(\d+)['\"]?\s*,\s*['\"](.*?)['\"]\s*\)", action)
    if fill_match:
        target_text = fill_match.group(2).lower()
        # If the text entered is visible in the next state, it's a productive step.
        if target_text and target_text in ns_lower:
            return 0.8
        else:
            # If text is not in next_state, it might have been submitted or failed.
            return 0.4

    # B. Handle 'click' action
    if "click" in a_lower:
        # Significant change in accessibility tree length often indicates a page transition.
        len_diff = abs(len(next_state) - len(state))
        if len_diff > 150:
            return 0.75
        
        # Check if the click target or context implies a submission/action.
        action_triggers = ["submit", "send", "save", "add", "delete", "create", "confirm", "ok", "apply"]
        if any(trigger in a_lower for trigger in action_triggers):
            return 0.65
        
        # If it caused any state change, it's likely progress (e.g., opening a menu).
        if next_state != state:
            return 0.5
        return 0.1

    # C. Handle 'press' action
    if "press" in a_lower:
        if "enter" in a_lower:
            return 0.6  # Enter is often used for form submission.
        return 0.4

    # D. Handle 'scroll' action
    if "scroll" in a_lower:
        return 0.4 if next_state != state else 0.1

    # 5. General Progress Fallback
    # If the state changed and it's not a known error or stagnation,
    # we assume some level of incremental progress.
    if next_state != state:
        return 0.3
    
    return 0.05