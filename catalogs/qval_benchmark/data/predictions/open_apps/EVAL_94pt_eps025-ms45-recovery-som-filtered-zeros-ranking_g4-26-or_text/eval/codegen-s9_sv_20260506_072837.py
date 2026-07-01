import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for an OpenApps environment state.
    The state-value represents the probability that the current state leads to 
    the goal being achieved within the remaining steps.
    """
    # Convert to lower case once for efficient searching
    s = state.lower()

    # 1. Immediate Success Detection (Goal is achieved)
    # We look for confirmation patterns commonly found in synthetic web apps.
    # Since reward is binary, if the state shows the action is completed, V(s) = 1.0.
    success_patterns = [
        r"success", r"completed", r"sent", r"saved", r"added",
        r"created", r"done", r"finished", r"confirmed", r"delivered"
    ]
    
    # Check for success, but exclude cases where it might be part of an error message.
    if any(re.search(p, s) for p in success_patterns):
        # If an error is present nearby, it's likely a "task failed" message.
        if not re.search(r"error|fail|invalid|not found|cannot|incorrect", s):
            return 1.0

    # 2. Impasse/Error Detection (Goal is likely impossible from here)
    # If the state explicitly shows an error, the value drops significantly.
    if re.search(r"error|invalid|failed|not found|cannot|incorrect|wrong|required", s):
        return 0.0

    # 3. Heuristic Progress Estimation
    # We build a score based on how "actionable" or "filled-out" the state is.
    # An empty form is less valuable than a partially filled form.
    # A form with a 'Submit' button is more valuable than just an input field.
    
    score = 0.1  # Base score for being in a valid, non-error state

    # Identify presence of core interactive UI components
    has_input = bool(re.search(r'input|role="textbox"', s))
    has_button = bool(re.search(r'button|role="button"', s))
    has_link = bool(re.search(r'link|role="link"', s))
    
    # Check for "progression" keywords (Submit/Save/Send buttons)
    progression_keywords = ['send', 'save', 'add', 'submit', 'ok', 'confirm', 'create', 'apply', 'go']
    has_progression_trigger = any(kw in s for kw in progression_keywords)

    # Check for "work done" (filled input fields)
    # Accessibility trees often use value="..." for inputs.
    filled_values = re.findall(r'value="([^"]+)"', s)
    meaningful_fills = 0
    for val in filled_values:
        v = val.strip().lower()
        # Filter out empty or non-semantic default values
        if v and v not in ("none", "null", "false", "undefined", "true", "0"):
            meaningful_fills += 1

    # Scoring Logic:
    if has_input:
        score += 0.2
    if has_button:
        score += 0.1
    if has_link:
        score += 0.05
    if has_progression_trigger:
        score += 0.2
    
    # Bonus for actually having data in the form
    if meaningful_fills > 0:
        # Cap the fill bonus to prevent extreme values
        score += 0.3 * min(meaningful_fills, 2)

    # Synergy bonuses (e.g., having both data and a way to submit it)
    if meaningful_fills > 0 and has_progression_trigger:
        score += 0.15
    if has_input and has_progression_trigger:
        score += 0.1

    # 4. Goal Contextualization (if goal info is embedded in the state string)
    # In many RL environments, the task description is prepended to the observation.
    goal_match = re.search(r'(?:goal|task|instruction):\s*(.*)', s)
    if goal_match:
        goal_text = goal_match.group(1)
        # Extract potential keywords from the goal (words > 3 chars)
        goal_keywords = re.findall(r'\b\w{4,}\b', goal_text)
        if goal_keywords:
            matches = sum(1 for kw in goal_keywords if kw in s)
            # Boost score based on how much of the goal's vocabulary is visible in the UI
            score += (matches / len(goal_keywords)) * 0.4

    # 5. Final Clamp
    # The value should never exceed 1.0 and should be bounded by 0.0.
    # We cap at 0.95 for non-terminal states to allow for uncertainty.
    return max(0.0, min(0.95, score))