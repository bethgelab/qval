import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the OpenApps environment.
    The value represents the expected discounted reward, which is binary (1.0 on goal).
    Since the reward is outcome-based and optimal play is assumed, V(s) is roughly
    the probability of success discounted by the remaining distance to the goal.
    """
    state_lower = state.lower()

    # 1. Success Detection (V = 1.0)
    # These keywords strongly indicate the goal has been achieved.
    success_markers = [
        "successfully", "task completed", "has been created",
        "has been sent", "has been added", "saved successfully",
        "confirmation", "completed", "done", "success"
    ]
    if any(marker in state_lower for marker in success_markers):
        return 1.0

    # 2. Goal Extraction
    # Attempt to isolate the goal from the state text to check for keyword overlap.
    goal = ""
    goal_match = re.search(r"(?:Goal|Task):\s*(.*?)(?:\n|Observation:|$)", state, re.IGNORECASE)
    if goal_match:
        goal = goal_match.group(1).lower()

    # 3. Value Estimation based on progress heuristics
    # V(s) starts at a base value for being in the environment.
    value = 0.3

    # Heuristic A: Goal Keyword Overlap
    # If the current page contains specific words from the goal, the agent is likely 
    # on the correct page.
    if goal:
        stop_words = {'a', 'an', 'the', 'to', 'in', 'at', 'of', 'and', 'is', 'it', 'please', 'add', 'create', 'send'}
        goal_words = [w for w in re.findall(r'\w+', goal) if w not in stop_words]
        for word in goal_words:
            if word in state_lower:
                value += 0.05

    # Heuristic B: Form Filling
    # BrowserGym accessibility trees often represent filled fields as value="text".
    # Finding non-empty values is a strong indicator of progress.
    filled_fields = re.findall(r'value=["\']([^"\']+)["\']', state)
    filled_count = len([v for v in filled_fields if v.strip()])
    value += filled_count * 0.1

    # Heuristic C: Proximity to Submission
    # The presence of action buttons (Submit, Save, Send) suggests the agent is 
    # at the final stage of the task.
    action_markers = ['submit', 'save', 'send', 'confirm', 'create', 'add', 'ok']
    if any(marker in state_lower for marker in action_markers):
        value += 0.2

    # 4. Penalty for Errors
    # Error messages indicate a setback, reducing the expected value.
    error_markers = ['error', 'invalid', 'required', 'incorrect', 'failed', 'wrong', 'missing']
    if any(marker in state_lower for marker in error_markers):
        value -= 0.3

    # 5. Final Normalization
    # The value is clamped between 0.0 and 0.98. 
    # 1.0 is reserved for confirmed success.
    return max(0.0, min(0.98, value))