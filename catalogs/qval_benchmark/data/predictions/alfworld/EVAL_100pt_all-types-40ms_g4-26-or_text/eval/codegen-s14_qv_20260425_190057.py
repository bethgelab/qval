import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value of an action in ALFWorld based on state transitions, 
    goal progress, and environment feedback.
    """
    s_low = state.lower()
    a_low = action.lower()
    n_low = next_state.lower()

    # 1. Check for immediate terminal success
    if any(x in n_low for x in ["task completed", "success", "goal achieved", "you have achieved"]):
        return 1.0

    # 2. Check for action failure or no change in the environment
    if n_low.strip() == s_low.strip():
        return 0.0
    if any(x in n_low for x in ["you can't", "nothing happens", "is not here", "not possible", "is not a valid"]):
        return 0.0

    q_val = 0.0

    # 3. Extract Goal from the state observation
    goal_words = set()
    goal_match = re.search(r"(?:goal|task|objective):\s*(.*?)(?:\.|\n|$)", s_low)
    if goal_match:
        goal_text = goal_match.group(1)
        goal_words = set(re.findall(r'\w+', goal_text))
        # Filter out common stop words to isolate key entities (objects/locations)
        stop_words = {'the', 'a', 'an', 'in', 'on', 'to', 'of', 'at', 'is', 'put', 'take', 'clean', 'move', 'with', 'into', 'your', 'achieve', 'target'}
        goal_words = goal_words - stop_words

    # 4. Scoring based on Goal alignment
    if goal_words:
        # A. Action alignment: Does the action mention parts of the goal?
        action_words = set(re.findall(r'\w+', a_low))
        matches = action_words.intersection(goal_words)
        if matches:
            q_val += 0.15 * len(matches)

        # B. Progress detection: Does the number of goal-related keywords increase?
        s_goal_count = 0
        n_goal_count = 0
        for w in goal_words:
            if w in s_low:
                s_goal_count += 1
            if w in n_low:
                n_goal_count += 1
        
        if n_goal_count > s_goal_count:
            # Significant progress made (e.g., an object is now mentioned in the new context)
            q_val += 0.4 * (n_goal_count - s_goal_count)
        elif n_goal_count == s_goal_count and n_goal_count > 0:
            # If we already have the goal words in view, reward goal-oriented verbs
            goal_verbs = {'put', 'take', 'clean', 'grab', 'place', 'wash', 'go', 'move', 'open', 'close', 'pick'}
            if any(v in a_low for v in goal_verbs):
                q_val += 0.1

    # 5. Heuristic-based state transition detection (independent of explicit goal parsing)
    # This acts as a fallback for progress detection.
    if any(x in a_low for x in ["take", "grab", "pick"]):
        if "holding" in n_low:
            q_val += 0.3
    elif any(x in a_low for x in ["put", "place", "drop"]):
        # Detecting if an object has been moved to a container or surface
        if re.search(r"is (in|on|inside|at)", n_low):
            q_val += 0.3
    elif any(x in a_low for x in ["clean", "wash"]):
        if "clean" in n_low:
            q_val += 0.3
    elif any(x in a_low for x in ["go", "move", "walk"]):
        # Detect movement by checking if the first line (often the room description) changed
        s_lines = s_low.split('\n')
        n_lines = n_low.split('\n')
        if s_lines and n_lines and s_lines[0].strip() != n_lines[0].strip():
            q_val += 0.1

    # Return the estimated value, clamped between 0.0 and 1.0
    return min(max(q_val, 0.0), 1.0)