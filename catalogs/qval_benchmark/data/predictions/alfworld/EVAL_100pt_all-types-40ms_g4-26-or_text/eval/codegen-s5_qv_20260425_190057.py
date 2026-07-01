import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    The Q-value represents the expected reward (1.0 for success, 0.0 otherwise).
    """
    # 1. Immediate Success Check
    # If the next state indicates the task is finished, the Q-value is 1.0.
    success_indicators = ["success", "completed", "finished", "task is complete"]
    next_state_lower = next_state.lower()
    if any(ind in next_state_lower for ind in success_indicators):
        return 1.0

    # 2. Goal Extraction
    # ALFWorld typically describes the goal as "Your task is to [goal]."
    goal_match = re.search(r"task is to (.*?)(?:\.|$)", state, re.IGNORECASE)
    if not goal_match:
        return 0.0
    goal_text = goal_match.group(1).lower()

    # 3. Keyword Extraction Helper
    def get_keywords(text: str) -> set:
        # Extract alphanumeric words of length 3 or more
        words = re.findall(r'\b[a-z0-9]{3,}\b', text.lower())
        # Define common stop words to ignore
        stop_words = {
            'the', 'put', 'in', 'on', 'at', 'to', 'with', 'your', 'task', 
            'move', 'find', 'place', 'clean', 'is', 'and', 'get', 'into',
            'you', 'are', 'has', 'have', 'can', 'there', 'this', 'that', 
            'from', 'for', 'not', 'all', 'some', 'one', 'two'
        }
        return {w for w in words if w not in stop_words}

    goal_keywords = get_keywords(goal_text)
    if not goal_keywords:
        return 0.0
        
    action_keywords = get_keywords(action)
    state_keywords = get_keywords(state)
    next_state_keywords = get_keywords(next_state_lower)

    # 4. Scoring Logic (Heuristic Progress Estimation)
    score = 0.0

    # A. Action-Goal Alignment
    # If the action involves keywords directly mentioned in the goal (e.g., "take apple"),
    # it's a strong indicator of a potential high-value action.
    action_goal_overlap = goal_keywords.intersection(action_keywords)
    if action_goal_overlap:
        score += 0.3
        # If the action was taken and the targeted keyword is now present in the next state 
        # (implying the action was successful, like "take apple" -> "holding apple"), 
        # we add a significant bonus.
        if action_goal_overlap.intersection(next_state_keywords):
            score += 0.4

    # B. State-Goal Progress (New Information)
    # If the next state contains goal-related keywords that were not present in the 
    # previous state, this indicates progress (e.g., discovering a location or picking up an object).
    new_goal_keywords = goal_keywords.intersection(next_state_keywords) - state_keywords
    if new_goal_keywords:
        # Scale the bonus by the proportion of goal keywords newly discovered.
        score += 0.3 * (len(new_goal_keywords) / len(goal_keywords))

    # C. Status Markers
    # Check for common indicators of object-state changes or possession.
    if "holding" in next_state_lower or "you have" in next_state_lower:
        # If the agent is now holding a goal keyword, increase score.
        if any(kw in next_state_keywords for kw in goal_keywords):
            score += 0.1

    # 5. Return normalized Q-value
    # Clip the score to the range [0.0, 0.99], as 1.0 is reserved for terminal success.
    return min(max(score, 0.0), 0.99)