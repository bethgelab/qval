import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the OpenApps environment.
    V(s) represents the expected discounted reward, approximated here as the 
    probability of reaching the goal from the current state.
    """
    state_lower = state.lower()
    
    # 1. Attempt to extract the goal from the state string
    # Common patterns: "Goal: ...", "Task: ..."
    goal_match = re.search(r"(?:goal|task):\s*(.*?)(?=observation:|current state:|$)", state_lower, re.DOTALL)
    goal_text = goal_match.group(1) if goal_match else ""
    
    # 2. Check for definitive success markers
    # These suggest the target state has been reached.
    success_markers = [
        'successfully', 'completed', 'created', 'sent', 'added', 
        'saved', 'confirmed', 'was successful', 'done', 'finished'
    ]
    if any(marker in state_lower for marker in success_markers):
        # Verify that the success is not part of an error message
        if 'fail' not in state_lower and 'error' not in state_lower:
            return 1.0

    # 3. Check for failure/error markers
    # These indicate a setback or a required action that was missed.
    error_markers = ['error', 'invalid', 'required', 'fail', 'wrong', 'incorrect', 'cannot']
    has_error = any(marker in state_lower for marker in error_markers)
    
    # 4. Identify final action buttons
    # Presence of a submission button often indicates proximity to the goal.
    final_actions = [
        'submit', 'save', 'send', 'create', 'add', 'confirm', 
        'ok', 'publish', 'delete', 'remove', 'update'
    ]
    # We look for these specifically as interactive elements (buttons/links)
    has_final_action = any(marker in state_lower for marker in final_actions)
    
    # 5. Calculate a progress score based on goal-keyword overlap
    score = 0.1  # Base value for being in the environment
    if goal_text:
        # Simple stop-word filter to get meaningful keywords from the goal
        stop_words = {'the', 'and', 'a', 'an', 'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from'}
        goal_words = [w for w in re.findall(r'\w+', goal_text) if len(w) > 2 and w not in stop_words]
        
        if goal_words:
            matches = sum(1 for w in goal_words if w in state_lower)
            # Higher overlap indicates the agent is on the correct page/module
            overlap_ratio = matches / len(goal_words)
            score = 0.2 + (0.3 * overlap_ratio)
    
    # 6. Boost score if a final action is available
    # If the agent is on the right page AND has a button to finish, the value is high.
    if has_final_action:
        # Combine the general progress and the action availability
        score = max(score, 0.6)
        if goal_text and any(w in state_lower for w in re.findall(r'\w+', goal_text) if len(w) > 3):
            score = max(score, 0.8)

    # 7. Apply penalty for errors
    if has_error:
        score -= 0.2
        
    # Clamp the final value between 0.0 and 1.0
    return max(0.0, min(1.0, score))