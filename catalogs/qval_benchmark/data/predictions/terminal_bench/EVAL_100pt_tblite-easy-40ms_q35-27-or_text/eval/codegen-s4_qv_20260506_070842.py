def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value estimate
    q_value = 0.0
    
    # Check for success indicators in next_state
    success_patterns = ['success', 'done', 'complete', 'passed', 'ok', 'finished', '✓', '✔', 'done', 'verified', 'correct']
    for pattern in success_patterns:
        if pattern.lower() in next_state.lower():
            q_value += 0.6
            break
    
    # Check for error indicators (negative signal)
    error_patterns = ['error', 'fail', 'exception', 'traceback', 'failed', 'stderr', 'permission denied', 'not found', 'syntax error']
    for pattern in error_patterns:
        if pattern.lower() in next_state.lower():
            q_value -= 0.4
            break
    
    # Check for progress indicators (percentages, fractions)
    progress_matches = re.findall(r'(\d+)%', next_state)
    if progress_matches:
        progress = float(max(progress_matches)) / 100.0
        q_value += progress * 0.3
    
    # Check for goal-related content
    goal_keywords = ['result', 'output', 'answer', 'solution', 'final', 'expected', 'match', 'equals']
    for keyword in goal_keywords:
        if keyword.lower() in next_state.lower():
            q_value += 0.15
    
    # Check if action appears productive (not empty or just navigation)
    action_lower = action.lower().strip()
    if action_lower:
        # Penalize purely navigational actions
        if action_lower.startswith(('cd ', 'cd\t', 'ls ', 'ls\t', 'pwd', 'cat ')) and len(action_lower) < 20:
            q_value += 0.05
        else:
            q_value += 0.1
    
    # Check for file creation/modification indicators
    if any(ind in next_state.lower() for ind in ['created', 'wrote', 'saved', 'generated', 'built', 'compiled']):
        q_value += 0.2
    
    # Check for completion markers
    completion_markers = ['---', '===', 'END', 'FINISHED', 'DONE', 'COMPLETE']
    if any(marker in next_state for marker in completion_markers):
        q_value += 0.2
    
    # Check for numerical results (often indicate successful computation)
    if re.search(r'[0-9]+\.[0-9]+|[0-9]+', next_state) and len(next_state) > 50:
        q_value += 0.1
    
    # Check if next_state is significantly different from state (indicates progress)
    if len(next_state) > len(state) * 1.5:
        q_value += 0.1
    
    # Clamp Q-value to reasonable range [-1.0, 1.0]
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value