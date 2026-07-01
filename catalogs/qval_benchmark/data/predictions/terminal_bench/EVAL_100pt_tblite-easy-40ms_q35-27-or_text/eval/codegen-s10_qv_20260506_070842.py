def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value starts at 0.5 (neutral)
    q_value = 0.5
    
    # Check for success indicators in next_state
    success_keywords = ['success', 'complete', 'done', 'passed', 'verified', 'ok', 'true', '✓', '✔', 'solved', 'correct', 'match', 'found']
    failure_keywords = ['error', 'fail', 'failed', 'exception', 'wrong', 'incorrect', 'false', '✗', '✘', 'permission denied', 'not found', 'cannot', 'refused', 'denied', 'timeout']
    
    next_state_lower = next_state.lower()
    
    # Boost for success indicators (weighted by strength)
    strong_success = ['success', 'passed', 'verified', 'solved', 'correct', 'match']
    weak_success = ['complete', 'done', 'ok', 'true', 'found']
    
    for keyword in strong_success:
        if keyword in next_state_lower:
            q_value += 0.35
            break
    for keyword in weak_success:
        if keyword in next_state_lower:
            q_value += 0.2
            break
    
    # Penalize for failure indicators
    for keyword in failure_keywords:
        if keyword in next_state_lower:
            q_value -= 0.3
            break
    
    # Check if action produced meaningful change (state transition)
    if state != next_state:
        q_value += 0.05
    
    # Check for goal-related patterns in state
    goal_patterns = ['goal', 'target', 'objective', 'task', 'complete', 'finish', 'end']
    for pattern in goal_patterns:
        if pattern in state.lower():
            q_value += 0.1
            break
    
    # Check if we're close to terminal (based on step count if visible)
    step_match = re.search(r'step[:\s]*(\d+)', state.lower())
    if step_match:
        step = int(step_match.group(1))
        if step > 35:  # Close to 40-step limit, penalize
            q_value -= 0.15
        elif step > 30:
            q_value -= 0.05
        elif step < 10:  # Early in episode, slight bonus for good progress
            q_value += 0.05
    
    # Check for file/task completion indicators
    completion_indicators = ['exists', 'created', 'written', 'saved', 'generated', 'output', 'result']
    for indicator in completion_indicators:
        if indicator in next_state_lower:
            q_value += 0.15
            break
    
    # Check for command success patterns
    if re.search(r'\b(exit code|return code)[:\s]*0\b', next_state_lower):
        q_value += 0.2
    
    # Penalize if next_state contains stack traces or long error output
    if 'traceback' in next_state_lower or 'stack trace' in next_state_lower:
        q_value -= 0.25
    
    # Check if action appears productive (non-empty, not just 'submit' without progress)
    if action and len(action.strip()) > 0:
        # Penalize empty/no-op actions
        if action.lower() in ['', ' ', 'noop', 'wait', 'sleep']:
            q_value -= 0.1
        # Bonus for meaningful commands
        elif any(cmd in action.lower() for cmd in ['cat', 'grep', 'find', 'python', 'run', 'execute', 'create', 'write', 'copy', 'move', 'install', 'pip', 'apt', 'chmod', 'chown']):
            q_value += 0.05
    
    # Check for partial progress indicators
    progress_patterns = ['progress', 'percent', 'completed', 'remaining', 'step', 'stage', 'phase']
    for pattern in progress_patterns:
        if pattern in next_state_lower:
            q_value += 0.1
            break
    
    # Clamp to reasonable range [0.0, 1.5]
    q_value = max(0.0, min(1.5, q_value))
    
    return q_value