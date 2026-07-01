def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value estimate
    q_value = 0.0
    
    # Positive indicators suggesting progress or success
    positive_keywords = ['success', 'completed', 'done', 'passed', 'verified', 
                        'ok', 'true', 'yes', 'found', 'created', 'generated',
                        'output', 'result', 'answer', 'solution', 'correct',
                        'completed', 'finished', 'ready', 'available', 'exists']
    
    # Negative indicators suggesting failure or errors
    negative_keywords = ['error', 'fail', 'failed', 'exception', 'traceback', 
                        'permission denied', 'not found', 'no such', 'cannot',
                        'timeout', 'killed', 'segfault', 'abort', 'refused',
                        'denied', 'invalid', 'missing', 'undefined']
    
    # Goal-related keywords
    goal_keywords = ['goal', 'target', 'objective', 'task', 'complete', 'finish']
    
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Count positive and negative indicators in next_state
    positive_count = sum(1 for kw in positive_keywords if kw in next_state_lower)
    negative_count = sum(1 for kw in negative_keywords if kw in next_state_lower)
    goal_count = sum(1 for kw in goal_keywords if kw in next_state_lower)
    
    # Adjust Q-value based on indicators
    q_value += positive_count * 0.15
    q_value -= negative_count * 0.25
    q_value += goal_count * 0.2
    
    # Detect state progression (new content in next_state)
    state_lines = set(state_lower.split('\n'))
    next_lines = set(next_state_lower.split('\n'))
    new_content = next_lines - state_lines
    if len(new_content) > 0:
        q_value += 0.1
    
    # Action effectiveness - productive commands
    productive_actions = ['python', 'run', 'execute', 'submit', 'verify', 'test',
                         'create', 'write', 'generate', 'process', 'compute',
                         'solve', 'calculate', 'build', 'compile', 'install']
    if any(kw in action_lower for kw in productive_actions):
        q_value += 0.1
    
    # Penalize potentially destructive or uncertain actions
    risky_actions = ['rm', 'delete', 'remove', 'kill', 'force', 'overwrite']
    if any(kw in action_lower for kw in risky_actions):
        q_value -= 0.15
    
    # Check for empty or minimal state changes (stagnation penalty)
    if len(next_state_lower.strip()) < 10 and len(state_lower.strip()) > 0:
        q_value -= 0.1
    
    # Strong success indicators boost value significantly
    if any(kw in next_state_lower for kw in ['success', 'passed', 'verified', 'completed']):
        q_value += 0.3
    
    # Strong failure indicators reduce value significantly
    if any(kw in next_state_lower for kw in ['error', 'failed', 'exception', 'traceback']):
        q_value -= 0.4
    
    # Clamp to reasonable range [-1.0, 1.0]
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value