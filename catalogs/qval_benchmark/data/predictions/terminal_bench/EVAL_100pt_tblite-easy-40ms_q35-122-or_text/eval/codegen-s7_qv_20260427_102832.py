def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    # Base Q-value starts at 0.5 (neutral expectation)
    q_value = 0.5
    
    # Success and failure indicators for TerminalBench tasks
    success_keywords = [
        'success', 'complete', 'done', 'pass', 'verified', 'correct', 
        'ok', 'true', 'passed', 'completed', 'finished', 'accepted',
        'match', 'equal', 'valid', 'ok', 'test passed'
    ]
    
    failure_keywords = [
        'error', 'failed', 'fail', 'incorrect', 'wrong', 'invalid',
        'exception', 'traceback', 'denied', 'permission', 'not found',
        'no such', 'cannot', 'failed to', 'unable', 'undefined'
    ]
    
    # Task completion patterns
    completion_patterns = [
        'created', 'written', 'saved', 'generated', 'output', 'result',
        'solution', 'answer', 'flag', 'key', 'password', 'hash'
    ]
    
    # Normalize strings for analysis
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Count indicators in next_state
    success_count = sum(1 for kw in success_keywords if kw in next_state_lower)
    failure_count = sum(1 for kw in failure_keywords if kw in next_state_lower)
    
    # Reward success indicators found in next_state
    if success_count > 0:
        q_value += 0.25 * min(success_count, 3)
    
    # Penalize failure indicators in next_state
    if failure_count > 0:
        q_value -= 0.2 * min(failure_count, 3)
    
    # Check if action appears to be a meaningful command
    meaningful_action_patterns = [
        r'\b(cat|ls|cd|grep|find|python|bash|sh|echo|mkdir|cp|mv|rm|chmod|chmod|chmod)\b',
        r'\b(test|verify|check|validate)\b',
        r'\b\(.*\)',  # Looks like a command with arguments
        r'\|',  # Contains pipe
        r'&&',  # Contains command chaining
    ]
    
    action_meaningful = any(re.search(pat, action_lower) for pat in meaningful_action_patterns)
    if action_meaningful:
        q_value += 0.1
    
    # Check for progress (next_state differs from state)
    if next_state != state and len(next_state.strip()) > 0:
        # Measure content change ratio
        state_len = max(len(state), 1)
        next_len = max(len(next_state), 1)
        if next_len > state_len * 0.5:  # Meaningful change
            q_value += 0.05
    
    # Reward completion patterns
    completion_count = sum(1 for pat in completion_patterns if pat in next_state_lower)
    if completion_count > 0:
        q_value += 0.2 * min(completion_count, 2)
    
    # Bonus for explicit test/verification success
    if 'test' in next_state_lower and 'passed' in next_state_lower:
        q_value += 0.4
    if 'verification' in next_state_lower and ('success' in next_state_lower or 'pass' in next_state_lower):
        q_value += 0.4
    
    # Penalize if action made state worse (more errors than before)
    state_failure_count = sum(1 for kw in failure_keywords if kw in state_lower)
    if failure_count > state_failure_count:
        q_value -= 0.15 * min(failure_count - state_failure_count, 3)
    
    # Bonus for specific TerminalBench task patterns
    if 'solution' in next_state_lower or 'answer' in next_state_lower:
        q_value += 0.3
    if 'flag' in next_state_lower or 'key' in next_state_lower:
        q_value += 0.35
    
    # Clamp Q-value to [0.0, 1.0] range
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)