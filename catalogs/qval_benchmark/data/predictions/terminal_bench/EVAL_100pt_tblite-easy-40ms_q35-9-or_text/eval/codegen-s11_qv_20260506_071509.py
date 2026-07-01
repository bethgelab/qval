import re

def signal_function(state: str, action: str, next_state: str) -> float:
    positive_patterns = ['success', 'done', 'completed', 'passed', 'finished', 'ok', 'true', 'yes', 'root', 'target', 'created', 'generated', 'wrote', 'saved', 'installed', 'updated', 'configured', 'initialized', 'ready', 'valid', 'accepted', 'verified', 'test passed', 'test succeeded']
    negative_patterns = ['error', 'failed', 'exception', 'timeout', 'refused', 'denied', 'false', 'no', 'broken', 'invalid', 'missing', 'unauthorized', 'permission', 'connection refused', 'killed', 'segfault', 'crash', 'abort', 'exit code', 'non-zero', 'warning', 'failed', 'invalid', 'corrupt', 'corrupted', 'unreachable', 'unavailable']
    
    state_positive_count = sum(1 for p in positive_patterns if p.lower() in state.lower())
    state_negative_count = sum(1 for p in negative_patterns if p.lower() in state.lower())
    state_score = state_positive_count - state_negative_count
    
    action_score = 0.0
    if action.strip():
        action_score = 0.2
    
    next_state_positive_count = sum(1 for p in positive_patterns if p.lower() in next_state.lower())
    next_state_negative_count = sum(1 for p in negative_patterns if p.lower() in next_state.lower())
    next_state_score = next_state_positive_count - next_state_negative_count
    
    improvement = next_state_score - state_score
    
    q_value = state_score * 0.3 + action_score * 0.2 + improvement * 0.5
    
    if next_state_positive_count > state_positive_count:
        q_value += 0.1
    if next_state_negative_count > state_negative_count:
        q_value -= 0.1
    
    return max(0.0, min(1.0, q_value))