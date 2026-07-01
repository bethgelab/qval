import re

def signal_function(state: str, action: str, next_state: str) -> float:
    action_score = 0.0
    if action.startswith('click('):
        action_score = 0.6
    elif action.startswith('fill('):
        action_score = 0.5
    elif action.startswith('press('):
        action_score = 0.4
    elif action.startswith('noop('):
        action_score = 0.1
    elif action.startswith('scroll('):
        action_score = 0.2
    else:
        action_score = 0.1
    
    state_score = 0.0
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    goal_indicators = ['goal achieved', 'task complete', 'success', 'done', 
                       'saved', 'created', 'added', 'sent', 'submitted',
                       'calendar', 'event', 'message', 'todo', 'code']
    for indicator in goal_indicators:
        if indicator in state_lower:
            state_score = max(state_score, 0.8)
            break
    
    if 'form' in state_lower and 'complete' in state_lower:
        state_score = max(state_score, 0.7)
    
    error_indicators = ['error', 'failed', 'invalid', 'missing', 'cannot']
    for indicator in error_indicators:
        if indicator in state_lower:
            state_score = max(0.0, state_score - 0.3)
            break
    
    progress_indicators = ['step', 'progress', 'loading', 'pending', 'waiting']
    for indicator in progress_indicators:
        if indicator in state_lower:
            state_score = max(state_score, 0.4)
            break
    
    bid_count = 0
    bid_pattern = re.compile(r'\d+')
    for match in bid_pattern.finditer(state):
        bid_count += 1
    for match in bid_pattern.finditer(next_state):
        bid_count += 1
    
    if bid_count > 0 and bid_count <= 30:
        state_score = max(state_score, 0.3)
    
    state_score = max(0.0, min(1.0, state_score))
    action_score = max(0.0, min(1.0, action_score))
    
    q_value = 0.7 * state_score + 0.3 * action_score
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value