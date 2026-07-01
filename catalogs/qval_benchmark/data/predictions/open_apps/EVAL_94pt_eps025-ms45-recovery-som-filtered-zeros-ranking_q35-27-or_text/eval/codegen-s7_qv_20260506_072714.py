import re

def signal_function(state: str, action: str, next_state: str) -> float:
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for goal achievement in next_state
    success_patterns = [
        'saved', 'created', 'added', 'completed', 'sent', 'success', 'done',
        'event created', 'task added', 'message sent', 'event saved', 'task completed'
    ]
    if any(p in next_state_lower for p in success_patterns):
        return 0.95
    
    # Base Q-value from action type (productivity score)
    if 'noop' in action_lower:
        base_q = -0.15
    elif 'fill' in action_lower:
        base_q = 0.45
    elif 'press' in action_lower:
        base_q = 0.4
    elif 'click' in action_lower:
        base_q = 0.35
    elif 'scroll' in action_lower:
        base_q = 0.1
    else:
        base_q = 0.05
    
    q_value = base_q
    
    # State relevance: presence of task-related keywords
    task_keywords = [
        'calendar', 'todo', 'messenger', 'maps', 'editor', 'event', 'task',
        'message', 'form', 'input', 'submit', 'save', 'create', 'add', 'send',
        'new', 'edit', 'delete', 'button', 'field', 'text'
    ]
    relevance_count = sum(1 for kw in task_keywords if kw in state_lower)
    q_value += min(relevance_count * 0.04, 0.25)
    
    # Error/failure indicators in next_state (penalize)
    error_keywords = ['error', 'failed', 'invalid', 'required', 'missing', 'not found', 'cannot']
    if any(e in next_state_lower for e in error_keywords):
        q_value -= 0.3
    
    # Progress indicators in next_state (reward)
    progress_keywords = ['filled', 'entered', 'selected', 'updated', 'changed', 'modified', 'set']
    if any(p in next_state_lower for p in progress_keywords):
        q_value += 0.2
    
    # Navigation indicators (being on relevant page)
    nav_keywords = ['calendar', 'todo list', 'messages', 'inbox', 'map', 'code editor']
    if any(nav in state_lower for nav in nav_keywords):
        q_value += 0.1
    
    # Form completion signals
    completion_signals = ['complete', 'finish', 'ready', 'confirmed', 'applied']
    if any(c in next_state_lower for c in completion_signals):
        q_value += 0.15
    
    # Clamp Q-value to reasonable bounds
    q_value = max(-0.5, min(1.0, q_value))
    
    return q_value