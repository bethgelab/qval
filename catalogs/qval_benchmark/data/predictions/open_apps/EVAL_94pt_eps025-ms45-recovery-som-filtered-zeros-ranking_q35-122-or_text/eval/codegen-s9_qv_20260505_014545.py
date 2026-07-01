def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Initialize base Q-value
    q_value = 0.0
    
    # Convert to lowercase for easier matching
    state_text = state.lower()
    next_text = next_state.lower()
    action_text = action.lower()
    
    # Check for task completion indicators in next state
    completion_signals = ['success', 'completed', 'saved', 'sent', 'added', 'created', 
                          'done', 'task added', 'event created', 'message sent',
                          'confirmation', 'confirm', 'successfully']
    
    for signal in completion_signals:
        if signal in next_text:
            return 1.0
    
    # Check for error/failure signals in next state
    error_signals = ['error', 'failed', 'invalid', 'wrong', 'missing', 'required',
                     'not found', 'cannot', 'unable', 'unavailable']
    
    error_count = sum(1 for signal in error_signals if signal in next_text)
    if error_count >= 2:
        return 0.0
    
    # Analyze action quality based on type and targets
    action_score = 0.0
    
    # Good actions: click, fill with meaningful targets
    if 'click' in action_text and 'bid' in action_text:
        action_score = 0.25
    elif 'fill' in action_text and 'bid' in action_text:
        action_score = 0.30
    elif 'press' in action_text and ('enter' in action_text or 'return' in action_text):
        action_score = 0.35
    elif 'scroll' in action_text:
        action_score = 0.10
    elif 'noop' in action_text:
        action_score = 0.05
    
    # Check for progress indicators between states
    progress_score = 0.0
    
    # Count filled form elements (value attributes with content)
    current_filled = len(re.findall(r"value=['\"]?[^'\"\s]+['\"]?", state_text))
    next_filled = len(re.findall(r"value=['\"]?[^'\"\s]+['\"]?", next_text))
    
    # Count bid numbers (interactive elements)
    current_bids = len(re.findall(r"bid['\"]?\s*[:=]?\s*['\"]?\d+", state_text))
    next_bids = len(re.findall(r"bid['\"]?\s*[:=]?\s*['\"]?\d+", next_text))
    
    # Progress if more elements filled or interacted with
    if next_filled > current_filled:
        progress_score = 0.25
    elif next_bids > current_bids:
        progress_score = 0.15
    elif next_bids == current_bids and next_filled == current_filled:
        progress_score = 0.10  # No regression
    
    # Check for task-specific keywords that indicate progress
    task_keywords = ['todo', 'calendar', 'event', 'message', 'task', 'note', 
                     'title', 'content', 'body', 'description', 'subject', 'date', 'time']
    
    task_relevant = sum(1 for kw in task_keywords if kw in next_text)
    if task_relevant >= 3:
        progress_score = max(progress_score, 0.20)
    
    # Final Q-value estimation
    q_value = action_score + progress_score
    
    # Normalize to [0, 1] range
    q_value = min(1.0, max(0.0, q_value))
    
    # Bonus for actions that seem to be near completion
    final_actions = ['submit', 'save', 'send', 'create', 'add', 'finish', 'done']
    if any(final in action_text for final in final_actions):
        q_value = min(1.0, q_value + 0.15)
    
    # Penalty for error signals (even if not complete failure)
    if error_count == 1:
        q_value = max(0.0, q_value - 0.1)
    
    return q_value