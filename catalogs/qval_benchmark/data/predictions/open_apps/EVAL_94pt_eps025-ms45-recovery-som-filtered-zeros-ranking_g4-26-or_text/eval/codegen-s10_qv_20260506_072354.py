import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state-action-next_state transition in the OpenApps environment.
    The estimate is based on identifying progress, success, and errors in the accessibility tree.
    """
    # Parse action: action_type('bid', 'value') or action_type('bid')
    # Handles click('1'), fill('1', 'text'), press('1', 'Enter'), noop(100), scroll(10, 20)
    match = re.search(r"(\w+)\s*\(\s*['\"]?(\d+)['\"]?(?:,\s*['\"](.*?)['\"]?)?\s*\)", action)
    
    act_type = ""
    act_val = ""
    if match:
        act_type = match.group(1)
        act_val = match.group(3) if match.group(3) else ""
    else:
        # Fallback for simple actions without bids
        if 'noop' in action: act_type = 'noop'
        elif 'scroll' in action: act_type = 'scroll'
        elif 'press' in action: act_type = 'press'
        elif 'click' in action: act_type = 'click'
        elif 'fill' in action: act_type = 'fill'

    next_lower = next_state.lower()
    state_lower = state.lower()
    
    # 1. Success and Error Markers (High Priority)
    # If the next state indicates the goal was reached, Q-value is 1.0
    success_markers = [
        'success', 'sent', 'added', 'saved', 'created', 'completed', 
        'done', 'confirmed', 'message sent', 'event added', 
        'todo added', 'appointment added', 'scheduled', 'task created'
    ]
    error_markers = [
        'error', 'failed', 'invalid', 'not found', 'incorrect', 
        'try again', 'cannot', 'wrong', 'error occurred'
    ]
    
    if any(m in next_lower for m in success_markers):
        return 1.0
    if any(m in next_lower for m in error_markers):
        return 0.0

    # 2. Heuristic scoring for progress
    score = 0.0
    
    if act_type == 'fill' and act_val:
        # If the text we just filled is now present in the accessibility tree, 
        # it indicates successful input progress.
        if len(act_val) > 1 and act_val.lower() in next_lower:
            score = 0.7
        else:
            # Likely still in the process of filling or the input is being processed
            score = 0.3
    
    elif act_type == 'click':
        # Use a similarity measure to detect if a click led to a new page or menu (navigation)
        # Extract words to compute Jaccard similarity
        s_words = set(re.findall(r'\w+', state_lower))
        n_words = set(re.findall(r'\w+', next_lower))
        
        if not s_words or not n_words:
            score = 0.2
        else:
            intersection_count = len(s_words.intersection(n_words))
            union_count = len(s_words.union(n_words))
            similarity = intersection_count / union_count if union_count > 0 else 0
            
            if similarity < 0.5:
                # Large change in content suggests navigation or significant UI update
                score = 0.6
            elif similarity < 0.8:
                # Moderate change, maybe a dropdown or element expanded
                score = 0.3
            else:
                # Minimal change
                score = 0.1
    
    elif act_type == 'press':
        # Pressing 'Enter' is a common pattern for submitting forms
        if 'enter' in action.lower():
            score = 0.5
        else:
            score = 0.1
            
    elif act_type == 'scroll':
        score = 0.1
        
    elif act_type == 'noop':
        score = 0.0
        
    # 3. Density Check
    # If the text content increases significantly, it often signifies a successful page load/transition.
    if len(next_state) > len(state) * 1.5:
        score = max(score, 0.5)

    # Final clamp to ensure the value is in [0.0, 1.0]
    return float(min(1.0, max(0.0, score)))