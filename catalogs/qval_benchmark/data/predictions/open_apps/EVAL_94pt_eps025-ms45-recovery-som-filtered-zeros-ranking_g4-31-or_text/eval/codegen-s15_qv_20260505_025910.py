import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    
    The estimate is based on identifying success/error indicators in the transition 
    and analyzing the intent of the action by examining the context of the target element.
    """
    # Keywords indicative of goal achievement
    success_keywords = {
        'success', 'created', 'added', 'sent', 'saved', 'completed', 
        'confirmed', 'done', 'finished', 'successfully', 'submitted'
    }
    # Keywords indicative of failure or obstacles
    error_keywords = {
        'error', 'invalid', 'required', 'failed', 'wrong', 
        'incorrect', 'unable', 'could not', 'missing'
    }
    # Keywords indicative of positive progress
    progress_keywords = {
        'save', 'submit', 'add', 'send', 'create', 'confirm', 
        'next', 'ok', 'yes', 'update', 'new'
    }
    # Keywords indicative of backtracking or cancellation
    negative_keywords = {
        'cancel', 'delete', 'remove', 'discard', 'back', 'exit', 'no', 'close'
    }

    s_low = state.lower()
    ns_low = next_state.lower()

    # 1. Check for goal achievement: High reward if success indicators appear in next_state
    for word in success_keywords:
        if word in ns_low and word not in s_low:
            return 1.0

    # 2. Check for errors: Low reward if error indicators appear in next_state
    for word in error_keywords:
        if word in ns_low and word not in s_low:
            return 0.1

    # 3. Analyze the specific action and the element it targeted
    # BrowserGym actions typically follow patterns like click('bid') or fill('bid', 'text')
    bid_match = re.search(r"'(.*?)'", action)
    if bid_match:
        bid = bid_match.group(1)
        # Find the bid in the current state string to understand the element's purpose
        start_idx = state.find(bid)
        if start_idx != -1:
            # Extract a local window of text around the bid to find descriptive labels
            window_start = max(0, start_idx - 40)
            window_end = min(len(state), start_idx + 40)
            window = state[window_start:window_end].lower()
            
            # Action targets an element associated with positive progress
            for word in progress_keywords:
                if word in window:
                    return 0.8
            
            # Action targets an element associated with negative progress (e.g., 'Cancel')
            for word in negative_keywords:
                if word in window:
                    return 0.2

    # 4. Evaluate by action type if no specific keyword was found in the element's context
    if "fill" in action:
        # Filling a form is generally productive progress
        return 0.6
    
    if "click" in action:
        # Clicking is often progress, but less certain than submitting/filling
        return 0.5

    # 5. General state transition analysis
    # If the state changed significantly, it's likely the agent moved to a new page or view
    if ns_low != s_low:
        return 0.4

    # Default value for actions that result in no observable progress (e.g., noop, redundant clicks)
    return 0.2