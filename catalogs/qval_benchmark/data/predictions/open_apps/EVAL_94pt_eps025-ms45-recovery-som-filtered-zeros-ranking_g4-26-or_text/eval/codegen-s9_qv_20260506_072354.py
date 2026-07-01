import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) by analyzing the effectiveness of an action
    and the semantic change in the environment's state.
    """
    # 1. Basic Parsing of the action
    # Handles variants like click('12'), click(12), fill('12', 'text'), fill(12, 'text')
    fill_match = re.search(r"fill\(['\"]?(\d+)['\"]?,\s*['\"](.*?)['\"]", action)
    click_match = re.search(r"click\(['\"]?(\d+)['\"]?\)", action)
    
    # Base probability: 0.15 assumes we are in a valid episode and progress is possible
    q_val = 0.15

    # 2. Immediate Outcome Detection (in next_state)
    # Check for terminal success indicators
    success_patterns = [
        r"\bsuccess\b", r"\bsent\b", r"\bsaved\b", r"\badded\b", 
        r"\bcreated\b", r"\bcomplete\b", r"\bdone\b", r"\bscheduled\b", 
        r"\bupdated\b", r"\bconfirmed\b", r"\bfinished\b"
    ]
    for pat in success_patterns:
        if re.search(pat, next_state, re.IGNORECASE):
            return 1.0
    
    # Check for terminal failure or error indicators
    error_patterns = [r"\berror\b", r"\bfailed\b", r"\binvalid\b", r"\bnot found\b", r"\bwrong\b"]
    for pat in error_patterns:
        if re.search(pat, next_state, re.IGNORECASE):
            return 0.0

    # 3. Evaluate Action Progress
    if fill_match:
        # bid = fill_match.group(1)
        val = fill_match.group(2)
        # If the text typed is now present in the state (e.g., in an input value field),
        # it indicates the action was accepted by the application.
        if val and val in next_state:
            q_val += 0.4
        else:
            q_val += 0.1
            
    elif click_match:
        bid = click_match.group(1)
        
        # Identify the target text associated with the bid in the current state
        # Most accessibility trees format this as "[bid] text" or "bid: text"
        target_pattern = rf"(?:\[?{bid}\]?)\s*([^\[\n]+)"
        target_match = re.search(target_pattern, state)
        
        if target_match:
            target_text = target_match.group(1).strip().lower()
            # High-value semantic clicks
            if any(w in target_text for w in ['submit', 'send', 'save', 'add', 'create', 'ok', 'confirm', 'go']):
                q_val += 0.5
            # Negative semantic clicks
            elif any(w in target_text for w in ['cancel', 'clear', 'delete', 'remove', 'back']):
                q_val -= 0.3
            else:
                q_val += 0.2
        else:
            q_val += 0.1
            
        # Navigation/Change indicator: if the content changed significantly, it's likely progress
        if abs(len(next_state) - len(state)) > 30:
            q_val += 0.15
            
    elif "noop" in action:
        # No-ops are slightly penalizing to encourage efficiency
        q_val -= 0.05

    # 4. Final Normalization
    return max(0.0, min(1.0, q_val))