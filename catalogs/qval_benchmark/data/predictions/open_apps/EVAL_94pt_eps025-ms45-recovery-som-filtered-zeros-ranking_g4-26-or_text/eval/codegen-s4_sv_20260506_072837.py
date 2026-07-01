import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment.
    The estimate is based on detecting success, progress (filled inputs), 
    and readiness (presence of action buttons).
    """
    if not state:
        return 0.0
    
    text_lower = state.lower()
    
    # 1. Terminal Success Detection (Highest Value)
    # Look for indicators that the task-level goal has been achieved.
    # In OpenApps, these are often reflected in status messages or confirmation text.
    success_indicators = [
        'success', 'sent', 'saved', 'created', 'added', 
        'completed', 'done', 'confirmed', 'scheduled',
        'message sent', 'event created', 'task completed'
    ]
    for indicator in success_indicators:
        # Check for indicator as a standalone word or phrase to avoid false positives
        if re.search(rf'\b{indicator}\b', text_lower):
            return 1.0
            
    # 2. Progress: Detection of Filled Inputs
    # In the accessibility tree, populated inputs are represented by 'value="content"'.
    # We count non-empty values to gauge how much of the task is completed.
    filled_values = re.findall(r'value="([^"]+)"', state)
    filled_count = 0
    for v in filled_values:
        if v.strip():
            filled_count += 1
            
    # 3. Readiness: Detection of Actionable Buttons
    # We look for keywords associated with the final steps of common tasks.
    action_keywords = [
        'send', 'submit', 'save', 'create', 'add', 
        'search', 'go', 'confirm', 'apply', 'schedule', 'post'
    ]
    has_action_button = False
    for kw in action_keywords:
        # Look for the keyword specifically in accessibility attributes
        # like name, text, label, or description.
        pattern = rf'(name|text|label|description)="{kw}"'
        if re.search(pattern, text_lower):
            has_action_button = True
            break
            
    # 4. Value Estimation (Heuristic)
    # The estimation follows a scale reflecting proximity to a success state:
    # - One step away: Data is entered and a submission button is present. (0.6 - 0.95)
    # - In progress: Data has been entered, but the next step is unclear. (0.3 - 0.6)
    # - Ready to start: Action button is present, but fields are empty. (0.1 - 0.2)
    # - Initial state: Default low value. (0.05)
    
    if filled_count > 0:
        # Base score for having entered information
        score = 0.3 + 0.3 * min(1.0, filled_count / 3.0)
        if has_action_button:
            # Significant boost if we are ready to commit the input
            score += 0.35
        return min(0.95, score)
        
    if has_action_button:
        # We are at a decision or input point (e.g., a blank form with a 'Submit' button)
        return 0.15
        
    # Default baseline for being in an active app state
    return 0.05