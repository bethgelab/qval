import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next_state.
    The Q-value is an approximation of the expected return, favoring actions 
    that lead to task completion or necessary progress.
    """
    # 1. Parse the action type and the target bid
    action_type_match = re.match(r"^(\w+)", action)
    if not action_type_match:
        return 0.0
    action_type = action_type_match.group(1)
    
    # Extract the bid (target element ID) from the action string (e.g., click('12'))
    bid_match = re.search(r"['\"](\d+)['\"]", action)
    bid = bid_match.group(1) if bid_match else None
    
    # 2. Detect task completion in the next_state
    # If the next_state shows signs of goal achievement, return the maximum reward.
    success_indicators = [
        'success', 'sent', 'added', 'saved', 'created', 'done', 
        'completed', 'message sent', 'event added', 'task added',
        'appointment added', 'todo added', 'task completed'
    ]
    next_state_lower = next_state.lower()
    if any(indicator in next_state_lower for indicator in success_indicators):
        return 1.0
        
    # 3. Analyze the action to estimate progress
    score = 0.0
    
    if action_type == 'click':
        element_name = ""
        if bid:
            # Attempt to extract the element's name/label from the accessibility tree
            # We check for the common pattern [bid="X", name="Y"] and [name="Y", bid="X"]
            name_pattern_fwd = rf'bid=["\']?{bid}["\']?.*?name=["\'](.*?)["\']'
            name_match_fwd = re.search(name_pattern_fwd, state, re.DOTALL)
            
            if name_match_fwd:
                element_name = name_match_fwd.group(1).lower()
            else:
                name_pattern_rev = rf'name=["\'](.*?)["\'].*?bid=["\']?{bid}["\']?'
                name_match_rev = re.search(name_pattern_rev, state, re.DOTALL)
                if name_match_rev:
                    element_name = name_match_rev.group(1).lower()
        
        # High-value actions: committing information or triggering creation
        commit_keywords = [
            'send', 'submit', 'add', 'save', 'create', 'confirm', 
            'search', 'ok', 'go', 'enter', 'delete', 'remove', 'clear'
        ]
        if any(kw in element_name for kw in commit_keywords):
            score = 0.7
        else:
            # General navigation or selection clicks
            score = 0.2
            
    elif action_type == 'fill':
        # Filling form fields is a key preparatory step for goal achievement
        score = 0.4
        
    elif action_type == 'press':
        # Pressing 'Enter' is often a shortcut for a 'submit' action
        if 'enter' in action.lower():
            score = 0.5
        else:
            score = 0.2
            
    elif action_type == 'scroll':
        # Scrolling is primarily navigational and has lower immediate value
        score = 0.1
        
    elif action_type == 'noop':
        # No-ops typically do not progress toward a goal
        score = 0.0
        
    else:
        # Default low score for unknown actions
        score = 0.1

    # Final Q-value is clamped between 0.0 and 1.0
    return max(0.0, min(score, 1.0))