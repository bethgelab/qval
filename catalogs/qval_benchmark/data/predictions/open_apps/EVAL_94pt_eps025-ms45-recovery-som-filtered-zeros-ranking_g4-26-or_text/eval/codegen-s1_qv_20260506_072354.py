import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value by analyzing the progress made between the current state 
    and the next state, the utility of the action taken, and keywords indicating 
    task completion or failure.
    """
    if not state or not next_state:
        return 0.0
    
    # 1. Check for stagnation (no change in state)
    # Taking an action that results in no change is generally suboptimal unless it's a necessary wait.
    if state == next_state:
        return 0.05
    
    # 2. Check for error or failure indicators in the new state
    # If the application shows error messages, the current action/path is likely bad.
    error_patterns = [
        r'\berror\b', r'\binvalid\b', r'\bfailed\b', r'\brequired\b', 
        r'\bnot found\b', r'\bunable to\b', r'\bcannot\b', r'\bwrong\b'
    ]
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.0
    
    # 3. Check for task completion (Goal achievement)
    # If the next state contains words indicating the task is done, return a high Q-value.
    completion_patterns = [
        r'\bsuccess\b', r'\bsent\b', r'\badded\b', r'\bsaved\b', r'\bcreated\b',
        r'\bscheduled\b', r'\bcompleted\b', r'\bdone\b', r'message sent',
        r'event created', r'direction found', r'task added',
        r'\bconfirmed\b', r'\bupdated\b'
    ]
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.95
            
    # 4. Analyze the action and its immediate impact
    score = 0.2  # Baseline score for a state change
    
    # Parse the action string using regex to extract type and arguments
    # Examples: click("10"), fill("5", "hello"), press("1", "Enter")
    action_match = re.match(r'(\w+)\((.*)\)', action)
    if action_match:
        act_type = action_match.group(1)
        act_args_str = action_match.group(2)
        # Extract quoted strings to get actual arguments
        args = re.findall(r"['\"]([^'\"]*)['\"]", act_args_str)
        
        if act_type == 'fill' and len(args) >= 2:
            # args[1] is the text being filled
            text_val = args[1]
            # If the text we just typed is now visible in the accessibility tree
            if text_val and text_val in next_state:
                score += 0.5
        
        elif act_type == 'click':
            # Detect if clicking led to a significant change (navigation or UI update)
            old_bids = len(re.findall(r'bid:\s*\d+', state))
            new_bids = len(re.findall(r'bid:\s*\d+', next_state))
            
            # Check if number of interactive elements changed or the text length changed drastically
            if old_bids > 0 and new_bids > 0:
                if new_bids != old_bids or abs(len(next_state) - len(state)) > 150:
                    score += 0.3
            elif new_bids > old_bids:
                score += 0.3

        elif act_type == 'press':
            # Pressing Enter or Return is often a way to submit a form
            if 'Enter' in act_args_str or 'Return' in act_args_str:
                score += 0.2
                
    # 5. General structural progress
    # If new interactive elements appeared, it's a sign of progress toward a goal.
    old_bids = len(re.findall(r'bid:\s*\d+', state))
    new_bids = len(re.findall(r'bid:\s*\d+', next_state))
    if new_bids > old_bids:
        score += 0.1
    elif old_bids > 0 and new_bids > 0 and abs(new_bids - old_bids) > 5:
        # Transitioning to a simpler/different layout (e.g., closing a menu)
        score += 0.1

    # Clamp the result between 0.0 and 1.0
    return min(max(float(score), 0.0), 1.0)