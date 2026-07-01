import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    completion_indicators = ['completed', 'success', 'done', 'saved', 'added', 
                             'created', 'sent', 'delivered', 'finished', 'goal', 
                             'check', 'checkmark', '✓', '✔', 'complete']
    
    if any(kw.lower() in next_state.lower() for kw in completion_indicators):
        q_value += 0.7
    
    if any(kw.lower() in state.lower() for kw in completion_indicators):
        q_value += 0.2
    
    action_match = re.search(r"click\('(\d+)'", action)
    if action_match:
        bid = action_match.group(1)
        if bid in next_state:
            q_value += 0.25
        else:
            q_value -= 0.1
    
    fill_match = re.search(r"fill\('(\d+)',", action)
    if fill_match:
        bid = fill_match.group(1)
        if bid in next_state:
            q_value += 0.15
        else:
            q_value -= 0.1
    
    press_match = re.search(r"press\('(\d+)',", action)
    if press_match:
        bid = press_match.group(1)
        if bid in next_state:
            q_value += 0.1
    
    state_bids = len(re.findall(r"bid='(\d+)'", state))
    next_bids = len(re.findall(r"bid='(\d+)'", next_state))
    
    if next_bids > state_bids:
        q_value += 0.1
    elif next_bids < state_bids:
        q_value -= 0.1
    
    if 'step' in state.lower() or 'limit' in state.lower():
        q_value -= 0.05
    
    q_value = max(-0.5, min(1.0, q_value))
    
    return q_value