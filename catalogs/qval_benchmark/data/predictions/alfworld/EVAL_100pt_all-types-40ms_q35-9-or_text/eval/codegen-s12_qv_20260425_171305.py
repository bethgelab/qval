import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    goal_keywords = ['goal', 'done', 'success', 'task completed', 'task complete', 
                     'all done', 'finished', 'completed', 'succeeded']
    
    for kw in goal_keywords:
        if kw in next_state.lower():
            q_value = 1.0
            break
    
    if q_value < 1.0:
        action_keywords = ['move', 'pick up', 'put down', 'open', 'close', 
                          'clean', 'go to', 'take', 'give', 'drop']
        
        has_valid_action = any(kw in action.lower() for kw in action_keywords)
        
        if has_valid_action:
            q_value = 0.25
        else:
            q_value = -0.5
        
        progress_keywords = ['holding', 'picked up', 'at', 'near', 'closer', 
                            'moved', 'arrived', 'reached', 'placed']
        
        for kw in progress_keywords:
            if kw in next_state.lower():
                q_value += 0.2
                break
        
        negative_keywords = ['error', 'failed', 'invalid', 'cannot', 'blocked', 
                            'impossible', 'unavailable', 'not able']
        
        for kw in negative_keywords:
            if kw in next_state.lower():
                q_value -= 0.6
                break
        
        location_change = False
        if state and next_state:
            state_loc = re.search(r'(room|kitchen|living|bedroom|bathroom|hallway|dining|study)', state.lower())
            next_loc = re.search(r'(room|kitchen|living|bedroom|bathroom|hallway|dining|study)', next_state.lower())
            
            if state_loc and next_loc:
                if state_loc.group(1) != next_loc.group(1):
                    location_change = True
        
        if location_change:
            q_value += 0.15
        
        if q_value < -0.8:
            q_value = -0.8
        
        q_value = max(-1.0, min(1.0, q_value))
    
    return q_value