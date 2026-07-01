import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Normalize inputs for case-insensitive matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # 1. Check for Terminal Success State
    # If the next state indicates task completion, the Q-value should be maximal (1.0)
    success_patterns = [
        'success', 
        'task complete', 
        'done', 
        'goal achieved', 
        'you have completed'
    ]
    if any(pattern in next_state_lower for pattern in success_patterns):
        return 1.0
    
    # 2. Check for Terminal Failure/Invalid State
    # If the action resulted in an error or invalid state, Q-value should be minimal (0.0)
    failure_patterns = [
        'cannot', 
        'invalid', 
        'error', 
        'nothing happened', 
        'fail', 
        'not found'
    ]
    if any(pattern in next_state_lower for pattern in failure_patterns):
        return 0.0
    
    # 3. Check for State Progress
    # If the state did not change, the action was likely ineffective or redundant
    if state == next_state:
        return 0.0
    
    # 4. Estimate Progress Quality based on Action Confirmation
    # In ALFWorld, successful actions are often confirmed in the observation text
    # Patterns indicating successful manipulation or navigation
    confirmation_patterns = [
        r'you\s+put\s+',
        r'you\s+took\s+',
        r'you\s+went\s+to',
        r'you\s+are\s+in',
        r'you\s+see',
        r'you\s+cleaned',
        r'you\s+ate'
    ]
    
    action_confirmed = False
    for pattern in confirmation_patterns:
        if re.search(pattern, next_state_lower):
            action_confirmed = True
            break
            
    # 5. Evaluate Action Productivity
    # Certain verbs are core to task completion in ALFWorld
    productive_verbs = [
        'put', 'take', 'go', 'clean', 'heat', 'cool', 'wash', 'eat', 'wipe', 'cut'
    ]
    is_productive = any(verb in action_lower for verb in productive_verbs)
    
    # 6. Calculate Q-value Heuristic
    # Reward is sparse (1.0 at end), so we approximate expected future return
    # based on immediate progress indicators.
    
    if action_confirmed and is_productive:
        # Strong progress: Action confirmed and verb is productive
        return 0.7
    elif action_confirmed:
        # Progress: Action confirmed but verb generic
        return 0.5
    elif is_productive:
        # Weak progress: Productive action but no explicit confirmation in text
        return 0.3
    else:
        # Minimal progress: State changed but not clearly linked to productive action
        return 0.1