import re
import json
import string

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    value = 0.5
    
    if 'completed' in state_lower or 'success' in state_lower or 'goal reached' in state_lower:
        return 0.99
    
    if 'failed' in state_lower or 'error' in state_lower or 'blocked' in state_lower:
        return 0.1
    
    if 'holding' in state_lower or 'carrying' in state_lower:
        value += 0.15
    
    if 'picked up' in state_lower or 'take' in state_lower:
        value += 0.1
    
    if 'clean' in state_lower or 'cleaning' in state_lower:
        value += 0.05
    
    if 'move' in state_lower or 'moved' in state_lower:
        value += 0.05
    
    if 'put' in state_lower or 'drop' in state_lower or 'place' in state_lower:
        value += 0.05
    
    if 'kitchen' in state_lower or 'bedroom' in state_lower or 'living room' in state_lower:
        value += 0.03
    
    if 'goal' in state_lower:
        value += 0.05
    
    if 'task' in state_lower:
        value += 0.02
    
    if 'remaining' in state_lower or 'steps' in state_lower:
        value -= 0.02
    
    if 'cannot' in state_lower or 'unable' in state_lower:
        value -= 0.1
    
    value = max(0.0, min(1.0, value))
    return value