import re

def signal_function(state: str) -> float:
    if not state:
        return 0.0
    
    state_lower = state.lower()
    
    # Check for immediate success or task completion
    success_keywords = ['success', 'task complete', 'done', 'complete']
    if any(kw in state_lower for kw in success_keywords):
        return 1.0
    
    # Check for blocked or error states which indicate failure to proceed
    block_keywords = ['cannot', 'broken', 'locked', 'impossible', 'error']
    if any(kw in state_lower for kw in block_keywords):
        return 0.0
    
    score = 0.0
    
    # Active manipulation (holding) is a strong indicator of progress towards a goal
    if re.search(r'holding|you are holding', state_lower):
        score += 0.5
    
    # Objects placed on surfaces indicates partial progress
    if re.search(r'on the|on a', state_lower):
        score += 0.3
    
    # Objects inside containers indicates partial progress
    if re.search(r'in the|in a', state_lower):
        score += 0.2
    
    # Being in a room context is necessary for task execution
    if re.search(r'room|kitchen|bedroom|bathroom|hallway', state_lower):
        score += 0.1
    
    # Ensure score is within [0, 1]
    return min(1.0, max(0.0, score))