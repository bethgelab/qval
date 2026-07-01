import re

def signal_function(state: str) -> float:
    """Estimate state value based on features indicating progress toward goal."""
    
    score = 0.0
    state_lower = state.lower()
    
    # Check for explicit task completion
    completion_keywords = ['success', 'completed', 'done', 'solved', 'goal accomplished', 'task complete']
    if any(kw in state_lower for kw in completion_keywords):
        return 1.0
    
    # Check for failure indicators
    failure_keywords = ['failed', 'impossible', 'cannot', 'error', 'invalid', 'timeout']
    if any(kw in state_lower for kw in failure_keywords):
        return 0.0
    
    # Check for holding/carrying an object (progress indicator)
    if any(kw in state_lower for kw in ['holding', 'inventory', 'you have', 'picked up', 'taken']):
        score += 0.15
    
    # Check for object manipulation progress
    manipulation_keywords = ['put', 'placed', 'moved', 'moved to', 'carried', 'dropped']
    if any(kw in state_lower for kw in manipulation_keywords):
        score += 0.12
    
    # Check for room navigation progress
    location_keywords = ['in ', 'on ', 'go to', 'walk to', 'enter', 'move to', 'at ']
    if any(kw in state_lower for kw in location_keywords):
        score += 0.08
    
    # Check for object state changes (clean, open, etc.)
    state_change_keywords = ['clean', 'open', 'close', 'turned on', 'turned off', 'filled', 'empty']
    if any(kw in state_lower for kw in state_change_keywords):
        score += 0.10
    
    # Check for goal-related object mentions
    goal_object_keywords = ['goal', 'target', 'needed', 'required', 'should', 'must']
    if any(kw in state_lower for kw in goal_object_keywords):
        score += 0.05
    
    # Check for recent action confirmation (shows agent is active)
    action_confirmed = any(kw in state_lower for kw in ['action', 'command', 'response', 'execute'])
    if action_confirmed:
        score += 0.03
    
    # Check for valid room/object descriptions (shows environment is responsive)
    room_pattern = re.search(r'\b(?:kitchen|bedroom|bathroom|living room|office|hallway|study|laundry)\b', state_lower)
    if room_pattern:
        score += 0.05
    
    object_pattern = re.search(r'\b(?:sink|table|counter|desk|shelf|cabinet|drawer|toilet|bathtub)\b', state_lower)
    if object_pattern:
        score += 0.03
    
    # Base score for being in a valid game state
    score += 0.02
    
    # Penalize if step limit is approaching (state suggests long episode)
    if 'step' in state_lower or 'turn' in state_lower:
        step_match = re.search(r'\b(\d+)\b', state_lower)
        if step_match:
            step_num = int(step_match.group(1))
            if step_num > 30:
                score *= 0.5  # Reduce value if near step limit
    
    return min(max(score, 0.0), 1.0)