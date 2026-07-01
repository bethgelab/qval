import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for a given state, action, and next state in ALFWorld.
    Based on step count, action relevance, and progress indicators.
    """
    
    # Extract step count from state text
    step_pattern = r'(\d+) step'
    steps_match = re.search(step_pattern, state, re.IGNORECASE)
    steps = int(steps_match.group(1)) if steps_match else 35
    
    # Calculate step-based score (fewer steps = higher value)
    step_score = 1.0 - (steps / 40.0)
    
    # Check if action appears task-relevant
    task_keywords = ['go', 'pick', 'put', 'clean', 'drop', 'open', 'close', 'hold', 'move', 'take', 'wash', 'dry']
    action_lower = action.lower()
    action_relevant = any(kw in action_lower for kw in task_keywords)
    
    # Check for progress indicators in next state
    progress_indicators = ['cleaned', 'placed', 'moved', 'completed', 'done', 'success', 'opened', 'closed', 'picked up']
    next_state_lower = next_state.lower()
    shows_progress = any(ind in next_state_lower for ind in progress_indicators)
    
    # Check if current state already shows progress
    state_progress = any(ind in state.lower() for ind in progress_indicators)
    
    # Calculate Q-value with weighted components
    # Step count is most important (sparse reward, efficiency matters)
    # Action relevance and progress are secondary indicators
    q_value = step_score * 0.6 + (1.0 if action_relevant else 0.0) * 0.2 + (1.0 if shows_progress else 0.0) * 0.15 + (1.0 if state_progress else 0.0) * 0.05
    
    # Clamp to valid range [0, 1]
    return max(0.0, min(1.0, q_value))