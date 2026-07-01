import re
import json

def signal_function(state: str) -> float:
    """
    Estimate state-value for ALFWorld environment based on state analysis.
    Higher values indicate states closer to successful goal completion.
    """
    state_lower = state.lower()
    
    # Keywords indicating near completion or success
    success_indicators = [
        'done', 'complete', 'success', 'successfully',
        'placed', 'put', 'move', 'moved', 'clean', 'cleaned',
        'pick', 'picked', 'take', 'taken', 'give', 'given',
        'achieved', 'finished', 'completed', 'succeed'
    ]
    
    # Keywords indicating failure or problems
    failure_indicators = [
        'error', 'fail', 'failed', 'cannot', 'unable', 'blocked',
        'invalid', 'wrong', 'not', 'missing', 'broken', 'impossible'
    ]
    
    # Check for explicit success/failure messages
    if 'task done' in state_lower or 'task complete' in state_lower or 'goal achieved' in state_lower:
        return 0.95
    if 'task failed' in state_lower or 'failed' in state_lower or 'error' in state_lower:
        return 0.05
    
    # Count indicator occurrences
    success_count = sum(1 for kw in success_indicators if kw in state_lower)
    failure_count = sum(1 for kw in failure_indicators if kw in state_lower)
    
    # Base score from indicators
    base_score = 0.5
    
    # Adjust based on indicators
    if success_count > 0:
        base_score += min(0.3, success_count * 0.05)
    if failure_count > 0:
        base_score -= min(0.3, failure_count * 0.05)
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, base_score))