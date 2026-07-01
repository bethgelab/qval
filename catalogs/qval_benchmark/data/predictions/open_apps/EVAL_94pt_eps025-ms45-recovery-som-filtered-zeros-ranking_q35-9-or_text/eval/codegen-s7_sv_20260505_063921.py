import re
import math

def signal_function(state: str) -> float:
    # Check for goal completion indicators
    goal_complete_keywords = ['goal achieved', 'success', 'complete', 'done', 'task finished', 'verified', 'target reached']
    if any(keyword.lower() in state.lower() for keyword in goal_complete_keywords):
        return 1.0
    
    # Check for failure/error indicators
    failure_keywords = ['error', 'failed', 'invalid', 'not found', 'unavailable', 'disabled', 'blocked']
    if any(keyword.lower() in state.lower() for keyword in failure_keywords):
        return 0.0
    
    # Count interactive elements (bids) - more options generally better
    bid_count = len(re.findall(r'\bid\d+\b', state))
    
    # Check for form elements (fillable fields)
    form_field_count = len(re.findall(r'fill\(', state))
    
    # Check for clickable elements
    clickable_count = len(re.findall(r'click\(', state))
    
    # Check for navigation elements
    nav_count = len(re.findall(r'link|button|menu|nav', state, re.IGNORECASE))
    
    # Calculate a progress score based on available actions
    action_score = bid_count + form_field_count + clickable_count + nav_count
    
    # Base value from available actions (normalized)
    base_value = min(action_score / 100.0, 1.0)
    
    # Check for progress indicators
    progress_keywords = ['progress', 'step', 'current', 'remaining', 'pending', 'waiting']
    has_progress = any(keyword.lower() in state.lower() for keyword in progress_keywords)
    
    # Check for completion state
    completion_keywords = ['completed', 'finished', 'submitted', 'saved', 'created', 'added']
    completion_state = any(keyword.lower() in state.lower() for keyword in completion_keywords)
    
    # Check for loading/waiting states
    loading_keywords = ['loading', 'pending', 'waiting', 'processing', 'submitting']
    is_loading = any(keyword.lower() in state.lower() for keyword in loading_keywords)
    
    # Adjust base value based on completion indicators
    if completion_state and not is_loading:
        base_value = min(base_value * 1.2, 0.95)
    
    # Adjust for loading states (slightly lower value)
    if is_loading:
        base_value *= 0.8
    
    # Add small bonus for having many interactive options (exploration potential)
    if action_score > 20:
        base_value = min(base_value + 0.05, 0.98)
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, base_value))