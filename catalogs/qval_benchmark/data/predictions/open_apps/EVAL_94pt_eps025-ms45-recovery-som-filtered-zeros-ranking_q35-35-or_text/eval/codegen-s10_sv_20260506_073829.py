import re
from collections import Counter

def signal_function(state: str) -> float:
    """
    Estimates the state-value for an OpenApps episode based on direct analysis of the state string.
    
    The value is a float between 0.0 and 1.0, representing the estimated probability of achieving
    the goal from the current state, assuming optimal play.
    
    Heuristics used:
    1. Goal Achievement: If the state explicitly indicates the goal is met (e.g., "Success", "Goal achieved"), return 1.0.
    2. Progress Indicators: Look for keywords related to the specific app goals (e.g., "Event added", "Message sent", "Task completed").
    3. Error States: Detect error messages or broken flows (e.g., "Error", "Failed", "Invalid") which significantly lower value.
    4. Navigation Depth: Penalize states that seem stuck in deep navigation loops or require excessive scrolling without progress.
    5. Form Completion: For tasks involving forms, check if required fields are indicated as filled vs empty.
    """
    
    # Normalize state for analysis
    state_lower = state.lower()
    
    # 1. Check for immediate goal success
    success_patterns = [
        "goal achieved", "success", "task completed", "event created", 
        "message sent", "task added", "appointment scheduled", "code saved"
    ]
    for pattern in success_patterns:
        if pattern in state_lower:
            return 1.0
    
    # 2. Check for explicit failure or error states
    error_patterns = [
        "error", "failed", "invalid", "not found", "unauthorized", 
        "timeout", "connection lost", "404", "403"
    ]
    error_count = sum(1 for p in error_patterns if p in state_lower)
    if error_count >= 2:
        return 0.0
    elif error_count == 1:
        # Single error might be recoverable, but value is low
        return 0.2
    
    # 3. Analyze progress based on app context
    # We look for "filled" or "completed" indicators in form fields
    # In accessibility trees, this often appears as "checkbox checked" or "input filled"
    progress_indicators = [
        "checked", "filled", "selected", "completed", "saved", "confirmed"
    ]
    progress_count = sum(1 for p in progress_indicators if p in state_lower)
    
    # Base value increases with progress indicators found
    if progress_count > 0:
        # Exponential decay of uncertainty as progress is made
        # Cap at 0.9 to leave room for final confirmation step
        return min(0.9, 0.3 + (progress_count * 0.15))
    
    # 4. Check for navigation depth/stuckness
    # If the state contains many "button" or "link" tags but no specific content, 
    # it might be a landing page or stuck.
    # Heuristic: If state is very long but lacks content keywords, it might be a deep menu.
    word_count = len(state.split())
    if word_count > 500 and progress_count == 0:
        # Might be a deep menu or complex page without progress
        return 0.1
    
    # 5. Default fallback
    # If no specific signals are found, assume a neutral starting position
    # This is a heuristic estimate for an unobserved intermediate state
    return 0.4