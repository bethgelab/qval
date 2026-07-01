import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for an OpenApps episode based on direct analysis of the state text.
    
    The value is estimated by checking:
    1. If the goal has already been achieved (binary reward 1.0).
    2. If the state contains error messages or dead ends (penalty).
    3. The progress towards the goal based on keywords related to the specific app and task type.
    4. A baseline value that decays slightly with the number of steps if the state is ambiguous, 
       though we don't have step count here, so we rely on the "closeness" of keywords.
    """
    
    # Normalize state for analysis
    state_lower = state.lower()
    
    # 1. Check for Goal Achievement
    # The reward is 1.0 if the goal is achieved. We look for success indicators.
    # Common success phrases in synthetic tasks might be "Task completed", "Event added", "Message sent", etc.
    success_patterns = [
        "task completed", "goal achieved", "success", "added", "sent", "created", 
        "saved", "confirmed", "event added", "message sent", "todo added"
    ]
    for pattern in success_patterns:
        if pattern in state_lower:
            return 1.0
            
    # 2. Check for Failure/Dead Ends
    # Error messages, "page not found", "invalid input", "blocked", etc.
    failure_patterns = [
        "error", "failed", "not found", "invalid", "blocked", "permission denied",
        "404", "exception", "timeout", "cannot proceed"
    ]
    for pattern in failure_patterns:
        if pattern in state_lower:
            return 0.0
            
    # 3. Estimate Progress based on App and Task Context
    # We assign higher values if the state contains keywords related to the specific goal steps.
    # Since the specific goal isn't passed, we infer from the presence of UI elements related to common tasks.
    
    progress_score = 0.0
    
    # General progress indicators (forms filled, buttons clicked)
    if "filled" in state_lower or "entered" in state_lower:
        progress_score += 0.2
    if "selected" in state_lower or "chosen" in state_lower:
        progress_score += 0.15
    if "dialog" in state_lower or "modal" in state_lower:
        # Being in a modal often means we are close to completing a sub-task
        progress_score += 0.1
        
    # App-specific heuristics
    # Todo: Look for "add", "task", "list"
    if "todo" in state_lower or "task" in state_lower:
        if "add" in state_lower or "create" in state_lower:
            progress_score += 0.3
        if "delete" in state_lower or "remove" in state_lower:
            progress_score += 0.1
            
    # Calendar: Look for "event", "calendar", "date"
    if "calendar" in state_lower or "event" in state_lower:
        if "add" in state_lower or "create" in state_lower or "new" in state_lower:
            progress_score += 0.3
        if "save" in state_lower or "confirm" in state_lower:
            progress_score += 0.2
            
    # Messenger: Look for "message", "chat", "send"
    if "message" in state_lower or "chat" in state_lower:
        if "send" in state_lower or "reply" in state_lower:
            progress_score += 0.3
        if "compose" in state_lower:
            progress_score += 0.15
            
    # Maps: Look for "map", "location", "route"
    if "map" in state_lower or "location" in state_lower:
        if "route" in state_lower or "direction" in state_lower:
            progress_score += 0.3
        if "search" in state_lower:
            progress_score += 0.2
            
    # Code Editor: Look for "code", "editor", "run", "save"
    if "code" in state_lower or "editor" in state_lower:
        if "run" in state_lower or "execute" in state_lower:
            progress_score += 0.3
        if "save" in state_lower:
            progress_score += 0.2
            
    # 4. Baseline and Clamping
    # If we found some progress indicators, we return a value between 0 and 1.
    # If no specific indicators are found, we give a small positive baseline (exploration bias) 
    # or 0 if it looks like a blank/initial state.
    
    # Check for initial state indicators (often low value)
    initial_indicators = ["welcome", "login", "home", "start", "new"]
    if all(ind not in state_lower for ind in initial_indicators) and progress_score == 0.0:
        # If we are not at home/login and have no progress, maybe we are in a complex state
        # but without specific cues, we stay neutral.
        progress_score = 0.05 # Small baseline
    elif progress_score == 0.0:
        progress_score = 0.0
        
    # Cap the value at 0.99 (since 1.0 is reserved for confirmed success)
    estimated_value = min(0.99, progress_score)
    
    return estimated_value