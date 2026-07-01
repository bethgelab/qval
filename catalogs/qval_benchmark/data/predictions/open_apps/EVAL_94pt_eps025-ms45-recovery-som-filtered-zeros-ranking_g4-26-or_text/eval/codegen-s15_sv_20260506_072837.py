import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) based on the accessibility tree representation.
    Since the goal is not explicitly provided, the function uses heuristics to 
    identify success, failure, and interaction potential.
    """
    # Normalize the input text for easier pattern matching
    text = state.lower()
    
    # 1. Terminal State Detection: Success
    # If these semantic markers are found, the task-level goal has likely been achieved.
    # In synthetic web apps, success is usually signaled by explicit confirmation text.
    success_patterns = [
        'success', 'sent', 'added', 'created', 'saved', 
        'updated', 'done', 'finished', 'completed',
        'message sent', 'event created', 'task added', 'task completed'
    ]
    if any(pattern in text for pattern in success_patterns):
        return 1.0
        
    # 2. Terminal State Detection: Failure
    # These keywords suggest the agent has reached a dead-end or encountered an error.
    error_patterns = [
        'error', 'failed', 'not found', 'invalid', 
        'incorrect', 'try again', 'no results', 'not possible'
    ]
    if any(pattern in text for pattern in error_patterns):
        return 0.0
        
    # 3. Progress Potential (Interactivity)
    # We measure the diversity of the interaction layer. A state with a variety 
    # of interactive elements is more promising for task completion than a static state.
    interactive_indicators = [
        'button', 'input', 'link', 'textarea', 'select', 
        'checkbox', 'radio', 'editable', 'role="button"', 
        'role="link"', 'aria-label', 'aria-expanded', 'aria-checked'
    ]
    
    found_interactive_types = 0
    for indicator in interactive_indicators:
        if indicator in text:
            found_interactive_types += 1
            
    # 4. Contextual Relevance
    # Presence of app-specific context keywords indicates the agent is in a functional area.
    context_keywords = [
        'todo', 'calendar', 'messenger', 'map', 'code', 
        'editor', 'search', 'nav', 'menu', 'dashboard'
    ]
    has_context = any(keyword in text for keyword in context_keywords)
    
    # 5. Heuristic Value Calculation
    # We construct the value as a probability estimate [0.0, 1.0].
    # - A base value of 0.1 represents an ongoing, valid episode.
    # - Interactivity provides a boost (up to 0.4) based on the variety of available controls.
    # - Context presence provides a small boost (0.1).
    # This ensures that non-terminal, highly interactive states are valued higher.
    
    value = 0.1
    # Cap interactivity boost to 0.4
    interactivity_boost = min(0.4, found_interactive_types * 0.06)
    value += interactivity_boost
    
    if has_context:
        value += 0.1
        
    # Ensure the returned value is clamped within the valid range [0.0, 1.0]
    return max(0.0, min(1.0, value))