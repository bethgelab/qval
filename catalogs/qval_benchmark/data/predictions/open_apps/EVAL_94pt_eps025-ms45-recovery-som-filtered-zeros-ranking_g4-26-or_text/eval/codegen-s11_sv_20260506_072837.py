import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) based on the text representation of the current state.
    V(s) approximates the expected discounted reward, where success is 1.0 and failure is 0.0.
    """
    s = state.lower()
    
    # 1. Success Check (Highest Priority)
    # If the state indicates the objective has been completed, return 1.0.
    # These markers are common in successful terminal states across the OpenApps suite.
    success_indicators = [
        "success", "completed", "confirmed", "done", 
        "message sent", "event added", "task added", 
        "saved successfully", "item added", "route found", "target reached"
    ]
    if any(ind in s for ind in success_indicators):
        return 1.0

    # 2. Error/Failure Check
    # If the state indicates an error or a dead end, return a very low value.
    error_indicators = ["error", "invalid", "failed", "could not find", "not found", "try again", "no results"]
    if any(ind in s for ind in error_indicators):
        return 0.05

    # 3. Feature Extraction (Interactivity and Progress)
    # Identify the density of interactive elements.
    # Accessibility trees typically use roles or tags like 'button', 'input', or 'link'.
    button_count = len(re.findall(r"button", s))
    input_count = len(re.findall(r"input|textbox|text field|edit|role=\"textbox\"", s))
    link_count = len(re.findall(r"link|a href|role=\"link\"", s))
    
    # Check for evidence of progress (non-empty input values).
    # In accessibility trees, 'value="something"' implies data has been entered.
    # We use [^"]+ to ensure the value is not an empty string.
    has_filled_data = bool(re.search(r'value="[^"]+"', s))

    # 4. Heuristic Scoring
    # We build a score starting from a baseline, representing the potential for task completion.
    score = 0.1
    
    # Add points for the presence of interactive elements.
    # Buttons and inputs are more critical for task progression than links.
    score += min(button_count * 0.12, 0.35)
    score += min(input_count * 0.12, 0.35)
    score += min(link_count * 0.05, 0.1)
    
    # If data has been filled into a field, we are likely mid-task.
    if has_filled_data:
        score += 0.25

    # 5. Contextual Task-Phase Boosts
    # If the state has both inputs and buttons, it's likely a high-engagement form state.
    if input_count > 0 and button_count > 0:
        score += 0.3
    # If there's data and buttons, we are likely at the "finalizing" or "submitting" phase.
    elif button_count > 0 and has_filled_data:
        score += 0.2
    # If there are inputs present, we are at least in an active entry state.
    elif input_count > 0:
        score += 0.15

    # 6. Finalization
    # Cap the value at 0.95 to leave room for the exact 1.0 success state.
    # Return the score rounded to two decimal places for stability.
    final_score = min(0.95, score)
    return max(0.0, round(final_score, 2))