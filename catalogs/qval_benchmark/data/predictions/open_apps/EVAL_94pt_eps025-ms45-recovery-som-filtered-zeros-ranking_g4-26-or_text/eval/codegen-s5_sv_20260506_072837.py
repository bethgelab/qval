import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given state in the OpenApps environment.
    The value is higher for states that are closer to completing a task (e.g., presence 
    of forms, action buttons, or correct application context).
    """
    s = state.lower()
    
    # 1. Terminal Success Check
    # If the state shows that the task goal has already been met.
    success_indicators = [
        "success", "sent", "added", "created", "saved", "done", 
        "completed", "task added", "message sent", "event created",
        "message delivered", "location found", "entry saved", "task completed",
        "event added", "contact added"
    ]
    if any(indicator in s for indicator in success_indicators):
        return 1.0

    # 2. Feature Detection
    # Input elements: common indicators of a form-filling task.
    input_terms = ["input", "textarea", "role=\"textbox\"", "type=\"text\"", "role=\"combobox\"", "contenteditable"]
    has_input = any(term in s for term in input_terms)
    
    # Button elements: common indicators of an actionable state.
    button_terms = ["button", "role=\"button\"", "role=\"link\"", "clickable"]
    has_button = any(term in s for term in button_terms)
    
    # Semantic cues: words that describe the intended action of the UI.
    action_terms = ["send", "add", "save", "submit", "create", "delete", "edit", "search", "go", "apply", "confirm"]
    has_action = any(term in s for term in action_terms)
    
    # Negative/Cancellation cues: words that suggest undoing or leaving the current task.
    neg_terms = ["cancel", "back", "close", "reset", "clear", "remove", "dismiss"]
    has_neg = any(term in s for term in neg_terms)
    
    # Error cues: words indicating a failure state.
    err_terms = ["error", "invalid", "failed", "not found", "incorrect", "try again", "alert"]
    has_err = any(term in s for term in err_terms)
    
    # Application context: identifying which app the agent is currently in.
    app_context = ["todo", "calendar", "messenger", "map", "code", "editor"]
    in_app = any(term in s for term in app_context)

    # 3. Heuristic Scoring Logic
    # We build a score based on the proximity to a goal-completing interaction.
    if has_input and has_button:
        # A state with both an input field and a button is very likely a form.
        if has_action and not has_neg:
            # High readiness: a form with an action button (e.g., "Send").
            score = 0.8
        elif has_neg:
            # Medium readiness: a form with a cancellation option.
            score = 0.4
        else:
            # Medium readiness: a form where the button's purpose is unclear.
            score = 0.5
    elif has_input:
        # Moderate readiness: an input field is present, but no button is visible.
        score = 0.3
    elif has_button:
        # Low readiness: an action button is present, but no input field.
        score = 0.2
    elif in_app:
        # Low readiness: agent has reached the correct app but hasn't found the task interface.
        score = 0.15
    else:
        # Very low readiness: general navigation or starting state.
        score = 0.05

    # 4. Refinement
    # Boost the score if we are in the correct app context.
    if in_app:
        score += 0.1
    
    # Penalize the score if there are errors present.
    if has_err:
        score -= 0.3
        
    # Clamp the result between 0.0 and 0.95.
    # We cap at 0.95 for non-terminal states to maintain a distinction from a true success (1.0).
    return max(0.0, min(0.95, score))