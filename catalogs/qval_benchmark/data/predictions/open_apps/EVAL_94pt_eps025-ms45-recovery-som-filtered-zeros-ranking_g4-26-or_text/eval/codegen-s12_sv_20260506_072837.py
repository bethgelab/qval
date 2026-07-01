import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment.
    The value represents the likelihood of achieving the task goal.
    """
    s = state.lower()
    
    # 1. Check for immediate Error/Failure states
    # If error messages are present, the value is likely 0.
    if any(x in s for x in ["error", "invalid", "failed", "not found", "please enter", "required field"]):
        return 0.0
        
    # 2. Check for immediate Success states
    # We look for success indicators that are likely part of the page text 
    # rather than the text of a button (e.g., "Task completed" vs a "Done" button).
    success_indicators = ["success", "completed", "done", "confirmed", "sent", "saved", "added", "created"]
    for word in success_indicators:
        if word in s:
            # Heuristic: If the word is preceded by 'button' within a short distance,
            # it's likely a button and not a confirmation message.
            # The regex looks for 'button' followed by non-letters (like quotes/colons) and the word.
            if not re.search(rf'button\s+[^a-z]{{0,15}}{word}', s):
                return 1.0

    # 3. Initialize score for progressive estimation
    # 0.1 is our baseline for any non-error, non-success state.
    score = 0.1
    
    # 4. Analyze element density and interaction readiness
    # We look for common accessibility tree markers for input and action elements.
    # Count elements that are filled (value="...")
    filled_inputs_count = len(re.findall(r'value="[^"]+"', s))
    # Count input-like elements
    input_elements_count = len(re.findall(r'input|textbox|text field', s))
    # Count buttons
    button_count = len(re.findall(r'button', s))
    
    # Increase score based on presence of input fields
    if input_elements_count > 0:
        score += 0.15
        
    # Increase score based on how much information has been provided
    if filled_inputs_count > 0:
        # We cap the benefit of filled inputs to prevent score explosion
        score += 0.25 * min(filled_inputs_count, 2)
        
    # 5. Check for "Submit Readiness" (The most critical predictor of success)
    # If a form is filled and a "submit-style" button is available, the state is very high value.
    submit_actions = ["send", "save", "submit", "add", "create", "confirm", "schedule", "go", "search"]
    has_submit_button = False
    for action in submit_actions:
        # Look for a button element that contains a submission-related keyword
        if re.search(rf'button.*?{action}', s):
            has_submit_button = True
            break
            
    if has_submit_button:
        score += 0.2
        if filled_inputs_count > 0:
            # If the form is also filled, we are likely one click away from the goal.
            score += 0.35
    elif button_count > 0:
        # Presence of buttons in general suggests we are in an actionable part of a task.
        score += 0.1
            
    # 6. Contextual validation (App identification)
    # Certain apps are more "ready" if their specific keywords are present.
    app_keywords = {
        "messenger": ["chat", "message", "send", "recipient", "conversation"],
        "todo": ["task", "todo", "list", "item"],
        "calendar": ["event", "calendar", "date", "schedule", "appointment"],
        "maps": ["map", "direction", "location", "search", "navigate"],
        "code": ["editor", "code", "file", "syntax", "run"]
    }
    
    for app, keywords in app_keywords.items():
        if app in s:
            # Count how many relevant keywords for this app are found
            matches = sum(1 for kw in keywords if kw in s)
            if matches > 0:
                score += 0.05 * matches
            break
            
    # 7. Final normalization
    # Ensure the returned value is strictly within [0.0, 1.0]
    return min(max(score, 0.0), 1.0)