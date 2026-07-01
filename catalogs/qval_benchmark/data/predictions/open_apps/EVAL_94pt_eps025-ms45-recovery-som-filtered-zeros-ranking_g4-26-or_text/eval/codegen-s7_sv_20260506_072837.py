import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given accessibility tree state string.
    The value represents the likelihood of achieving the task goal.
    """
    # Base score
    score = 0.1
    
    # 1. Interactive Elements Analysis
    # Identifying elements the agent can interact with to progress.
    interactive_roles = r'button|link|textbox|checkbox|menuitem|combobox|input|textarea|edittext'
    interactives = re.findall(rf'role="({interactive_roles})"', state, re.I)
    num_interactives = len(interactives)
    
    if num_interactives > 0:
        score += 0.15
    
    # 2. Progress via Data Entry
    # Finding filled-in fields (e.g., value="text") suggests the agent is mid-task.
    # We search for non-empty, non-whitespace values.
    filled_fields = re.findall(r'value="([^"\s][^"]*)"', state)
    num_filled = len(filled_fields)
    
    if num_filled > 0:
        # Incremental bonus for more filled information
        score += 0.25 + (0.1 * min(2.0, num_filled))
        
    # 3. Success Indicators (Goal achieved or near achievement)
    # We check for keywords indicating completion.
    # We attempt to distinguish between a "success" status and a "Send" button label.
    success_words = ['success', 'completed', 'sent', 'added', 'saved', 'created', 'scheduled', 'done', 'finished', 'confirmed']
    for word in success_words:
        if re.search(rf'\b{word}\b', state, re.I):
            # If the word is likely an attribute of an interactive element (like a button name), 
            # it's likely an action to be taken, not a result.
            is_action_button = re.search(rf'role="(button|link)"[^>]*?\b{word}\b', state, re.I)
            if not is_action_button:
                score += 0.5
                break
                
    # 4. Error/Obstacle Indicators (Task blocked)
    error_words = ['error', 'failed', 'invalid', 'wrong', 'required', 'denied', 'not found']
    for word in error_words:
        if re.search(rf'\b{word}\b', state, re.I):
            score -= 0.4
            break
            
    # 5. Early-Stage/Setup Indicators (Starting phase)
    # Being on a login or credential screen usually means the task hasn't progressed significantly.
    start_words = ['login', 'sign in', 'sign-in', 'username', 'password', 'credentials']
    for word in start_words:
        if re.search(rf'\b{word}\b', state, re.I):
            score -= 0.2
            break

    # 6. Terminal/Dead-end Check
    # If there are no interactive elements and no success message, it's a failed or stuck state.
    if num_interactives == 0 and not re.search(r'\b(success|completed|sent|added|saved|done)\b', state, re.I):
        score -= 0.3

    # Normalize the score within [0.0, 1.0]
    final_value = max(0.0, min(1.0, score))
    
    return float(final_value)