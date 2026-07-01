import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) by analyzing the accessibility tree text.
    The value represents the likelihood of reaching the task goal from the current state.
    """
    if not state or len(state.strip()) < 5:
        return 0.0
        
    s = state.lower()
    
    # 1. Terminal Success Indicators
    # If the state contains words implying the goal is reached, return maximum value.
    success_terms = {'success', 'completed', 'sent', 'saved', 'added', 'created', 'deleted', 'done', 'finished', 'confirmed'}
    if any(term in s for term in success_terms):
        return 1.0
        
    # 2. Feature Extraction
    # Count interactive elements and action-oriented keywords using regex.
    # These are proxies for the agent's ability to interact with the environment.
    input_elements = len(re.findall(r'input|textbox|textarea|select|checkbox|edit|field', s))
    action_elements = len(re.findall(r'button|link|click|menu|tab|menuitem|icon', s))
    action_verbs = len(re.findall(r'send|save|add|create|delete|search|edit|open|type|submit|go|nav', s))
    
    # 3. Heuristic Scoring
    # Start with a low baseline score.
    score = 0.1
    
    # If interactive elements are found, the state is "actionable".
    if input_elements > 0 or action_elements > 0:
        # If both inputs and action buttons are present, it's highly likely a functional form.
        if input_elements > 0 and action_elements > 0:
            score += 0.4
        else:
            score += 0.2
        
        # Add incremental value based on the density of interactive elements and verbs.
        score += min(0.1 * input_elements, 0.2)
        score += min(0.1 * action_elements, 0.2)
        score += min(0.1 * action_verbs, 0.2)
        
    # 4. Contextual Progress Indicators
    # Look for patterns like "new event", "add task", or "send message" which indicate 
    # the user is currently in the middle of a workflow.
    if re.search(r'(new|add|create|delete|send|search)\s+\w+', s):
        score += 0.15
        
    # 5. Complexity/Informational density adjustment
    # If the text is very long but contains no interactive elements, it might be a 
    # static info page which is less useful for achieving active tasks.
    if input_elements == 0 and action_elements == 0 and len(s) > 1000:
        score *= 0.5

    # Final value clamping to stay within the expected range (0.0 to 0.95).
    # 1.0 is reserved exclusively for the terminal success state.
    return min(0.95, max(0.0, score))