import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) by analyzing the accessibility tree text.
    The value is higher when the state shows progress towards completing a task,
    such as having filled-in forms, open dialogs, or available action buttons.
    """
    s = state.lower()
    
    # 1. Terminal Success Check
    # If the goal has been achieved, the reward is 1.0.
    # We look for indicators of task completion common in these synthetic apps.
    success_indicators = [
        r'successfully', r'completed', r'done', r'sent', r'saved', 
        r'created', r'added', r'message sent', r'event created', 
        r'task added', r'all set'
    ]
    for pattern in success_indicators:
        if re.search(pattern, s):
            # Avoid false positives like "not successfully"
            if not re.search(r'not (successfully|done|completed|sent)', s):
                return 1.0
                
    # 2. Error/Failure Check
    # Errors typically signal a step backward or a failed attempt, reducing the expected reward.
    error_indicators = [r'error', r'failed', r'invalid', r'not found', r'incorrect', r'try again', r'alert']
    if any(re.search(pattern, s) for pattern in error_indicators):
        # If an error is present, the value is significantly reduced.
        # However, we don't return 0 immediately in case the error is minor or recoverable.
        return 0.05

    # 3. Heuristic Progress Scoring
    score = 0.0
    
    # Identify interactive element counts
    # These provide a baseline for whether the state is "active" or "dead".
    inputs = re.findall(r'input', s)
    buttons = re.findall(r'button', s)
    links = re.findall(r'link|a\s', s)
    
    if not (inputs or buttons or links):
        return 0.01  # Very low value if no interaction is possible
        
    # Base score for existence of elements
    score += len(inputs) * 0.05
    score += len(buttons) * 0.05
    
    # 4. Progress via Input Content
    # A key indicator of progress is seeing text already populated in fields.
    # We look for common attributes like value="...", text="...", or name="..."
    filled_content = re.findall(r'(?:value|text|name)="([^"]+)"', s)
    for content in filled_content:
        content = content.lower().strip()
        # Ignore generic placeholders and empty values
        if content and content not in [
            '', 'type here', 'search', 'enter text', 'none', 'null', 
            'select...', 'none', 'input', 'placeholder'
        ]:
            score += 0.25
    
    # 5. Proximity to Action (Submit/Send/Save buttons)
    # Being on a page with a "Submit" or "Send" button suggests we are close to the terminal state.
    action_words = ['send', 'save', 'submit', 'create', 'add', 'confirm', 'done', 'search', 'post', 'apply']
    for word in action_words:
        # Search for the action word specifically in a context that looks like a button label
        # In the accessibility tree, this often looks like 'button: Send' or 'role="button" name="Send"'
        if re.search(rf'\b{word}\b', s):
            # We check if the word is near a button/link keyword to increase confidence
            if re.search(rf'{word}.*?button|button.*?{word}|{word}.*?link|link.*?{word}', s):
                score += 0.45
            else:
                score += 0.15
                
    # 6. Contextual Clues (Modals/Dialogs)
    # A modal or dialog often indicates a specific task-related interaction is in progress.
    if any(x in s for x in ['dialog', 'modal', 'popup', 'overlay']):
        score += 0.2
            
    # 7. Final Normalization and Clipping
    # The heuristic score is mapped to the range [0.0, 1.0].
    # We use a clamp to ensure valid return values.
    final_val = max(0.0, min(1.0, score))
    
    # Ensure a minimum floor for states that have some interactive potential
    if final_val < 0.1 and (inputs or buttons):
        return 0.05
    elif final_val < 0.1:
        return 0.0
        
    return final_val