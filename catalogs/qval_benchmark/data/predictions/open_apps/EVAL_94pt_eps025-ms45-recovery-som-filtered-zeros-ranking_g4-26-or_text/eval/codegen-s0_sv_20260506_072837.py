import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) by analyzing the accessibility tree text.
    The heuristic identifies progress toward goal completion by detecting 
    interaction-related elements (inputs, buttons), task-specific keywords 
    (form fields), and completion markers.
    """
    s = state.lower()
    
    # 1. Terminal state detection
    # If the state contains indicators of successful completion, return 1.0 immediately.
    if re.search(r'\b(success|sent|added|created|saved|done|completed|confirmed)\b', s):
        return 1.0
    
    # 2. Feature extraction
    # Count interactive elements and keywords using word boundaries to avoid false positives.
    n_bids = len(re.findall(r'bid', s))
    n_input = len(re.findall(r'\b(input|textarea|text|textbox|search|field)\b', s))
    n_button = len(re.findall(r'\b(button|link|checkbox|radio)\b', s))
    n_action = len(re.findall(r'\b(send|add|save|submit|create|edit|search|delete|apply|ok|go)\b', s))
    n_form = len(re.findall(r'\b(message|subject|name|event|title|date|todo|task|calendar|email|address|description)\b', s))
    
    # 3. Scoring Heuristic
    # We build a score based on the "completeness" or "actionability" of the current view.
    score = 0.0
    
    # Base value for being in an interactive state
    if n_bids > 0:
        score += 0.1
        
    # Progress indicator: The presence of an input field combined with an 
    # action verb (like 'submit' or 'add') suggests the agent is deep in a task.
    if n_input > 0 and n_action > 0:
        score += 0.4
    elif n_input > 0:
        score += 0.2
    elif n_action > 0:
        score += 0.1
        
    # Contextual/Navigation indicators
    # High presence of form-related words increases confidence in the task context.
    if n_form > 0:
        score += 0.2
    
    # High presence of clickable elements (buttons/links) increases actionability.
    if n_button > 0:
        score += 0.1
        
    # Density bonus: A page with many interactive bids is likely a functional app page.
    if n_bids > 5:
        score += 0.1
        
    # 4. Final value clamping
    # We clamp the score to [0.0, 0.95]. 
    # 1.0 is reserved for states where the goal is explicitly achieved.
    return min(max(score, 0.0), 0.95)