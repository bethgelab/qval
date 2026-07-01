import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment.
    The value represents the expected discounted cumulative reward,
    which is binary (1.0 for goal achievement, 0.0 otherwise).
    """
    state_lower = state.lower()
    
    # Attempt to separate the goal description from the actual page content
    # The 'state' string in BrowserGym/OpenApps often starts with a "Goal: ..." section.
    goal_match = re.search(r"goal:\s*(.*)", state, re.IGNORECASE)
    if goal_match:
        # Extract the first line as the goal text
        goal_text = goal_match.group(1).split('\n')[0].lower()
        # Define content as everything following the goal description to avoid self-matching
        content_start = state_lower.find(goal_text) + len(goal_text)
        content = state_lower[content_start:]
    else:
        goal_text = ""
        content = state_lower

    # 1. Success Signals: High confidence that the goal has been achieved.
    # These keywords usually appear in confirmation messages or success alerts.
    success_markers = [
        "successfully", "completed", "confirmed", "sent", 
        "created", "saved", "done", "success", "correctly"
    ]
    if any(marker in content for marker in success_markers):
        return 0.98
    
    # 2. Progress Signals: Goal-related keywords appearing in the accessibility tree.
    # If the current page contains terms from the goal, the agent is likely on the correct path.
    if goal_text:
        stop_words = {
            "a", "an", "the", "to", "in", "on", "at", "of", 
            "and", "with", "for", "is", "please", "my", "your"
        }
        # Extract meaningful keywords from the goal
        goal_words = [w for w in re.findall(r'\w+', goal_text) if w not in stop_words]
        
        # Calculate overlap between goal keywords and page content
        overlap = sum(1 for w in goal_words if w in content)
        if overlap >= 3: 
            return 0.8  # Strong indicator of being on the target page/action
        if overlap >= 1: 
            return 0.5  # Moderate indicator of being in the right section
            
    # 3. Interaction Signals: Presence of interactive elements used for completing tasks.
    # Being in a form-filling or submission stage is a mid-stage progress indicator.
    action_markers = [
        "submit", "save", "create", "add", "send", 
        "input", "textbox", "dropdown", "button", "click here"
    ]
    if any(marker in content for marker in action_markers):
        return 0.3
        
    # 4. Baseline Signals: Indicators of being at the start or on a generic page.
    home_markers = ["home", "dashboard", "main", "welcome", "index", "landing"]
    if any(marker in content for marker in home_markers):
        return 0.2
        
    # Default value for states with no strong predictive features
    return 0.1