import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given state in the OpenApps environment.
    The value represents the expected discounted cumulative reward, which is 1.0 
    upon task completion and 0.0 otherwise.
    """
    # 1. Extract Goal from the state if possible
    # The goal is often provided in the text preceding the accessibility tree.
    goal = ""
    goal_match = re.search(r"Goal:\s*(.*?)(?:\n|Task:|$)", state, re.I)
    if goal_match:
        goal = goal_match.group(1).lower()

    # 2. Check for Goal Completion (Terminal State Estimation)
    # If the state indicates the task has been completed, the value is 1.0.
    if goal:
        # Map verbs to their past tense equivalents for completion detection
        transitions = {
            "send": "sent",
            "add": "added",
            "create": "created",
            "schedule": "scheduled",
            "delete": "deleted",
            "save": "saved",
            "write": "written",
            "type": "typed",
            "search": "searched",
            "make": "made"
        }
        
        for verb, past in transitions.items():
            if verb in goal:
                # If the past tense of the goal verb appears, we are likely at the goal
                if re.search(rf"\b{past}\b", state, re.I):
                    return 1.0
        
        # General success keyword check
        success_keywords = [r"\bsuccess\b", r"\bcompleted\b", r"\bconfirmed\b", r"\bdone\b"]
        if any(re.search(pattern, state, re.I) for pattern in success_keywords):
            # We return a value close to 1.0; in RL, terminal state value is the reward.
            return 0.95

    # 3. Initialize Score and App Detection
    score = 0.1  # Base value for any non-empty state
    
    # Define app-specific context to check if the agent is in the right environment
    apps = {
        'todo': ['todo', 'task', 'checklist', 'item'],
        'calendar': ['calendar', 'event', 'date', 'schedule', 'time'],
        'messenger': ['messenger', 'message', 'chat', 'send', 'contact'],
        'maps': ['maps', 'location', 'address', 'navigate', 'search'],
        'code': ['code', 'editor', 'file', 'script', 'python']
    }
    
    current_app = None
    for app, keywords in apps.items():
        if any(k in state.lower() for k in keywords):
            current_app = app
            break
            
    # 4. Scoring based on Contextual Alignment
    if goal:
        goal_app = None
        for app, keywords in apps.items():
            if any(k in goal for k in keywords):
                goal_app = app
                break
        
        if goal_app:
            if current_app == goal_app:
                score += 0.3  # High alignment: in the correct application
            else:
                score += 0.0  # Low alignment: in the wrong application
        else:
            score += 0.1  # Minimal boost if a goal exists but app isn't identified
    
    # 5. Scoring based on Actionability (Proximity to Goal)
    # Check for presence of interactive elements (buttons, inputs, etc.)
    interactive_elements = re.findall(r"(button|textbox|link|input|checkbox|menuitem|textarea)", state, re.I)
    if interactive_elements:
        # More actionable elements increase the probability of being able to proceed
        score += min(len(interactive_elements) * 0.05, 0.4)

    # Check if elements related to specific goal actions are visible
    if goal:
        action_keywords = ["send", "add", "create", "save", "search", "schedule", "delete", "click", "type"]
        for word in action_keywords:
            if word in goal:
                # If the specific action required is visible in the text, boost the score
                if re.search(rf"\b{word}\b", state, re.I):
                    score += 0.2
                    break

    # 6. Ensure the final estimate is within the valid [0.0, 1.0] range
    return min(max(score, 0.0), 1.0)