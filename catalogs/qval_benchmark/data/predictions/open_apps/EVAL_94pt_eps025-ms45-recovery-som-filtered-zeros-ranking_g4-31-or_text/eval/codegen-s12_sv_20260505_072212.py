import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the OpenApps environment.
    The value is an approximation of the probability of achieving the goal 
    from the current state, reflecting progress and proximity to the target state.
    """
    # Extract the goal description if it exists in the state string
    goal_match = re.search(r"(?:Goal|Task):\s*(.*)", state, re.IGNORECASE)
    goal = goal_match.group(1).lower() if goal_match else ""
    
    # Separate the environment content (AX-tree/screenshot text) from the goal
    content = state
    if goal_match:
        content = state[goal_match.end():]

    # 1. Terminal Success Markers (V = 1.0)
    # Look for phrases that strongly indicate the task has been completed.
    success_patterns = [
        r"\bsuccessfully\b", 
        r"has been (?:created|added|sent|saved|deleted|removed)", 
        r"was successfully", 
        r"\bconfirmation\b", 
        r"\bcompleted\b", 
        r"\bsuccess\b"
    ]
    for pattern in success_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            # Ensure it's not just echoing the goal
            # (Though in these environments, success messages usually appear after actions)
            return 1.0

    # 2. Final Action Markers (V = 0.7 - 0.9)
    # Look for interactive elements that typically finalize a process.
    # We distinguish between 'finalizing' actions and 'initiating' actions.
    finalizing_actions = ["save", "submit", "confirm", "send", "post", "update"]
    for action in finalizing_actions:
        # Search for the action word within a button or link tag/label
        if re.search(rf'(?:button|link|click).*?{action}', content, re.IGNORECASE):
            # Higher value if the action keyword is actually in the goal
            if goal and action in goal:
                return 0.9
            return 0.7

    # 3. Progress/On-Track Markers (V = 0.3 - 0.6)
    # Look for 'initiating' actions (like "Add" or "New") or goal-relevant keywords.
    initiating_actions = ["add", "create", "new", "edit", "delete", "remove", "compose"]
    for action in initiating_actions:
        if re.search(rf'(?:button|link|click).*?{action}', content, re.IGNORECASE):
            if goal and action in goal:
                return 0.6
            return 0.4

    # 4. Contextual Relevance (V = 0.1 - 0.3)
    # Check if keywords from the goal are present in the page content.
    if goal:
        # Filter out common stop words to get meaningful goal keywords
        stop_words = {"a", "an", "the", "to", "in", "of", "at", "for", "with", "and", "or", "is"}
        goal_words = [w for w in re.findall(r'\w+', goal) if w not in stop_words]
        
        matches = 0
        for word in goal_words:
            if word in content.lower():
                matches += 1
        
        # Scale value based on how many goal keywords are visible on the page
        if matches >= 3:
            return 0.3
        elif matches >= 1:
            return 0.15

    # 5. Baseline Value (V = 0.05)
    # Default value for states that don't show clear progress toward the goal.
    return 0.05