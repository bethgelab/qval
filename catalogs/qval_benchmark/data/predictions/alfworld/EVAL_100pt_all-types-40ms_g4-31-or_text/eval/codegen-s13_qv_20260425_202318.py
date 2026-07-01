def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an ALFWorld environment.
    The value is based on a heuristic of progress towards the goal, where higher values 
    represent actions that are more likely to be part of an optimal path to task completion.
    """
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()

    # 1. Terminal Success: The most valuable state.
    if "task completed" in ns or "successfully completed" in ns:
        return 1.0
    
    # 2. Failure or Redundancy: Low value for actions that don't change the state meaningfully
    # or are explicitly rejected by the environment.
    failure_keywords = ["cannot", "nothing happens", "already", "invalid", "unable", "don't see"]
    if any(word in ns for word in failure_keywords):
        return 0.0

    # 3. High-Value Progress: Actions that typically signal the final or near-final step.
    # Placing an object or cleaning an item are often the goal-achieving actions.
    if ("put" in a or "place" in a) and ("put" in ns or "placed" in ns):
        return 0.9
    
    if "clean" in a and ("cleaned" in ns or "is now clean" in ns):
        return 0.9

    # 4. Mid-Value Progress: Acquiring the target object is a critical milestone.
    if ("take" in a or "pick up" in a) and ("holding" in ns or "took" in ns or "picked up" in ns):
        return 0.7
    
    # 5. Preparatory Progress: Opening containers to access items.
    if "open" in a and ("opened" in ns or "is now open" in ns):
        return 0.5
        
    # 6. Low-Value Progress: Navigation.
    # Moving to a new room is necessary but generally further from the final reward.
    if ("go to" in a or "move to" in a) and ("you are in" in ns):
        if s != ns:
            return 0.4
        else:
            # Agent was already in the destination room.
            return 0.0
    
    # 7. General State Change: Any action that successfully changed the environment state.
    if s != ns:
        return 0.2
        
    # 8. No-op: No change and no specific progress identified.
    return 0.0