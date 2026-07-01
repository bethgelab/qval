def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the Frozen Lake grid environment.
    V(s) is approximated based on the Manhattan distance to the goal
    and the proximity of the agent to holes.
    """
    # Parse the ASCII grid to find agent, goal, and holes
    lines = state.splitlines()
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    # If agent or goal is missing, we cannot estimate value
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    ra, ca = agent_pos
    rg, cg = goal_pos
    
    # Manhattan distance to the goal
    dist = abs(ra - rg) + abs(ca - cg)
    
    # Reaching the goal immediately gives the maximum reward
    if dist == 0:
        return 1.0
        
    # Base value: Discounted reward based on distance.
    # We use a decay factor (gamma) of 0.9 to reflect that shorter paths are better.
    # 0.9^14 is approximately 0.22, 0.9^1 is 0.9.
    value = 0.9 ** dist
    
    # Adjust value based on proximity to holes (risk factor).
    # Holes closer to the agent increase the probability of failure.
    for rh, ch in holes:
        dist_to_hole = abs(ra - rh) + abs(ca - ch)
        if dist_to_hole == 1:
            # Immediate neighbor hole is high risk
            value *= 0.75
        elif dist_to_hole == 2:
            # A hole 2 steps away is a moderate risk
            value *= 0.95
            
    # Slight penalty for the overall density of holes in the map
    # The more holes there are, the harder the navigation in general.
    num_holes = len(holes)
    value *= (0.99 ** num_holes)
    
    # Ensure the value is bounded between 0.0 and 1.0
    return float(max(0.0, min(1.0, value)))