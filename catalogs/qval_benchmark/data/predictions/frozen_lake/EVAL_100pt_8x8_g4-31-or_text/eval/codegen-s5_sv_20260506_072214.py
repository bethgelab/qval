def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an 8x8 Frozen Lake environment.
    The value is based on the Manhattan distance to the goal and the 
    density of holes in the bounding box between the agent and the goal.
    """
    # Parse the grid to find coordinates of the agent, goal, and holes.
    lines = state.strip().split('\n')
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, line in enumerate(lines):
        # Remove spaces to ensure coordinate indices are based on actual grid cells.
        clean_line = line.replace(' ', '')
        for c, char in enumerate(clean_line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    # If essential components are missing, the state value is 0.
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # Calculate Manhattan distance.
    dist = abs(ar - gr) + abs(ac - gc)
    
    # The expected value V(s) is higher when the agent is closer to the goal.
    # We use a decay factor to estimate the probability of reaching the goal.
    # 0.95^14 is approx 0.48, keeping the value reasonably high for far goals.
    value = 0.95 ** dist
    
    # Identify holes within the bounding box formed by the agent and the goal.
    # These holes are more likely to obstruct the optimal path.
    r_min, r_max = min(ar, gr), max(ar, gr)
    c_min, c_max = min(ac, gc), max(ac, gc)
    
    hole_penalty = 0.0
    for hr, hc in holes:
        if r_min <= hr <= r_max and c_min <= hc <= c_max:
            # Holes closer to the agent represent a higher immediate risk.
            h_dist = abs(hr - ar) + abs(hc - ac)
            hole_penalty += 0.05 / (h_dist + 1)
            
    # Subtract the hole penalty from the distance-based value, ensuring it stays non-negative.
    estimated_value = max(0.0, value - hole_penalty)
    
    return float(estimated_value)