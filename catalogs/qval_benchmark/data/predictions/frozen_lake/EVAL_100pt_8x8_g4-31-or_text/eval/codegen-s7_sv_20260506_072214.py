def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake grid environment.
    The value is an approximation of the probability of reaching the goal 'G' 
    from the agent's current position '@' without falling into holes 'H'.
    """
    # Parse the state grid
    grid = state.strip().split('\n')
    agent_pos = None
    goal_pos = None
    holes = []

    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))

    # If for some reason agent or goal is not found, return a neutral value
    if agent_pos is None or goal_pos is None:
        return 0.0

    # 1. Manhattan Distance
    ar, ac = agent_pos
    gr, gc = goal_pos
    dist = abs(ar - gr) + abs(ac - gc)

    # Immediate win condition
    if dist == 0:
        return 1.0

    # Base value decays as distance to goal increases
    # Max distance in 8x8 is 14. We use 16.0 to keep base_val > 0.
    base_val = 1.0 - (dist / 16.0)

    # 2. Hole Proximity Penalty
    # Optimal agents avoid holes, but in slippery environments, proximity is risky.
    adj_hole_penalty = 0.0
    for hr, hc in holes:
        h_dist = abs(ar - hr) + abs(ac - hc)
        if h_dist == 1:
            adj_hole_penalty += 0.15
        elif h_dist == 2:
            adj_hole_penalty += 0.05

    # 3. Path Obstruction Penalty
    # We estimate the "difficulty" of the path by checking the density of holes 
    # in the bounding box between the agent and the goal.
    r_min, r_max = min(ar, gr), max(ar, gr)
    c_min, c_max = min(ac, gc), max(ac, gc)
    box_area = (r_max - r_min + 1) * (c_max - c_min + 1)
    
    holes_in_box = 0
    for hr, hc in holes:
        if r_min <= hr <= r_max and c_min <= hc <= c_max:
            holes_in_box += 1
            
    # Density penalty: proportional to the fraction of the bounding box filled with holes
    box_penalty = (holes_in_box / box_area) * 0.4 if box_area > 0 else 0.0

    # Combine metrics
    estimated_value = base_val - adj_hole_penalty - box_penalty

    # Clamp the value between 0.0 and 1.0
    return max(0.0, min(1.0, estimated_value))