def signal_function(state: str):
    """
    Estimates the state-value V(s) for the Frozen Lake 8x8 environment.
    The value is based on the proximity to the goal, adjusted by a dynamic 
    penalty for holes that obstruct the optimal path or pose immediate risk.
    """
    # Parse the ASCII grid to locate agent, goal, and holes
    grid = state.splitlines()
    if not grid:
        return 0.0
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, line in enumerate(grid):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # Manhattan distance from agent to goal
    dist_sg = abs(ar - gr) + abs(ac - gc)
    
    # Terminal state: agent reached the goal
    if dist_sg == 0:
        return 1.0, {"dist_score": 1.0, "hole_penalty": 0.0}
    
    # Base value is derived from a discount factor gamma^distance.
    # Using gamma=0.96 to provide a strong but steady gradient towards the goal.
    dist_score = 0.96 ** dist_sg
    
    # Dynamic hole penalty based on proximity and obstruction.
    total_hole_penalty = 0.0
    for hr, hc in holes:
        # Distance from agent to hole
        d_sh = abs(ar - hr) + abs(ac - hc)
        # Distance from hole to goal
        d_gh = abs(hr - gr) + abs(hc - gc)
        
        if d_sh == 0:
            continue # Agent should not be on a hole, but for safety.

        # Determine the impact weight of the hole.
        # 1. Hole is exactly on a shortest path to the goal.
        # 2. Hole is within the bounding box created by agent and goal.
        # 3. Hole is elsewhere on the map.
        if d_sh + d_gh == dist_sg:
            impact = 0.20  # High impact: blocks optimal path
        elif (min(ar, gr) <= hr <= max(ar, gr) and 
              min(ac, gc) <= hc <= max(ac, gc)):
            impact = 0.05  # Medium impact: generally in the way
        else:
            impact = 0.01  # Low impact: unlikely to obstruct
            
        # Use a proximity power (1.5) to make the penalty more dynamic.
        # Closer holes result in a significantly higher penalty, 
        # creating a "danger zone" around the hole.
        total_hole_penalty += impact / (d_sh ** 1.5)
    
    # Cap the penalty to ensure the signal remains positive and 
    # doesn't completely zero out the value unless the state is truly dire.
    effective_penalty = min(total_hole_penalty, dist_score * 0.8)
    
    total = max(0.0, dist_score - effective_penalty)
    
    return total, {
        "dist_score": dist_score,
        "hole_penalty": -effective_penalty,
    }