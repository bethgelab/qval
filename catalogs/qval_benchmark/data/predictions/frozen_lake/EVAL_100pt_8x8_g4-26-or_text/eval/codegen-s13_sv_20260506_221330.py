def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given Frozen Lake state.
    The estimate is based on the Manhattan distance to the goal and the 
    proximity/obstruction level of holes (H).
    """
    # Parse the grid into a 2D list of characters
    grid = []
    for line in state.strip().split('\n'):
        row = [c for c in line if c != ' ']
        if row:
            grid.append(row)
            
    if not grid:
        return 0.0
        
    rows = len(grid)
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Locate the agent, the goal, and all holes in the grid
    for r in range(rows):
        for c in range(len(grid[r])):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    # If essential landmarks are missing, we cannot estimate a value
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # Calculate Manhattan distance to the goal
    dist_ag = abs(ar - gr) + abs(ac - gc)
    
    # If agent is at the goal
    if dist_ag == 0:
        return 1.0
        
    # If the goal is too far away given the step limit
    if dist_ag > 30:
        return 0.0
        
    # Base value decays with distance to reflect efficiency and discounting
    # 0.95^d is a common approximation for value decay in grid worlds
    v_base = 0.95 ** dist_ag
    
    # Safety factor: reduces the value based on the presence of holes
    safety = 1.0
    for hr, hc in holes:
        d_ah = abs(ar - hr) + abs(ac - hc)
        d_hg = abs(hr - gr) + abs(hc - gc)
        
        penalty = 0.0
        # High penalty if the hole is on a potential shortest path
        if d_ah + d_hg == dist_ag:
            penalty = 0.5 / (d_ah + 1.0)
        # Moderate penalty if the hole is adjacent to the agent
        elif d_ah <= 1:
            penalty = 0.4 / (d_ah + 1.0)
        # Moderate penalty if the hole is adjacent to the goal
        elif d_hg <= 1:
            penalty = 0.3 / (d_hg + 1.0)
            
        # Apply the penalty to the safety multiplier
        safety *= (1.0 - penalty)
        
    # The final value is the decayed base value multiplied by the safety factor
    return float(v_base * safety)