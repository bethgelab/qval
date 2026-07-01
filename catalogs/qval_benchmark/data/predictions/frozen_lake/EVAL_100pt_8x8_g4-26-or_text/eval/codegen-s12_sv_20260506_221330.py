def signal_function(state: str) -> float:
    import math

    # Parse the grid from the ASCII representation
    grid = []
    for line in state.strip().split('\n'):
        # Filter for relevant grid characters
        row = [c for c in line if c in ('S', 'F', 'H', 'G', '@')]
        if row:
            grid.append(row)
    
    if not grid:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0])
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Scan the grid to locate the agent, the goal, and all holes
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    # If the agent cannot be found, return 0.0
    if agent_pos is None:
        return 0.0
        
    # If the goal position is not visible, it is highly likely the agent is 
    # currently occupying the goal cell (making the reward 1.0).
    if goal_pos is None:
        return 1.0
        
    # Calculate Manhattan distance to the goal
    dist_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If the agent is on the goal, the value is 1.0
    if dist_goal == 0:
        return 1.0
        
    # Calculate the distance to the nearest hole to assess danger
    min_dist_hole = float('inf')
    for hr, hc in holes:
        d = abs(agent_pos[0] - hr) + abs(agent_pos[1] - hc)
        if d < min_dist_hole:
            min_dist_hole = d
            
    # Safety factor: reduce the value estimate if the agent is near a hole.
    # Proximity to holes increases the probability of falling in.
    if min_dist_hole == 1:
        safety = 0.3
    elif min_dist_hole == 2:
        safety = 0.7
    elif min_dist_hole == 3:
        safety = 0.9
    else:
        safety = 1.0
        
    # Estimate V(s) using an exponential decay based on the distance to the goal.
    # This incorporates the preference for shorter paths.
    # V(s) approx P(success) * decay(dist)
    v_est = math.exp(-0.15 * dist_goal) * safety
    
    # Ensure the result is within the theoretical bounds [0, 1]
    return max(0.0, min(1.0, v_est))