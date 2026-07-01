def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an agent in a frozen lake grid.
    The value is based on the distance to the goal and the proximity to holes.
    """
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        # Filter to only include characters relevant to the grid layout
        row = [char for char in line if char in 'SFGH@']
        if row:
            grid.append(row)
    
    if not grid:
        return 0.0
    
    rows = len(grid)
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Scan the grid to find agent, goal, and hole positions
    for r in range(rows):
        for c in range(len(grid[r])):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    # If the agent position is not explicitly '@', it might be 'S' (start)
    # though in the context of the value function, we are looking for the current '@' position.
    if agent_pos is None:
        for r in range(rows):
            for c in range(len(grid[r])):
                if grid[r][c] == 'S':
                    # If we only find S, we can't know the agent's current position 
                    # unless S is the current position.
                    agent_pos = (r, c)
                    break
            if agent_pos:
                break
                
    # If we can't find the agent or the goal, we can't calculate distance-based value
    if agent_pos is None or goal_pos is None:
        # In some edge cases, the agent might be at the goal, and 'G' is replaced by '@'
        # However, without knowing the goal position, we return a default.
        return 0.0

    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # Calculate Manhattan distance to the goal
    dist_to_goal = abs(ar - gr) + abs(ac - gc)
    
    # If the agent is at the goal, the value is 1.0
    if dist_to_goal == 0:
        return 1.0
        
    # The episode has a step limit of 30. If the goal is too far, the value is 0.
    if dist_to_goal > 30:
        return 0.0
    
    # Estimate value using a discount factor based on distance.
    # Using gamma=0.95 as a heuristic for the expected discounted reward.
    gamma = 0.95
    value = gamma ** dist_to_goal
    
    # Proximity to holes increases risk, especially in stochastic environments.
    # We reduce the value if the agent is very close to a hole.
    if holes:
        # Find the distance to the nearest hole
        min_hole_dist = min(abs(ar - hr) + abs(ac - hc) for hr, hc in holes)
        
        if min_hole_dist == 1:
            # Adjacent to a hole is very dangerous
            value *= 0.4
        elif min_hole_dist == 2:
            # Two steps away is moderately dangerous
            value *= 0.8
            
    # Ensure the value is within the valid range [0.0, 1.0]
    return float(max(0.0, min(1.0, value)))