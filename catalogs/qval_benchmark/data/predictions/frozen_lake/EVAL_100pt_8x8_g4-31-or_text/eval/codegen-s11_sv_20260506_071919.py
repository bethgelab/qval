def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a frozen lake grid state.
    
    The value is based on:
    1. Proximity to the goal (G).
    2. Presence of obstacles (H).
    3. Step efficiency (implicit in the density of rewards).
    """
    lines = state.strip().split('\n')
    grid = [line.strip() for line in lines if line.strip()]
    
    if not grid:
        return 0.0

    rows = len(grid)
    cols = len(grid[0])
    
    agent_pos = None
    goal_pos = None
    holes = []

    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))

    # If agent is at goal, value is 1.0
    if agent_pos == goal_pos:
        return 1.0
    
    # If agent is in a hole (shouldn't happen in valid state, but for safety), value is 0.0
    # Check if agent position is actually a hole index
    for h in holes:
        if agent_pos == h:
            return 0.0

    if not goal_pos:
        return 0.0

    # Manhattan distance to goal
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Maximum possible distance in an 8x8 grid is 14
    max_dist = (rows - 1) + (cols - 1)
    
    # Base value: closer to goal is better. 
    # We use a decay model to simulate the expected reward.
    # Since the step limit is 30, if dist > 30, value is 0.
    if dist_to_goal > 30:
        return 0.0
    
    # We want a value that decreases as distance increases.
    # A simple linear or exponential decay. 
    # Let's use a decay that accounts for the sparse reward.
    # V(s) approx (gamma^dist) where gamma is some discount factor.
    # Given the sparse reward of 1.0, we can estimate:
    value = 1.0 / (dist_to_goal + 1)
    
    # Penalty for holes:
    # If there are many holes or holes are blocking the path, the value should decrease.
    # A simple heuristic: find if a hole is adjacent to the agent.
    adj_holes = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 'H':
                adj_holes += 1
    
    # Reduce value if surrounded by holes
    value *= (0.5 ** adj_holes)

    # Adjust value based on general "danger" (hole density)
    # This is a very loose heuristic
    hole_density = len(holes) / (rows * cols)
    value *= (1.0 - (hole_density * 0.5))

    return float(max(0.0, min(1.0, value)))