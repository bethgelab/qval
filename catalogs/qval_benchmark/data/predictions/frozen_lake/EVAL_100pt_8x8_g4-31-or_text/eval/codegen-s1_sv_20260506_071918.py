def signal_function(state: str) -> float:
    lines = [line.strip() for line in state.split('\n') if line.strip()]
    
    # Find the 8x8 grid by looking for a contiguous block of 8 lines with at least 8 characters
    grid = []
    for i in range(len(lines) - 7):
        if all(len(lines[i+j]) >= 8 for j in range(8)):
            grid = [lines[i+j][:8] for j in range(8)]
            break
    
    if not grid:
        # Fallback: take the first 8 lines if no contiguous block is found
        grid = lines[:8]
        if len(grid) < 8:
            return 0.0

    agent_pos = None
    goal_pos = None
    holes = []

    # First pass: identify holes and the goal location
    for r in range(8):
        for c in range(8):
            if c < len(grid[r]):
                char = grid[r][c]
                if char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.append((r, c))

    # Second pass: identify the agent's current position
    # The agent is marked as '@', but could potentially be 'S' in some initial states
    for r in range(8):
        for c in range(8):
            if c < len(grid[r]):
                char = grid[r][c]
                if char == '@' or char == 'S':
                    agent_pos = (r, c)
                    break
        if agent_pos:
            break

    if agent_pos is None:
        return 0.0
    
    # Special Case: If 'G' is not in the grid but 'G' exists in the state text,
    # it is likely the agent is standing on the goal.
    if goal_pos is None:
        if 'G' in state:
            return 1.0
        else:
            return 0.0

    # Calculate Manhattan distance to the goal
    dist_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If the agent is at the goal, the value is 1.0
    if dist_goal == 0:
        return 1.0

    # Heuristic value based on distance to goal (discounted reward)
    # Using a discount factor gamma = 0.95 to reflect efficiency and proximity
    gamma = 0.95
    v = gamma ** dist_goal
    
    # Penalty for being near holes to reflect risk
    for hr, hc in holes:
        dist_hole = abs(agent_pos[0] - hr) + abs(agent_pos[1] - hc)
        if dist_hole == 1:
            # Immediate adjacency to a hole is high risk
            v *= 0.5
        elif dist_hole == 2:
            # Proximity to a hole is moderate risk
            v *= 0.8
            
    # Return the clamped value between 0.0 and 1.0
    return float(max(0.0, min(1.0, v)))