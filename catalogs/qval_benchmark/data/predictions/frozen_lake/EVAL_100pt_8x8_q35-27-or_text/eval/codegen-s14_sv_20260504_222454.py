def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    
    # Find agent, goal, and holes
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    # If no agent or goal found, return 0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Count holes nearby (within 2 cells)
    nearby_holes = 0
    for hole in holes:
        dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if dist <= 2:
            nearby_holes += 1
    
    # Check if agent is adjacent to holes (dangerous position)
    dangerous = False
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if (nr, nc) in holes:
                dangerous = True
                break
    
    # Check if agent is in a corner or edge (potentially trapped)
    on_edge = (agent_pos[0] == 0 or agent_pos[0] == rows - 1 or 
               agent_pos[1] == 0 or agent_pos[1] == cols - 1)
    
    # Estimate base value based on distance to goal
    # Closer to goal = higher value (exponential decay)
    base_value = 1.0 / (1.0 + manhattan_dist * 0.15)
    
    # Penalize for nearby holes
    hole_penalty = nearby_holes * 0.08
    
    # Penalize if in dangerous position (adjacent to hole)
    dangerous_penalty = 0.20 if dangerous else 0.0
    
    # Penalize if on edge (less maneuverability)
    edge_penalty = 0.05 if on_edge else 0.0
    
    # Account for step limit (30 steps)
    # If distance is large relative to steps, success probability drops
    step_factor = max(0.0, (30.0 - manhattan_dist * 1.5) / 30.0)
    
    # Combine all factors
    value = base_value * (1.0 - hole_penalty - dangerous_penalty - edge_penalty) * step_factor
    
    # Ensure value is in [0, 1]
    return max(0.0, min(1.0, value))