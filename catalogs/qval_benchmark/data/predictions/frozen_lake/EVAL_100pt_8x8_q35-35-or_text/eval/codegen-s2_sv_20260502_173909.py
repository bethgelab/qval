import math

def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        if line:
            grid.append(list(line))
    
    if not grid:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Scan grid to find positions
    for r in range(rows):
        for c in range(len(grid[r])):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    # If agent or goal is missing, value is 0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Check if agent is at the goal
    if agent_pos == goal_pos:
        return 1.0
    
    # Check if agent is on a hole (should not happen in valid state, but handle it)
    if agent_pos in holes:
        return 0.0
    
    # Calculate Manhattan distance
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Base value decays with distance (exponential decay favors shorter paths)
    # Max distance is 14 in 8x8 grid. 0.92^14 is approx 0.33
    base_value = 0.92 ** dist
    
    # Hole penalty: More holes generally reduce success probability
    # We count total holes and holes in the bounding box between agent and goal
    total_holes = len(holes)
    box_holes = 0
    r_min = min(agent_pos[0], goal_pos[0])
    r_max = max(agent_pos[0], goal_pos[0])
    c_min = min(agent_pos[1], goal_pos[1])
    c_max = max(agent_pos[1], goal_pos[1])
    
    for (hr, hc) in holes:
        if r_min <= hr <= r_max and c_min <= hc <= c_max:
            box_holes += 1
            
    # Apply penalties
    # 1. Hole density penalty
    hole_density_penalty = 1.0 - (total_holes / (rows * cols)) * 0.5
    
    # 2. Direct path obstruction penalty (holes in bounding box)
    path_penalty = 0.7 if box_holes > 0 else 1.0
    
    # 3. Edge penalty (fewer valid moves if at edge)
    edge_penalty = 0.95 if (agent_pos[0] in (0, rows-1) or agent_pos[1] in (0, cols-1)) else 1.0
    
    estimated_value = base_value * hole_density_penalty * path_penalty * edge_penalty
    
    # Clamp to valid range [0.0, 1.0]
    return max(0.0, min(1.0, estimated_value))