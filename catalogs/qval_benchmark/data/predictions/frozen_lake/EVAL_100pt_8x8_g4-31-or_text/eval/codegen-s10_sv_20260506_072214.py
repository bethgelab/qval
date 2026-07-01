def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake environment.
    The value is based on the Manhattan distance to the goal, the density of holes 
    in the bounding box between the agent and the goal, and connectivity checks.
    """
    lines = state.split('\n')
    grid_data = []
    for line in lines:
        # Extract only non-whitespace characters to handle space-separated grids
        row = [char for char in line if char not in (' ', '\r', '\t')]
        if row:
            grid_data.append(row)
    
    if not grid_data:
        return 0.0
        
    ar, ac = -1, -1
    gr, gc = -1, -1
    holes = []
    
    # Locate agent, goal, and holes
    for r in range(len(grid_data)):
        for c in range(len(grid_data[r])):
            char = grid_data[r][c]
            if char == '@':
                ar, ac = r, c
            elif char == 'G':
                gr, gc = r, c
            elif char == 'H':
                holes.append((r, c))
    
    # Basic sanity checks
    if ar == -1 or gr == -1:
        return 0.0
    if ar == gr and ac == gc:
        return 1.0
        
    # Manhattan distance to the goal
    dist = abs(ar - gr) + abs(ac - gc)
    
    # Check if agent is immediately trapped (surrounded by holes or boundaries)
    agent_trapped = True
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < len(grid_data) and 0 <= nc < len(grid_data[0]):
            if grid_data[nr][nc] != 'H':
                agent_trapped = False
                break
    if agent_trapped:
        return 0.0
    
    # Check if goal is immediately trapped
    goal_trapped = True
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = gr + dr, gc + dc
        if 0 <= nr < len(grid_data) and 0 <= nc < len(grid_data[0]):
            if grid_data[nr][nc] != 'H':
                goal_trapped = False
                break
    if goal_trapped:
        return 0.0
    
    # Analyze the bounding box between agent and goal for hole density
    # Higher density suggests the optimal path is likely obstructed or longer.
    r1, r2 = min(ar, gr), max(ar, gr)
    c1, c2 = min(ac, gc), max(ac, gc)
    holes_in_box = 0
    for hr, hc in holes:
        if r1 <= hr <= r2 and c1 <= hc <= c2:
            holes_in_box += 1
    
    box_area = (r2 - r1 + 1) * (c2 - c1 + 1)
    hole_density = holes_in_box / box_area if box_area > 0 else 0
    
    # Heuristic formula:
    # 1. Use a discount-like factor for distance to reflect the preference for shorter paths.
    # 2. Scale down the value based on the concentration of holes in the direct path area.
    value = (0.92 ** dist) * (1.0 - hole_density)
    
    return max(0.0, min(1.0, value))