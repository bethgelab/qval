def signal_function(state: str):
    lines = state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    holes = []
    grid_rows = len(lines)
    grid_cols = len(lines[0]) if lines else 0
    
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0, {
            "progress": 0.0,
            "safety": 0.0,
            "path_blockage": 0.0,
            "step_limit": 0.0
        }
    
    dr = abs(agent_pos[0] - goal_pos[0])
    dc = abs(agent_pos[1] - goal_pos[1])
    manhattan_dist = dr + dc
    
    if manhattan_dist == 0:
        return 1.0, {
            "progress": 1.0,
            "safety": 1.0,
            "path_blockage": 1.0,
            "step_limit": 1.0
        }
    
    # Progress: Distance-based value with power law for sharper discrimination
    # Closer states get exponentially higher values
    progress = 1.0 - (manhattan_dist / 14.0) ** 1.5
    progress = max(0.0, progress)
    
    # Safety: Consider both proximity to holes AND holes on potential paths
    min_hole_dist = float('inf')
    holes_on_path = 0
    for hole in holes:
        dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        min_hole_dist = min(min_hole_dist, dist)
        
        # Check if hole is in the bounding box between agent and goal
        if (min(agent_pos[0], goal_pos[0]) <= hole[0] <= max(agent_pos[0], goal_pos[0]) and
            min(agent_pos[1], goal_pos[1]) <= hole[1] <= max(agent_pos[1], goal_pos[1])):
            holes_on_path += 1
    
    # Safety combines proximity risk with path obstruction risk
    if min_hole_dist == float('inf'):
        proximity_safety = 1.0
    else:
        # Sharper decay for close holes
        proximity_safety = 1.0 - 0.85 * (0.35 ** min_hole_dist)
        proximity_safety = max(0.0, proximity_safety)
    
    # Path obstruction penalty based on holes blocking direct routes
    if holes_on_path == 0:
        obstruction_safety = 1.0
    else:
        # More holes on path = higher risk
        obstruction_safety = 1.0 - 0.3 * min(1.0, holes_on_path * 0.25)
        obstruction_safety = max(0.0, obstruction_safety)
    
    safety = proximity_safety * obstruction_safety
    
    # Path blockage: More sophisticated detection considering multiple factors
    path_blockage = 1.0
    if holes:
        min_r = min(agent_pos[0], goal_pos[0])
        max_r = max(agent_pos[0], goal_pos[0])
        min_c = min(agent_pos[1], goal_pos[1])
        max_c = max(agent_pos[1], goal_pos[1])
        
        # Count holes that could block the path
        blocking_holes = 0
        for h in holes:
            if min_r <= h[0] <= max_r and min_c <= h[1] <= max_c:
                blocking_holes += 1
        
        # Consider hole density in the bounding box
        bounding_area = max(1, (max_r - min_r + 1) * (max_c - min_c + 1))
        hole_density = blocking_holes / bounding_area
        
        # Penalize based on both count and density
        blockage_factor = min(1.0, blocking_holes * 0.18 + hole_density * 0.4)
        path_blockage = 1.0 - blockage_factor
    
    # Step limit: Account for detours needed due to holes
    # More holes on path means more detour steps needed
    steps_needed = manhattan_dist + holes_on_path * 2  # Estimate detour cost
    step_limit = max(0.0, 1.0 - (steps_needed / 35.0))
    step_limit = min(1.0, step_limit)
    
    # Combine with recalibrated weights - emphasize path complexity
    # Progress and path_blockage are most discriminative
    total = progress * 0.50 + safety * 0.20 + path_blockage * 0.20 + step_limit * 0.10
    
    return total, {
        "progress": progress,
        "safety": safety,
        "path_blockage": path_blockage,
        "step_limit": step_limit,
    }