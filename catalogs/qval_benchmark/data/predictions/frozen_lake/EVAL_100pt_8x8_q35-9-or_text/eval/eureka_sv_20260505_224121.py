def signal_function(state: str):
    lines = state.split('\n')
    grid = [list(line) for line in lines]
    
    agent_pos = None
    goal_pos = None
    holes = []
    total_cells = 0
    
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            cell = grid[r][c]
            total_cells += 1
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0, {
            "distance_to_goal": 0.0,
            "min_dist_to_hole": 0.0,
            "remaining_steps_estimate": 0.0,
            "progress_score": 0.0,
            "safety_score": 0.0,
            "feasibility_score": 0.0,
            "total": 0.0
        }
    
    agent_r, agent_c = agent_pos
    goal_r, goal_c = goal_pos
    
    dist_to_goal = abs(agent_r - goal_r) + abs(agent_c - goal_c)
    
    min_dist_to_hole = float('inf')
    if holes:
        for hole_r, hole_c in holes:
            dist = abs(agent_r - hole_r) + abs(agent_c - hole_c)
            min_dist_to_hole = min(min_dist_to_hole, dist)
    
    if min_dist_to_hole == float('inf'):
        min_dist_to_hole = 100
    
    safety_score = max(0.0, min(1.0, 1.0 - (min_dist_to_hole / 15.0)))
    
    detour_buffer = 0
    if min_dist_to_hole <= 1:
        detour_buffer = 4
    elif min_dist_to_hole <= 2:
        detour_buffer = 3
    elif min_dist_to_hole <= 3:
        detour_buffer = 2
    elif min_dist_to_hole <= 4:
        detour_buffer = 1
    else:
        detour_buffer = 0
    
    estimated_path_length = max(0, dist_to_goal + detour_buffer)
    remaining_steps_estimate = max(0, 30 - estimated_path_length)
    
    feasibility_score = max(0.0, min(1.0, remaining_steps_estimate / 30.0))
    
    progress_score = max(0.0, min(1.0, 1.0 - (dist_to_goal / 30.0)))
    
    total = progress_score * 0.5 + safety_score * 0.3 + feasibility_score * 0.2
    
    total = max(0.0, min(1.0, total))
    
    return total, {
        "distance_to_goal": float(dist_to_goal),
        "min_dist_to_hole": float(min_dist_to_hole) if min_dist_to_hole != float('inf') else 999.0,
        "remaining_steps_estimate": float(remaining_steps_estimate),
        "progress_score": float(progress_score),
        "safety_score": float(safety_score),
        "feasibility_score": float(feasibility_score),
        "total": float(total)
    }