def signal_function(state: str):
    import math
    
    lines = state.strip().split('\n')
    grid = [line for line in lines if line]
    
    if not grid:
        return 0.01, {"progress_value": 0.01, "safety_penalty": 0.0, "reachability_bonus": 0.0, "blockage_penalty": 0.0, "efficiency_bonus": 0.0}
    
    # Find agent, goal, and holes
    agent_pos = None
    goal_pos = None
    holes = []
    
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == '@':
                agent_pos = (i, j)
            elif cell == 'G':
                goal_pos = (i, j)
            elif cell == 'H':
                holes.append((i, j))
    
    if agent_pos is None or goal_pos is None:
        return 0.01, {"progress_value": 0.01, "safety_penalty": 0.0, "reachability_bonus": 0.0, "blockage_penalty": 0.0, "efficiency_bonus": 0.0}
    
    # Calculate Manhattan distance to goal
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Progress value - exponential decay based on distance with better scaling
    progress_value = math.exp(-dist / 10.0)
    
    # Safety penalty - weighted by proximity to holes with better variance
    safety_penalty = 0.0
    for hole in holes:
        hole_dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if hole_dist == 1:
            safety_penalty += 0.22  # Immediate danger
        elif hole_dist == 2:
            safety_penalty += 0.10
        elif hole_dist == 3:
            safety_penalty += 0.04
        elif hole_dist <= 5:
            safety_penalty += 0.015
    
    # Cap safety penalty
    safety_penalty = min(safety_penalty, 0.50)
    
    # Blockage penalty - check if agent is surrounded by holes
    blockage_penalty = 0.0
    blocked_neighbors = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = agent_pos[0] + di, agent_pos[1] + dj
        if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
            if grid[ni][nj] == 'H':
                blocked_neighbors += 1
    
    if blocked_neighbors >= 3:
        blockage_penalty = 0.25
    elif blocked_neighbors == 2:
        blockage_penalty = 0.12
    elif blocked_neighbors == 1:
        blockage_penalty = 0.04
    
    # Reachability bonus - more nuanced distance-based scaling
    reachability_bonus = 0.0
    if dist <= 4:
        reachability_bonus = 0.10
    elif dist <= 8:
        reachability_bonus = 0.07
    elif dist <= 12:
        reachability_bonus = 0.04
    elif dist <= 16:
        reachability_bonus = 0.02
    
    # Efficiency bonus - rewards shorter paths (fewer steps = better with discounting)
    efficiency_bonus = 0.0
    if dist <= 3:
        efficiency_bonus = 0.15
    elif dist <= 6:
        efficiency_bonus = 0.10
    elif dist <= 10:
        efficiency_bonus = 0.06
    elif dist <= 15:
        efficiency_bonus = 0.03
    
    # Combine all components
    total = progress_value + reachability_bonus + efficiency_bonus - safety_penalty - blockage_penalty
    total = max(0.01, min(1.0, total))
    
    return total, {
        "progress_value": progress_value,
        "safety_penalty": safety_penalty,
        "reachability_bonus": reachability_bonus,
        "blockage_penalty": blockage_penalty,
        "efficiency_bonus": efficiency_bonus,
    }