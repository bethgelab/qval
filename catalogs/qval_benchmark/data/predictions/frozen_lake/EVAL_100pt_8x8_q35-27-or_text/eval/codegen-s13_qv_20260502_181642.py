def signal_function(state: str, action: str, next_state: str) -> float:
    import math
    
    # Parse current state grid
    lines = state.split('\n')
    grid = [list(line) for line in lines if line.strip()]
    
    # Find agent, goal, and holes in current state
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
    
    # Parse next_state to find new agent position
    next_lines = next_state.split('\n')
    next_grid = [list(line) for line in next_lines if line.strip()]
    
    next_agent_pos = None
    for i, row in enumerate(next_grid):
        for j, cell in enumerate(row):
            if cell == '@':
                next_agent_pos = (i, j)
                break
        if next_agent_pos:
            break
    
    # Check if we reached the goal
    if next_agent_pos == goal_pos:
        return 1.0
    
    # Check if we fell into a hole
    if next_agent_pos in holes:
        return -0.5
    
    # Calculate Manhattan distance to goal
    if agent_pos and goal_pos:
        dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    else:
        dist_to_goal = 14
    
    if next_agent_pos and goal_pos:
        next_dist = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
    else:
        next_dist = 14
    
    # Base value: closer to goal is better
    base_value = 0.5
    
    # Reward for moving closer to goal
    if next_dist < dist_to_goal:
        progress_bonus = (dist_to_goal - next_dist) / max(dist_to_goal, 1)
        base_value += 0.3 * progress_bonus
    # Penalty for moving away from goal
    elif next_dist > dist_to_goal:
        base_value -= 0.2 * (next_dist - dist_to_goal) / max(dist_to_goal, 1)
    
    # Penalize proximity to holes in next position
    if holes and next_agent_pos:
        min_hole_dist = float('inf')
        for hole in holes:
            dist = abs(next_agent_pos[0] - hole[0]) + abs(next_agent_pos[1] - hole[1])
            min_hole_dist = min(min_hole_dist, dist)
        
        if min_hole_dist <= 1:
            base_value -= 0.4  # Very close to hole
        elif min_hole_dist <= 2:
            base_value -= 0.2  # Near hole
    
    # Bonus for being on safe ground (not near holes)
    if next_agent_pos and holes:
        safe_dist = True
        for hole in holes:
            dist = abs(next_agent_pos[0] - hole[0]) + abs(next_agent_pos[1] - hole[1])
            if dist <= 2:
                safe_dist = False
                break
        if safe_dist:
            base_value += 0.1
    
    # Penalty for staying in place (no progress)
    if next_agent_pos == agent_pos:
        base_value -= 0.1
    
    # Clip to reasonable range
    return max(-0.5, min(1.0, base_value))