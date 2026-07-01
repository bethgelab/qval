def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse current state grid
    lines = state.split('\n')
    grid = [list(line) for line in lines if line.strip()]
    
    # Find agent, goal, and holes in current state
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
    
    # Parse next_state grid
    next_lines = next_state.split('\n')
    next_grid = [list(line) for line in next_lines if line.strip()]
    
    # Find agent, goal, and holes in next state
    next_agent_pos = None
    next_goal_pos = None
    next_holes = []
    
    for r, row in enumerate(next_grid):
        for c, cell in enumerate(row):
            if cell == '@':
                next_agent_pos = (r, c)
            elif cell == 'G':
                next_goal_pos = (r, c)
            elif cell == 'H':
                next_holes.append((r, c))
    
    # Check if goal reached in next state
    if next_agent_pos == next_goal_pos:
        return 1.0
    
    # Check if agent fell into a hole (on top of H)
    if next_agent_pos:
        for hole in next_holes:
            if next_agent_pos == hole:
                return 0.0
    
    # Calculate Manhattan distance to goal
    if agent_pos and goal_pos:
        dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    else:
        dist_to_goal = 100
    
    if next_agent_pos and next_goal_pos:
        next_dist_to_goal = abs(next_agent_pos[0] - next_goal_pos[0]) + abs(next_agent_pos[1] - next_goal_pos[1])
    else:
        next_dist_to_goal = 100
    
    # Progress toward goal (positive = moved closer)
    progress = dist_to_goal - next_dist_to_goal
    
    # Base Q-value on progress
    q_value = progress * 0.15
    
    # Bonus for being close to goal
    if next_dist_to_goal <= 3:
        q_value += 0.3
    elif next_dist_to_goal <= 6:
        q_value += 0.15
    
    # Penalize proximity to holes in next state
    if next_agent_pos:
        min_dist_to_hole = float('inf')
        for hole in next_holes:
            dist = abs(next_agent_pos[0] - hole[0]) + abs(next_agent_pos[1] - hole[1])
            min_dist_to_hole = min(min_dist_to_hole, dist)
        
        if min_dist_to_hole <= 2:
            q_value -= 0.4
        elif min_dist_to_hole <= 4:
            q_value -= 0.2
    
    # Penalize moving away from goal
    if progress < 0:
        q_value -= abs(progress) * 0.1
    
    # Account for remaining steps (episode limit is 30)
    # Assume we're roughly halfway through on average
    remaining_steps = 25
    if next_dist_to_goal > remaining_steps:
        q_value -= 0.2
    
    # Clamp to reasonable range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value