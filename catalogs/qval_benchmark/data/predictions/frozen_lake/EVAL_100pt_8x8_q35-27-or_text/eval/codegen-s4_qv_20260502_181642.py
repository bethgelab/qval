def signal_function(state: str, action: str, next_state: str) -> float:
    import math
    
    # Parse the current state grid
    lines = state.strip().split('\n')
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
    next_lines = next_state.strip().split('\n')
    next_grid = [list(line) for line in next_lines if line.strip()]
    
    # Find agent and goal in next state
    next_agent_pos = None
    next_goal_pos = None
    
    for r, row in enumerate(next_grid):
        for c, cell in enumerate(row):
            if cell == '@':
                next_agent_pos = (r, c)
            elif cell == 'G':
                next_goal_pos = (r, c)
    
    # Use goal position from current state if not found in next
    if next_goal_pos is None and goal_pos is not None:
        next_goal_pos = goal_pos
    
    # Check if agent fell into a hole (agent on H in next state)
    if next_agent_pos is not None:
        nr, nc = next_agent_pos
        if 0 <= nr < len(next_grid) and 0 <= nc < len(next_grid[0]):
            if next_grid[nr][nc] == 'H':
                return -10.0
    
    # Check if reached goal
    if next_agent_pos is not None and next_goal_pos is not None:
        if next_agent_pos == next_goal_pos:
            return 1.0
    
    # Calculate Manhattan distances
    if agent_pos is not None and goal_pos is not None:
        dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    else:
        dist_to_goal = 20
    
    if next_agent_pos is not None and next_goal_pos is not None:
        next_dist_to_goal = abs(next_agent_pos[0] - next_goal_pos[0]) + abs(next_agent_pos[1] - next_goal_pos[1])
    else:
        next_dist_to_goal = 20
    
    # Calculate progress toward goal
    progress = dist_to_goal - next_dist_to_goal
    
    # Calculate hole risk in next state
    hole_risk = 0.0
    if next_agent_pos is not None and holes:
        nr, nc = next_agent_pos
        for hr, hc in holes:
            dist_to_hole = abs(nr - hr) + abs(nc - hc)
            if dist_to_hole <= 3:
                hole_risk += 1.0 / (dist_to_hole + 0.5)
    
    # Base Q-value estimate
    base_q = 0.0
    
    # Reward for making progress toward goal
    if progress > 0:
        base_q += 0.4 * min(progress, 3.0) / (dist_to_goal + 1)
    elif progress < 0:
        base_q -= 0.2 * min(abs(progress), 2.0) / (dist_to_goal + 1)
    
    # Penalty for hole proximity
    base_q -= 0.3 * hole_risk
    
    # Bonus for being close to goal
    if dist_to_goal <= 2:
        base_q += 0.5
    elif dist_to_goal <= 4:
        base_q += 0.25
    
    # Penalty for being far from goal (episode limit consideration)
    if dist_to_goal > 10:
        base_q -= 0.1 * (dist_to_goal - 10) / 10.0
    
    # Action-specific adjustments based on direction
    if next_agent_pos is not None and agent_pos is not None and goal_pos is not None:
        dr = next_agent_pos[0] - agent_pos[0]
        dc = next_agent_pos[1] - agent_pos[1]
        goal_dr = goal_pos[0] - agent_pos[0]
        goal_dc = goal_pos[1] - agent_pos[1]
        
        # Check if action moves toward goal
        toward_goal = (dr * goal_dr >= 0) and (dc * goal_dc >= 0)
        if toward_goal and (dr != 0 or dc != 0):
            base_q += 0.15
    
    # Clamp to reasonable range
    q_value = max(-5.0, min(5.0, base_q))
    
    return q_value