def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse the grid from state
    lines = [line for line in state.split('\n') if line.strip()]
    grid = [list(line) for line in lines]
    
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
    
    # Parse next_state
    next_lines = [line for line in next_state.split('\n') if line.strip()]
    next_grid = [list(line) for line in next_lines]
    
    # Find agent and goal in next_state
    next_agent_pos = None
    next_goal_pos = None
    
    for i, row in enumerate(next_grid):
        for j, cell in enumerate(row):
            if cell == '@':
                next_agent_pos = (i, j)
            elif cell == 'G':
                next_goal_pos = (i, j)
    
    # If we reached the goal in next_state, return high value
    if next_agent_pos and next_goal_pos and next_agent_pos == next_goal_pos:
        return 1.0
    
    # Check if agent fell into a hole in next_state
    if next_agent_pos:
        row, col = next_agent_pos
        if row < len(next_grid) and col < len(next_grid[0]):
            if next_grid[row][col] == 'H':
                return 0.0
    
    # If we can't parse positions, return baseline
    if agent_pos is None or goal_pos is None or next_agent_pos is None:
        return 0.3
    
    # Calculate Manhattan distances
    current_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    next_dist = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
    
    # Progress toward goal (positive if closer)
    progress = current_dist - next_dist
    
    # Base Q-value estimate
    q_value = 0.4  # baseline for safe non-terminal states
    
    # Reward for making progress toward goal
    if progress > 0:
        q_value += progress * 0.15
    elif progress < 0:
        q_value -= abs(progress) * 0.1
    
    # Penalty for being close to holes in next_state
    for hole in holes:
        hole_dist = abs(next_agent_pos[0] - hole[0]) + abs(next_agent_pos[1] - hole[1])
        if hole_dist <= 1:
            q_value -= 0.25
        elif hole_dist <= 2:
            q_value -= 0.1
    
    # Bonus for being very close to goal
    if next_dist <= 2:
        q_value += 0.2
    
    # Slight preference for actions that maintain or improve position
    if next_dist <= current_dist:
        q_value += 0.05
    
    # Clamp to reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value