def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [list(line) for line in lines]
        agent_pos = None
        goal_pos = None
        holes = set()
        
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == '@':
                    agent_pos = (i, j)
                elif cell == 'G':
                    goal_pos = (i, j)
                elif cell == 'H':
                    holes.add((i, j))
        
        return grid, agent_pos, goal_pos, holes
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_in_bounds(pos, rows, cols):
        i, j = pos
        return 0 <= i < rows and 0 <= j < cols
    
    grid, agent_pos, goal_pos, holes = parse_grid(state)
    next_grid, next_agent_pos, _, _ = parse_grid(next_state)
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    dist_to_goal = manhattan_distance(agent_pos, goal_pos)
    max_dist = rows + cols - 2  # Maximum Manhattan distance in grid
    
    # Base value from distance to goal (closer = higher value)
    distance_value = (max_dist - dist_to_goal) / max_dist if max_dist > 0 else 0.0
    
    # Determine if action moves toward goal
    action_direction = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    
    toward_goal_bonus = 0.0
    if action in action_direction:
        dx, dy = action_direction[action]
        goal_dx = goal_pos[0] - agent_pos[0]
        goal_dy = goal_pos[1] - agent_pos[1]
        
        # Check if action reduces distance in each dimension
        if goal_dx != 0 and (goal_dx > 0 and dx > 0) or (goal_dx < 0 and dx < 0):
            toward_goal_bonus += 0.1
        if goal_dy != 0 and (goal_dy > 0 and dy > 0) or (goal_dy < 0 and dy < 0):
            toward_goal_bonus += 0.1
    
    # Safety check - penalty if action leads to hole
    safety_penalty = 0.0
    if action in action_direction:
        dx, dy = action_direction[action]
        new_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
        if new_pos in holes:
            safety_penalty = -0.4
        elif not is_in_bounds(new_pos, rows, cols):
            safety_penalty = -0.1  # Hitting wall penalty
    
    # Next state value - if next state is closer to goal, bonus
    next_dist_bonus = 0.0
    if next_agent_pos is not None:
        next_dist = manhattan_distance(next_agent_pos, goal_pos)
        if next_dist < dist_to_goal:
            next_dist_bonus = 0.15
        elif next_dist == dist_to_goal:
            next_dist_bonus = 0.05
        elif next_dist > dist_to_goal:
            next_dist_bonus = -0.1
    
    # Goal reached bonus in next state
    goal_bonus = 0.0
    if next_agent_pos == goal_pos:
        goal_bonus = 0.5
    
    # Combine all factors
    q_value = distance_value + toward_goal_bonus + safety_penalty + next_dist_bonus + goal_bonus
    
    # Clamp to reasonable range
    return max(0.0, min(1.0, q_value))