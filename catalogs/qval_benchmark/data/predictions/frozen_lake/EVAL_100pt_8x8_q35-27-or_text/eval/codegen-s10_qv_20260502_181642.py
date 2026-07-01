def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse the grid from state
    lines = state.strip().split('\n')
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
    next_lines = next_state.strip().split('\n')
    next_grid = [list(line) for line in next_lines]
    
    # Find agent position in next_state
    next_agent_pos = None
    for i, row in enumerate(next_grid):
        for j, cell in enumerate(row):
            if cell == '@':
                next_agent_pos = (i, j)
                break
    
    # Check if we reached goal in next_state
    if next_agent_pos and goal_pos and next_agent_pos == goal_pos:
        return 1.0
    
    # Check if we fell in hole in next_state
    if next_agent_pos:
        for hole in holes:
            if next_agent_pos == hole:
                return -1.0
    
    # If agent didn't move, action might be invalid or blocked
    if agent_pos and next_agent_pos and agent_pos == next_agent_pos:
        action_dir = action.lower()
        if action_dir == 'up':
            delta_row, delta_col = -1, 0
        elif action_dir == 'down':
            delta_row, delta_col = 1, 0
        elif action_dir == 'left':
            delta_row, delta_col = 0, -1
        elif action_dir == 'right':
            delta_row, delta_col = 0, 1
        else:
            delta_row, delta_col = 0, 0
        
        new_row = agent_pos[0] + delta_row
        new_col = agent_pos[1] + delta_col
        
        # Check if trying to move into hole
        for hole in holes:
            if (new_row, new_col) == hole:
                return -0.5
        
        # Check if moving into boundary
        if not (0 <= new_row < len(grid) and 0 <= new_col < len(grid[0])):
            return 0.0
    
    # Estimate Q-value based on distance to goal
    if agent_pos and goal_pos:
        # Calculate Manhattan distance
        dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        
        # Maximum possible distance in 8x8 grid is 14
        max_dist = 14
        
        # Base value inversely proportional to distance
        base_value = 1.0 - (dist / max_dist)
        
        # Adjust based on action direction relative to goal
        if next_agent_pos and agent_pos:
            action_dir = action.lower()
            if action_dir == 'up':
                delta_row, delta_col = -1, 0
            elif action_dir == 'down':
                delta_row, delta_col = 1, 0
            elif action_dir == 'left':
                delta_row, delta_col = 0, -1
            elif action_dir == 'right':
                delta_row, delta_col = 0, 1
            else:
                delta_row, delta_col = 0, 0
            
            # New position after action
            new_row = agent_pos[0] + delta_row
            new_col = agent_pos[1] + delta_col
            
            # Check if new position is within bounds
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                new_dist = abs(new_row - goal_pos[0]) + abs(new_col - goal_pos[1])
                
                # If action reduces distance, bonus
                if new_dist < dist:
                    base_value += 0.2
                # If action increases distance, penalty
                elif new_dist > dist:
                    base_value -= 0.2
        
        # Penalize proximity to holes
        hole_penalty = 0.0
        for hole in holes:
            hole_dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
            if hole_dist <= 2:
                hole_penalty += 0.1
        
        base_value -= hole_penalty
        
        # Ensure value is in reasonable range
        return max(-1.0, min(1.0, base_value))
    
    return 0.0