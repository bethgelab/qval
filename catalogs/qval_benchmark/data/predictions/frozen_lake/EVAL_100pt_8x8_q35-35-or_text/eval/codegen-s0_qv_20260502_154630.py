def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse the next state grid
    next_grid = next_state.strip().split('\n')
    
    # Find agent position, goal position, and hole positions in next_state
    agent_pos = None
    goal_pos = None
    has_hole = False
    
    for r, row in enumerate(next_grid):
        for c, cell in enumerate(row):
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                has_hole = True
    
    # If agent reached goal, return maximum reward
    if agent_pos and goal_pos and agent_pos == goal_pos:
        return 1.0
    
    # If agent fell into hole, return minimum reward
    if agent_pos and has_hole:
        for r, row in enumerate(next_grid):
            for c, cell in enumerate(row):
                if cell == 'H' and (r, c) == agent_pos:
                    return 0.0
    
    # If no agent position (terminal state), check if goal was reached
    if agent_pos is None:
        # Check if goal exists and agent might have reached it
        goal_found = False
        for r, row in enumerate(next_grid):
            if 'G' in row:
                goal_found = True
                break
        if goal_found:
            return 1.0
        return 0.0
    
    # Calculate Manhattan distance to goal
    if agent_pos and goal_pos:
        distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        
        # Higher Q-value for closer distance to goal
        # Use exponential decay for smoother estimation
        import math
        q_value = math.exp(-distance / 4.0)
        return q_value
    
    # Default fallback
    return 0.0