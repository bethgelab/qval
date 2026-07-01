def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        hole_positions = []
        
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    agent_pos = (row_idx, col_idx)
                elif char == 'G':
                    goal_pos = (row_idx, col_idx)
                elif char == 'H':
                    hole_positions.append((row_idx, col_idx))
        
        return agent_pos, goal_pos, hole_positions
    
    curr_agent, curr_goal, curr_holes = parse_grid(state)
    next_agent, next_goal, next_holes = parse_grid(next_state)
    
    # If agent fell into a hole in next_state, very negative Q-value
    if next_agent and next_holes and next_agent in next_holes:
        return -1.0
    
    # If agent reached the goal in next_state, maximum Q-value
    if next_agent and next_goal and next_agent == next_goal:
        return 1.0
    
    # Calculate Manhattan distance from agent to goal in next_state
    if next_agent and next_goal:
        distance = abs(next_agent[0] - next_goal[0]) + abs(next_agent[1] - next_goal[1])
        
        # Maximum possible Manhattan distance on 8x8 grid is 14
        max_distance = 14
        
        # Calculate progress ratio (1.0 at goal, 0.0 at max distance)
        progress = 1.0 - (distance / max_distance)
        
        # Consider if distance is achievable within step limit (30 steps)
        # If distance exceeds remaining steps, lower the value
        remaining_steps = 30
        feasibility = min(1.0, remaining_steps / max(distance, 1))
        
        # Check if action made progress (agent moved closer to goal)
        if curr_agent and next_agent:
            curr_distance = abs(curr_agent[0] - next_goal[0]) + abs(curr_agent[1] - next_goal[1])
            made_progress = curr_distance > distance
            progress_bonus = 0.1 if made_progress else 0.0
        else:
            made_progress = False
            progress_bonus = 0.0
        
        # Base Q-value from progress, scaled by feasibility and progress bonus
        q_value = progress * 0.5 * feasibility + progress_bonus
        
        # If no progress was made and distance is large, penalize slightly
        if not made_progress and distance > max_distance / 2:
            q_value *= 0.8
        
        return q_value
    
    # Fallback if parsing failed
    return 0.0