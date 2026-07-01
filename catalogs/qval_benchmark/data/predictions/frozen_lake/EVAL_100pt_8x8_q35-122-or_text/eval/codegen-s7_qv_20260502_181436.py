def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        """Parse the grid string to find agent, goal, and hole positions"""
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
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # Parse the next_state to evaluate the outcome of the action
    agent_pos, goal_pos, hole_positions = parse_grid(next_state)
    
    # If we cannot parse the state, return neutral value
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Agent fell into a hole - very negative outcome
    if agent_pos in hole_positions:
        return -1.0
    
    # Agent reached the goal - maximum positive reward
    if agent_pos == goal_pos:
        return 1.0
    
    # Calculate Manhattan distance to goal
    distance = manhattan_distance(agent_pos, goal_pos)
    
    # Maximum possible Manhattan distance on 8x8 grid is 14 (7+7)
    max_distance = 14
    
    # Base score from proximity to goal (closer = better)
    distance_score = 1.0 - (distance / max_distance)
    
    # Apply discount factor based on expected steps to reach goal
    # With 30 step limit and sparse reward, closer states have higher value
    discount = 0.95
    q_value = distance_score * (discount ** distance)
    
    return q_value