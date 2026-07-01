import math

def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().split('\n')
    grid = [line.strip() for line in lines if line.strip()]
    
    if not grid:
        return 0.0
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    # Scan the grid for positions
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                hole_positions.append((r, c))
    
    # If agent is not found, assume invalid or terminal state
    if agent_pos is None:
        return 0.0
    
    # If goal is not found, assume agent is at goal (terminal state reached)
    if goal_pos is None:
        return 1.0
    
    # Calculate Manhattan distance to goal
    d_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If agent is at goal, value is 1.0
    if d_goal == 0:
        return 1.0
    
    # Calculate minimum Manhattan distance to any hole
    d_hole = float('inf')
    for hole in hole_positions:
        dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if dist < d_hole:
            d_hole = dist
    
    # If agent is on a hole, value is 0.0
    if d_hole == 0:
        return 0.0
    
    # Base value based on distance to goal (inverse relationship)
    # Closer to goal = higher value
    base_value = 1.0 / (1.0 + d_goal)
    
    # Safety factor based on proximity to holes
    # Closer to hole = lower value
    # d_hole > 0 is guaranteed here
    safety_factor = d_hole / (d_hole + 1.0)
    
    # Combined value estimate
    estimated_value = base_value * safety_factor
    
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, estimated_value))