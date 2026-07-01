import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an 8x8 Frozen Lake environment.
    The value is based on the Manhattan distance to the goal and the 
    density of holes on the map, approximating the probability of 
    reaching the goal safely and efficiently.
    """
    # Parse the ASCII grid from the state string
    lines = state.strip().split('\n')
    grid = [line.strip() for line in lines if line.strip()]
    
    if not grid:
        return 0.0
    
    agent_pos = None
    goal_pos = None
    hole_count = 0
    
    # Scan the grid for the agent (@), goal (G), and holes (H)
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                hole_count += 1
                
    # If agent or goal is missing, we cannot estimate a path; return 0
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    # Manhattan distance to the goal
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If the agent is already at the goal, the value is 1.0
    if dist == 0:
        return 1.0
        
    # Hole density: probability of a random cell being a hole
    # The grid is 8x8 = 64 cells.
    rho = hole_count / 64.0
    
    # Survival probability: The probability of not hitting a hole in one step
    # approximated by (1 - rho).
    # We also incorporate a decay factor (gamma) to reward efficiency.
    gamma = 0.98
    survival_prob_per_step = (1.0 - rho)
    
    # Estimated Value V(s) = (gamma * survival_prob)^dist
    # This reflects the probability of reaching the goal over 'dist' steps
    # while penalizing longer paths.
    estimated_value = math.pow(gamma * survival_prob_per_step, dist)
    
    # Ensure the value is bounded between 0 and 1
    return max(0.0, min(1.0, estimated_value))