import math

def signal_function(state: str) -> float:
    # Parse the grid from the text representation
    lines = state.strip().split('\n')
    grid = [line for line in lines if line]
    
    agent_pos = None
    goal_pos = None
    
    # Scan grid to find agent and goal positions
    for r, row in enumerate(grid):
        if '@' in row:
            agent_pos = (r, row.index('@'))
        if 'G' in row:
            goal_pos = (r, row.index('G'))
            
    # If no agent is found, value is 0
    if agent_pos is None:
        return 0.0
    
    ra, ca = agent_pos
    
    # If goal is not explicitly marked, assume agent is on the goal
    if goal_pos is None:
        return 1.0
    
    rg, cg = goal_pos
    
    # If agent coordinates match goal coordinates
    if agent_pos == goal_pos:
        return 1.0
    
    # Calculate Manhattan distance as a proxy for steps to goal
    dist = abs(ra - rg) + abs(ca - cg)
    
    # Discount factor for delayed rewards (standard in RL)
    gamma = 0.9
    
    # Base value estimate: gamma^distance
    value = math.pow(gamma, dist)
    
    # Local risk assessment: Check for holes adjacent to the agent
    # Neighbors: up, down, left, right
    neighbors = [
        (ra - 1, ca), (ra + 1, ca),
        (ra, ca - 1), (ra, ca + 1)
    ]
    
    hole_count = 0
    valid_neighbor_count = 0
    
    for nr, nc in neighbors:
        # Check if neighbor is within grid bounds
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
            cell = grid[nr][nc]
            valid_neighbor_count += 1
            if cell == 'H':
                hole_count += 1
    
    # Apply penalty for proximity to holes
    # Higher hole proximity reduces the estimated value
    if valid_neighbor_count > 0:
        risk_factor = 1.0 - (hole_count * 0.15)
        value *= risk_factor
    
    # Ensure value is within valid range [0.0, 1.0]
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
        
    return value