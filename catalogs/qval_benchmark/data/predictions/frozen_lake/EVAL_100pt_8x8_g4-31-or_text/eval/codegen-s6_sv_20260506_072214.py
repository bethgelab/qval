import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the Frozen Lake 8x8 environment.
    The value is based on the distance to the goal and the presence of 
    obstacles (holes) in the direct bounding box between the agent and the goal.
    """
    # Parse the grid representation
    rows = [line.strip() for line in state.splitlines() if line.strip()]
    grid = [row.replace(' ', '') for row in rows]
    
    if not grid or len(grid) < 1:
        return 0.0
        
    # Locate agent (@) and goal (G)
    r, c = -1, -1
    rg, cg = -1, -1
    for row_idx, row in enumerate(grid):
        for col_idx, char in enumerate(row):
            if char == '@':
                r, c = row_idx, col_idx
            elif char == 'G':
                rg, cg = row_idx, col_idx
                
    # If agent or goal is not found, return a baseline low value
    if r == -1 or rg == -1:
        return 0.0
        
    # Manhattan distance is a strong predictor of efficiency (shorter is better)
    dist = abs(r - rg) + abs(c - cg)
    
    # Define the bounding box between the agent and the goal
    r_start, r_end = min(r, rg), max(r, rg)
    c_start, c_end = min(c, cg), max(c, cg)
    
    # Evaluate the density of holes within this bounding box
    # A high density of holes suggests a higher probability that the path is blocked
    holes_in_box = 0
    box_area = 0
    for i in range(r_start, r_end + 1):
        if i >= len(grid): break
        row_str = grid[i]
        for j in range(c_start, c_end + 1):
            if j >= len(row_str): break
            box_area += 1
            if row_str[j] == 'H':
                holes_in_box += 1
    
    # Calculate hole density; higher density lowers the value
    density = holes_in_box / box_area if box_area > 0 else 0.0
    
    # Use a discount factor gamma^dist to prioritize shorter paths
    # gamma = 0.99 is common for RL value functions
    gamma = 0.99
    proximity_value = math.pow(gamma, dist)
    
    # Combine proximity and path clearance
    # The value decreases as the bounding box becomes more cluttered with holes
    estimated_value = proximity_value * (1.0 - density)
    
    # Return the value clipped between 0.0 and 1.0
    return max(0.0, min(1.0, estimated_value))