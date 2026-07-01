import math

def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().split('\n')
    grid = [line.strip() for line in lines if line.strip()]
    
    if not grid:
        return 0.0
    
    # Find positions
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    # If goal is missing, value is 0
    if goal_pos is None:
        return 0.0
    
    # If agent is missing, assume it cannot reach goal
    if agent_pos is None:
        return 0.0
    
    # If agent is on goal, value is 1.0
    if agent_pos == goal_pos:
        return 1.0
    
    # Calculate Manhattan distance
    r_a, c_a = agent_pos
    r_g, c_g = goal_pos
    dist = abs(r_a - r_g) + abs(c_a - c_g)
    
    # Base value based on distance (inverse relationship)
    # Max distance on 8x8 is 14. We want value to decay but remain positive.
    # Coefficient 0.3 ensures reasonable decay.
    base_value = 1.0 / (1.0 + 0.3 * dist)
    
    # Check if goal is surrounded by holes (unreachable)
    # Check 4-neighbors of goal
    goal_neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r_g + dr, c_g + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
            if grid[nr][nc] == 'H':
                goal_neighbors.append('H')
            elif grid[nr][nc] != '.':
                goal_neighbors.append('O') # Obstacle (boundary or other)
            else:
                goal_neighbors.append('.')
    
    # If all valid neighbors are holes or boundaries, goal is blocked
    # We consider boundaries as obstacles too.
    valid_neighbors = [n for n in goal_neighbors if n in ['.', 'H']]
    if valid_neighbors and all(n == 'H' for n in valid_neighbors):
        return 0.0
    
    # Apply penalties
    value = base_value
    
    # Penalty for holes adjacent to agent
    adjacent_holes = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r_a + dr, c_a + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
            if grid[nr][nc] == 'H':
                adjacent_holes += 1
    
    if adjacent_holes > 0:
        value *= (1.0 - 0.1 * adjacent_holes)
    
    # Penalty for holes in the bounding box between agent and goal
    # This approximates path difficulty
    hole_count_in_box = 0
    min_r, max_r = min(r_a, r_g), max(r_a, r_g)
    min_c, max_c = min(c_a, c_g), max(c_a, c_g)
    
    for r, c in holes:
        if min_r <= r <= max_r and min_c <= c <= max_c:
            hole_count_in_box += 1
            
    if hole_count_in_box > 0:
        value *= (1.0 - 0.05 * hole_count_in_box)
    
    # Ensure value is within [0.0, 1.0]
    return max(0.0, min(1.0, value))