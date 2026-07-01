import math

def signal_function(state: str) -> float:
    lines = state.splitlines()
    if not lines:
        return 0.0
    
    rows = len(lines)
    cols = len(lines[0]) if rows > 0 else 0
    
    agent_pos = None
    goal_pos = None
    
    # Scan for Agent (@) and Goal (G)
    for r in range(rows):
        for c in range(len(lines[r])):
            char = lines[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
    
    # If goal or agent not found, value is 0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    r_a, c_a = agent_pos
    r_g, c_g = goal_pos
    
    # Calculate Manhattan distance
    dist = abs(r_a - r_g) + abs(c_a - c_g)
    
    # Base value decay based on distance (exponential decay simulates discounting)
    # Max distance on 8x8 is 14. exp(-14 * 0.2) approx 0.06
    base_value = math.exp(-dist * 0.2)
    
    # Penalty for adjacent holes (safety risk)
    hole_neighbors = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r_a + dr, c_a + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if lines[nr][nc] == 'H':
                hole_neighbors += 1
    
    # Reduce value if holes are nearby (up to 50% reduction for 2+ adjacent holes)
    safety_factor = 1.0 - (0.15 * hole_neighbors)
    
    value = base_value * safety_factor
    
    return max(0.0, value)