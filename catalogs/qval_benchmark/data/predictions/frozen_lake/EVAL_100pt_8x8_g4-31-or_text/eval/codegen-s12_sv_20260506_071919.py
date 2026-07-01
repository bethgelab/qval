def signal_function(state: str) -> float:
    if not state or not state.strip():
        return 0.0
    
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        row = list(line.strip())
        if row:
            grid.append(row)
            
    if not grid:
        return 0.0
        
    rows = len(grid)
    cols = len(grid[0])
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    if agent_pos == goal_pos:
        return 1.0
        
    # Calculate Manhattan distance as a proxy for the optimal path length
    d_g = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If the Manhattan distance exceeds the step limit, the value is zero
    if d_g > 30:
        return 0.0
        
    # Calculate distance to the nearest hole
    min_d_h = float('inf')
    for hr, hc in holes:
        dist = abs(agent_pos[0] - hr) + abs(agent_pos[1] - hc)
        if dist < min_d_h:
            min_d_h = dist
            
    if min_d_h == 0:
        return 0.0
        
    # Check if the agent is immediately surrounded by holes
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
            
    if neighbors and all(grid[nr][nc] == 'H' for nr, nc in neighbors):
        return 0.0
        
    # The value function approximates gamma^d_g, where gamma is a discount factor.
    # We use 0.9 as a representative discount to reflect efficiency.
    val = 0.9 ** d_g
    
    # Proximity to holes acts as a penalty, even if the agent is playing optimally.
    # This accounts for the fact that in many grid worlds, being near an obstacle
    # reduces the "safety" or "quality" of the state.
    if min_d_h == 1:
        val *= 0.5
        
    return float(val)