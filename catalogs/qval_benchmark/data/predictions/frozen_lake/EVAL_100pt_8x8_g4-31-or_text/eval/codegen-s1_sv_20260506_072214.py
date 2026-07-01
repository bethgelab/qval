def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a frozen lake grid environment.
    The value is based on the probability of reaching the goal (G) from 
    the current agent position (@), given optimal play.
    """
    # Parse the text representation into a grid
    lines = [line for line in state.split('\n') if line.strip()]
    if not lines:
        return 0.0
    
    grid = lines
    rows = len(grid)
    
    ax, ay = -1, -1
    gx, gy = -1, -1
    holes = []
    
    # Find coordinates of the agent (@), goal (G), and holes (H)
    for r in range(rows):
        line = grid[r]
        for c in range(len(line)):
            char = line[c]
            if char == '@':
                ax, ay = r, c
            elif char == 'G':
                gx, gy = r, c
            elif char == 'H':
                holes.append((r, c))
    
    # If agent or goal is missing, state value is 0
    if ax == -1 or gx == -1:
        return 0.0
    
    # If the agent has already reached the goal
    if ax == gx and ay == gy:
        return 1.0
        
    # Base value calculation: higher value for states closer to the goal
    dist = abs(ax - gx) + abs(ay - gy)
    # Start with a value close to 1.0, penalizing slightly for distance
    val = 1.0 - (dist / 100.0)
    
    # Helper function to determine if a position is completely trapped by holes or boundaries
    def is_trapped(r, c):
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows:
                line = grid[nr]
                if 0 <= nc < len(line):
                    if line[nc] != 'H':
                        return False
        return True

    # If either the agent or the goal is trapped, the value is 0
    if is_trapped(ax, ay) or is_trapped(gx, gy):
        return 0.0
        
    # Penalize based on the density and position of holes relative to the path
    for hr, hc in holes:
        # Penalize holes that lie within the bounding box between agent and goal
        if min(ax, gx) <= hr <= max(ax, gx) and min(ay, gy) <= hc <= max(ay, gy):
            val -= 0.02
        # Penalize holes immediately adjacent to the agent (increases risk)
        if abs(hr - ax) + abs(hc - ay) == 1:
            val -= 0.05
        # Penalize holes immediately adjacent to the goal (creates a bottleneck)
        if abs(hr - gx) + abs(hc - gy) == 1:
            val -= 0.05
            
    # Ensure the returned value is clamped between 0.0 and 1.0
    return max(0.0, min(1.0, val))