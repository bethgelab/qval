import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a frozen lake 8x8 grid.
    The value is approximated by gamma^d, where d is the shortest path distance
    from the agent (@) to the goal (G), avoiding holes (H).
    """
    # 1. Parse the 8x8 grid from the state string.
    # We look for lines that contain exactly 8 non-whitespace characters.
    grid_rows = []
    for line in state.split('\n'):
        row = [c for c in line if not c.isspace()]
        if len(row) == 8:
            grid_rows.append(row)
            
    if len(grid_rows) < 8:
        # Fallback: If no 8-char rows are found, try a more liberal parsing 
        # of characters from the grid description.
        chars = [c for c in state if c in 'FHSG@']
        if len(chars) == 64:
            grid_rows = [chars[i:i+8] for i in range(0, 64, 8)]
        else:
            return 0.0
            
    # Use the first 8 valid rows identified.
    grid = grid_rows[:8]
    
    # 2. Locate the agent's position (@), the goal (G), and identify holes (H).
    start_pos = None
    goal_pos = None
    
    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char == '@':
                start_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
                
    # If the agent '@' is on the goal 'G', we handle the overlap.
    # However, the prompt implies '@' and 'G' are distinct characters.
    # If '@' is not found, we can't determine the current state.
    if start_pos is None or goal_pos is None:
        # Search for '@' or 'G' if they are not found in the standard 8x8 scan.
        # (This is a fallback for slightly malformed ASCII)
        return 0.0

    # 3. BFS to find the shortest path from agent to goal, avoiding holes.
    # We use BFS because it is optimal for finding the shortest path in an unweighted grid.
    queue = collections.deque([(start_pos[0], start_pos[1], 0)])
    visited = {start_pos}
    
    while queue:
        r, c, dist = queue.popleft()
        
        # Check if we reached the goal.
        if (r, c) == goal_pos:
            # Use an approximation of the discounted value: V(s) = gamma^d.
            # We assume gamma = 0.95.
            return 0.95 ** dist
        
        # If within step limit, explore neighbors.
        if dist < 30:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    # Any cell that is not a hole 'H' is considered traversable.
                    if grid[nr][nc] != 'H' and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))
                        
    # If no path is found to the goal within the step limit.
    return 0.0