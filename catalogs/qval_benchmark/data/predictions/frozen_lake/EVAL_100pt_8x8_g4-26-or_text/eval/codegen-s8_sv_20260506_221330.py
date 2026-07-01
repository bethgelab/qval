from collections import deque

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) of the current grid configuration.
    The value is estimated as the discounted reward (gamma^d) based on the shortest 
    path distance from the agent '@' to the goal 'G', avoiding holes 'H'.
    """
    # Parse the grid from the string representation
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        row = []
        # Only include relevant characters for the frozen lake grid
        for char in line.strip():
            if char in ('@', 'G', 'H', '.', 'S'):
                row.append(char)
        if row:
            grid.append(row)
            
    if not grid:
        return 0.0
        
    rows = len(grid)
    cols = len(grid[0])
    
    # Identify the current position of the agent and the goal
    agent_pos = None
    goal_pos = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '@':
                agent_pos = (r, c)
            elif grid[r][c] == 'G':
                goal_pos = (r, c)
                
    # If the agent or goal is not found, the value is 0.0
    if not agent_pos or not goal_pos:
        return 0.0
        
    # BFS to find the shortest path in the deterministic grid (optimal play)
    # We treat 'H' (holes) as impassable obstacles.
    queue = deque([(agent_pos[0], agent_pos[1], 0)])
    visited = {agent_pos}
    
    while queue:
        r, c, dist = queue.popleft()
        
        # Check if we've reached the goal
        if (r, c) == goal_pos:
            # The value is the discounted reward. Reaching the goal in fewer steps
            # is preferred, modeled by gamma^dist. We use gamma = 0.9.
            # The episode limit is 30 steps.
            if dist <= 30:
                return 0.9 ** dist
            else:
                return 0.0
                
        # Explore 4-directional neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] != 'H' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
                    
    # If no path to the goal exists, the value is 0.0
    return 0.0