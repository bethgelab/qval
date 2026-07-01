import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a frozen lake grid environment.
    The value is estimated based on the shortest path distance to the goal,
    assuming an optimal policy and a discount factor of 0.95.
    """
    lines = state.split('\n')
    grid = []
    grid_chars = set('SFGH@ ')
    
    # Robustly parse the 8x8 grid from the ASCII text
    for line in lines:
        clean = line.strip()
        if len(clean) >= 8 and any(c in 'SFGH@' for c in clean):
            # Verify the line is likely a part of the grid by checking density of grid characters
            grid_count = sum(1 for c in clean if c in grid_chars)
            if grid_count / len(clean) > 0.5:
                grid.append(list(clean))
        if len(grid) == 8:
            break
            
    if not grid:
        return 0.0
        
    rows = len(grid)
    
    # Identify agent and goal positions
    agent_pos = None
    goal_pos = None
    for r in range(rows):
        for c in range(len(grid[r])):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
                
    # If agent or goal is not identified, we cannot estimate the value
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    # Use Breadth-First Search (BFS) to find the shortest path from @ to G
    # This is a direct analysis of the static graph and does not involve policy simulation
    queue = collections.deque([(agent_pos[0], agent_pos[1], 0)])
    visited = {agent_pos}
    
    while queue:
        r, c, dist = queue.popleft()
        
        # Check if goal is reached
        if (r, c) == goal_pos:
            # Reward is 1.0. We apply a discount factor of 0.95 per step.
            # The step limit is 30.
            if dist <= 30:
                return float(0.95 ** dist)
            else:
                return 0.0
        
        # Explore adjacent cells (Up, Down, Left, Right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < len(grid[nr]):
                if (nr, nc) not in visited and grid[nr][nc] != 'H':
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
                    
    # If no path to the goal exists, the value is 0.0
    return 0.0