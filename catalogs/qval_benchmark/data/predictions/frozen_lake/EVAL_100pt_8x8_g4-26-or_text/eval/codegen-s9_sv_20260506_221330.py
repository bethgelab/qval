import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given frozen lake grid state.
    The value is approximated by the discounted distance to the goal,
    accounting for the shortest path that avoids holes.
    """
    # 1. Parse the grid from the ASCII representation
    grid = []
    for line in state.strip().splitlines():
        # Extract non-whitespace characters to build the grid rows
        row = [c for c in line if not c.isspace()]
        if row:
            grid.append(row)
    
    if not grid:
        return 0.0
        
    rows_count = len(grid)
    agent_pos = None
    goal_pos = None
    holes = set()
    
    # 2. Locate the agent (@), goal (G), and holes (H)
    for r in range(rows_count):
        for c in range(len(grid[r])):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.add((r, c))
                
    # If the agent '@' is not explicitly present, check for the start tile 'S'
    if agent_pos is None:
        for r in range(rows_count):
            for c in range(len(grid[r])):
                if grid[r][c] == 'S':
                    agent_pos = (r, c)
                    break
            if agent_pos:
                break
                
    # If essential positions are missing, the value is zero
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    # If the agent is already at the goal
    if agent_pos == goal_pos:
        return 1.0
        
    # 3. Use BFS to find the shortest path distance from agent to goal avoiding holes
    # This provides an informed estimate of the expected reward.
    queue = collections.deque([(agent_pos, 0)])
    visited = {agent_pos}
    
    while queue:
        (curr_r, curr_c), dist = queue.popleft()
        
        # If we have reached the goal, calculate the discounted reward
        if (curr_r, curr_c) == goal_pos:
            # A step limit of 30 is given. A discount factor (e.g., 0.9) 
            # encourages efficiency and reflects the diminishing probability 
            # of reaching the goal in stochastic environments.
            if dist <= 30:
                return 0.9 ** dist
            else:
                return 0.0
        
        # Explore 4-connectivity (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr_r + dr, curr_c + dc
            
            # Check boundaries and ensure the cell is not a hole and not visited
            if 0 <= nr < rows_count and 0 <= nc < len(grid[nr]):
                if grid[nr][nc] != 'H' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
                    
    # If no path to the goal is found within the grid
    return 0.0