import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given 8x8 frozen lake grid.
    The estimate is based on the shortest path distance to the goal while avoiding holes,
    incorporating a discount factor.
    """
    # 1. Parse the grid
    # Attempt to identify the 8x8 grid from the text representation.
    grid = []
    lines = [l for l in state.split('\n') if len(l) >= 8]
    
    if len(lines) >= 8:
        # Standard case: The grid is provided in separate lines.
        grid = [l[:8] for l in lines[:8]]
    else:
        # Fallback: The grid might be a single line or have different formatting.
        # Remove newlines but preserve spaces to keep the spatial structure.
        flat = state.replace('\n', '').replace('\r', '')
        idx = flat.find('@')
        if idx == -1:
            idx = flat.find('S')
        
        if idx != -1:
            # Find the 8-aligned start of the 8x8 grid block containing the agent.
            start_idx = (idx // 8) * 8
            # Capture the 64-character window.
            grid_chunk = flat[start_idx : start_idx + 64]
            if len(grid_chunk) == 64:
                for i in range(0, 64, 8):
                    grid.append(grid_chunk[i : i + 8])
            else:
                # If we can't get a full 64 chars, fallback to simple parsing.
                return 0.0
        else:
            return 0.0

    if not grid or len(grid) < 8:
        return 0.0

    # 2. Identify critical locations
    agent_pos = None
    goal_pos = None
    holes = set()
    
    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.add((r, c))
    
    # If '@' was not found, try to find 'S' as a starting position.
    if agent_pos is None:
        for r in range(8):
            for c in range(8):
                if grid[r][c] == 'S':
                    agent_pos = (r, c)
                    break
    
    # If essential positions are missing, the state is uninterpretable.
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # If the agent is currently in a hole, the value is 0.
    if grid[agent_pos[0]][agent_pos[1]] == 'H':
        return 0.0

    # 3. Perform BFS to find the shortest path to the goal avoiding holes.
    # This provides a direct informed approximation of the optimal value.
    queue = collections.deque([(agent_pos[0], agent_pos[1], 0)])
    visited = {agent_pos}
    
    # A discount factor of 0.95 is used to reflect the preference for efficiency.
    gamma = 0.95
    
    while queue:
        r, c, d = queue.popleft()
        
        # Check if we reached the goal.
        if (r, c) == goal_pos:
            return gamma ** d
        
        # If we haven't reached the step limit, explore neighbors.
        if d < 30:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    if (nr, nc) not in visited and grid[nr][nc] != 'H':
                        visited.add((nr, nc))
                        queue.append((nr, nc, d + 1))
                        
    # If no path to goal exists within the step limit.
    return 0.0