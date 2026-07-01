import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value of a Frozen Lake environment using a shortest-path 
    heuristic and a proximity-to-hole penalty.
    """
    # Parse the ASCII grid representation
    lines = [line.strip() for line in state.strip().split('\n') if line.strip()]
    if not lines:
        return 0.0
    
    grid = []
    for line in lines:
        # Handle both space-separated and non-space-separated grid strings
        row = line.split()
        if len(row) == 1 and len(row[0]) > 1:
            row = list(row[0])
        grid.append(row)
    
    if not grid:
        return 0.0
        
    rows = len(grid)
    cols = len(grid[0])
    
    agent_pos = None
    goal_pos = None
    
    # Locate the agent (@), goal (G), and holes (H)
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
    
    # Handle edge cases where the agent might have reached the goal 
    # (the '@' character may have replaced the 'G' character)
    if agent_pos is not None and goal_pos is None:
        return 1.0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # If the agent is already at the goal, return max reward
    if agent_pos == goal_pos:
        return 1.0

    # Use Breadth-First Search (BFS) to find the shortest path distance to the goal
    # This is a static property of the current grid configuration
    queue = collections.deque([(agent_pos[0], agent_pos[1], 0)])
    visited = {agent_pos}
    shortest_dist = -1
    
    while queue:
        r, c, d = queue.popleft()
        if (r, c) == goal_pos:
            shortest_dist = d
            break
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Only traverse tiles that are not holes
                if grid[nr][nc] != 'H' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, d + 1))
    
    # If no path to the goal exists or it's beyond the step limit, value is 0
    if shortest_dist == -1 or shortest_dist >= 30:
        return 0.0
    
    # Base heuristic: A linear decay from 1.0 (at d=0) to 0.0 (at d=30)
    # This reflects the prompt's preference for efficiency (fewer steps)
    val = (30.0 - shortest_dist) / 30.0
    
    # Safety factor: Penalize states where the agent is adjacent to a hole
    # This accounts for the risk of falling into a hole due to stochasticity (slippery)
    danger = 0
    ar, ac = agent_pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 'H':
                danger += 1
    
    # Apply the danger penalty (up to 60% reduction if surrounded by holes)
    val *= (1.0 - (danger * 0.15))
    
    return max(0.0, min(1.0, val))