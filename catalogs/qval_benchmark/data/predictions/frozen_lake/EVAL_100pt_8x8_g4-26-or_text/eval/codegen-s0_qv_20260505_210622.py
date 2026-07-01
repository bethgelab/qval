import collections

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    The estimation is based on the distance to the goal and the presence of holes.
    """
    def get_info(grid_str):
        lines = grid_str.strip().split('\n')
        rows = []
        for line in lines:
            row = [c for c in line if not c.isspace()]
            if row:
                rows.append(row)
        
        # Handle cases where the grid might be a single flattened line of 64 characters
        if len(rows) == 1 and len(rows[0]) == 64:
            flat = rows[0]
            rows = [flat[i:i+8] for i in range(0, 64, 8)]
            
        agent_pos = None
        goal_pos = None
        holes = set()
        
        for r in range(len(rows)):
            for c in range(len(rows[r])):
                char = rows[r][c]
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.add((r, c))
        return agent_pos, goal_pos, holes, rows

    # Get grid information for both current and next states
    a_state, g_state, h_state, rows_state = get_info(state)
    a_next, g_next, h_next, rows_next = get_info(next_state)
    
    # The actual goal position is where G was, or where '@' is if G was replaced
    goal_pos = g_next if g_next else g_state
    # The actual holes are the set of all known hole positions
    holes = h_state.union(h_next)
    # The agent's position in the next state
    agent_pos = a_next
    
    # Safety check for invalid parsing
    if agent_pos is None:
        return 0.0
        
    # If the agent reached the goal
    if goal_pos is not None and agent_pos == goal_pos:
        return 1.0
        
    # If the agent fell into a hole
    if agent_pos in holes:
        return 0.0
        
    # Determine grid dimensions (using state as a reliable template)
    R = len(rows_state)
    if R == 0:
        return 0.0
    C = len(rows_state[0])

    # Use BFS to find the shortest path to the goal from the agent's current position
    # We treat the state grid as the physical map of the environment.
    queue = collections.deque([(agent_pos, 0)])
    visited = {agent_pos}
    
    while queue:
        (r, c), dist = queue.popleft()
        
        if (r, c) == goal_pos:
            # Use a decay factor to reward shorter paths
            return 0.9 ** dist
        
        # Explore adjacent cells
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                if (nr, nc) not in visited and (nr, nc) not in holes:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
                    
    # If no path is found to the goal
    return 0.0