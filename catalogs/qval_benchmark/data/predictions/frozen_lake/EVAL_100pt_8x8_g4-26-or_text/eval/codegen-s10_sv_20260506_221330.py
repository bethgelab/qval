import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake grid state.
    The estimate is based on the shortest path distance to the goal,
    accounting for the step limit and using a discount factor to reward efficiency.
    """
    # Parse the grid from the text representation
    lines = [line for line in state.splitlines() if line.strip()]
    if not lines:
        return 0.0
    
    rows = len(lines)
    agent_pos = None
    goal_pos = None
    holes = set()
    
    for r in range(rows):
        row_str = lines[r]
        for c in range(len(row_str)):
            char = row_str[c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.add((r, c))
                
    # If the agent's current position or the goal cannot be determined, return 0
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    # Iterative Breadth-First Search (BFS) to find the shortest path to the goal
    # This is an efficient way to approximate the optimal value in a static grid.
    queue = collections.deque([(agent_pos, 0)])
    visited = {agent_pos}
    
    while queue:
        (r, c), dist = queue.popleft()
        
        # Check if we have reached the goal
        if (r, c) == goal_pos:
            # If the path exceeds the step limit, the value is effectively 0
            if dist <= 30:
                # Return gamma^dist where gamma is the discount factor (assumed 0.9)
                return 0.9 ** dist
            else:
                return 0.0
        
        # Explore adjacent cells (Up, Down, Left, Right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            # Ensure the new position is within grid boundaries
            if 0 <= nr < rows and 0 <= nc < len(lines[nr]):
                # Only move to cells that are not holes and haven't been visited
                if (nr, nc) not in holes and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
                    
    # If no path to the goal exists, the value is 0
    return 0.0