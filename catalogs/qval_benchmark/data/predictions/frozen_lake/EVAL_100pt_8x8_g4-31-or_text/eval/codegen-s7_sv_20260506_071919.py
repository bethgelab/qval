import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake agent.
    The value is approximated based on the shortest path distance to the goal,
    avoiding holes, and applying a discount factor.
    """
    # Parse the grid from the state text
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        # Remove whitespace to get the grid characters
        row = [c for c in line if not c.isspace()]
        if row:
            grid.append(row)
            
    if not grid:
        return 0.0
        
    rows = len(grid)
    start_pos = None
    goal_pos = None
    holes = set()
    
    # Identify key locations: Agent (@), Goal (G), Holes (H), and Start (S)
    for r in range(rows):
        for c in range(len(grid[r])):
            char = grid[r][c]
            if char == '@':
                start_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.add((r, c))
            elif char == 'S' and start_pos is None:
                start_pos = (r, c)
                
    # If the agent is at the goal, the episode ends with a reward of 1.0.
    # However, in many representations, 'G' is replaced by '@' when the agent is on it.
    # Without knowing the goal's coordinates, we assume the goal is identifiable.
    if start_pos is None:
        return 0.0
    if goal_pos is None:
        # If no 'G' is found, it's impossible to know if the agent is on the goal
        # or if the goal is simply not present in the current view.
        return 0.0

    # Breadth-First Search (BFS) to find the shortest path distance to the goal
    # avoiding all hole cells. This is an efficient way to estimate distance in a grid.
    queue = collections.deque([(start_pos, 0)])
    visited = {start_pos}
    
    while queue:
        (r, c), dist = queue.popleft()
        
        # If we reach the goal cell, return the discounted reward
        if (r, c) == goal_pos:
            # Respect the episode step limit of 30. 
            # A discount factor of 0.95 is used to reflect preference for efficiency.
            return (0.95 ** dist) if dist <= 30 else 0.0
            
        # Explore neighbors (Up, Down, Left, Right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            # Check boundaries and ensure the cell is not a hole and not visited
            if 0 <= nr < rows and 0 <= nc < len(grid[nr]):
                if (nr, nc) not in visited and (nr, nc) not in holes:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
                    
    # If no path to the goal exists, the expected value is 0.0
    return 0.0