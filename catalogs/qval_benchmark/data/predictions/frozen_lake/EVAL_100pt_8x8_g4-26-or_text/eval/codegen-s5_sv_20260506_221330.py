import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) of a frozen lake grid state.
    The value is estimated based on the shortest path distance to the goal,
    accounting for obstacles (holes).
    """
    grid = []
    lines = state.strip().split('\n')
    for line in lines:
        # Extract all potential cell characters from each line.
        # S: Start, G: Goal, H: Hole, @: Agent, .: Path
        row = [char for char in line if char in 'SHG.@.']
        if row:
            grid.append(row)
    
    if not grid:
        return 0.0
        
    rows = len(grid)
    cols = len(grid[0])
    
    agent_pos = None
    goal_pos = None
    
    # Find the positions of the agent and the goal.
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            # In cases where the agent is at the starting position and marked 'S'.
            elif cell == 'S' and agent_pos is None:
                agent_pos = (r, c)
    
    # Fallback search if the goal wasn't identified in the first pass.
    if not agent_pos or not goal_pos:
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 'G':
                    goal_pos = (r, c)
                if grid[r][c] == '@' and agent_pos is None:
                    agent_pos = (r, c)
                if grid[r][c] == 'S' and agent_pos is None:
                    agent_pos = (r, c)
                    
    if not agent_pos or not goal_pos:
        return 0.0

    # BFS to calculate the shortest path distance from agent to goal.
    # This is a direct analysis of the current state's connectivity.
    queue = collections.deque([(agent_pos, 0)])
    visited = {agent_pos}
    
    while queue:
        (r, c), dist = queue.popleft()
        
        # If goal is reached, return a value that decays with distance.
        if (r, c) == goal_pos:
            # Using a discount-like factor to favor shorter paths.
            # 0.95 is used to provide a smooth decay for an 8x8 grid.
            return 0.95 ** dist
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            # Check bounds, obstacle avoidance, and visitation.
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] != 'H' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
                    
    # If no path to the goal exists, the value is 0.
    return 0.0