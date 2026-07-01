import collections

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given state in a Frozen Lake grid.
    The value is estimated based on the shortest path to the goal 'G'
    avoiding holes 'H', using a discount factor for efficiency.
    """
    # 1. Parse the grid from the ASCII text
    # We're looking for an 8x8 block of characters in the set 'SFHG@'
    grid_rows = []
    for line in state.splitlines():
        # Clean the line to focus on grid elements
        row = [c for c in line if c in 'SFHG@']
        if len(row) >= 8:
            # Take only the first 8 characters if the row is longer
            grid_rows.append(row[:8])
        elif len(row) == 8:
            grid_rows.append(row)

    # Look for a contiguous 8x8 block that contains both the agent '@' and the goal 'G'
    grid = []
    if len(grid_rows) >= 8:
        for i in range(len(grid_rows) - 7):
            candidate = grid_rows[i:i+8]
            has_agent = any('@' in r for r in candidate)
            has_goal = any('G' in r for r in candidate)
            if has_agent and has_goal:
                grid = candidate
                break
        if not grid:
            # Fallback to the first 8 rows if no perfect block is found
            grid = grid_rows[:8]

    # If we cannot find a valid 8x8 grid, the value is 0.0
    if len(grid) < 8:
        return 0.0

    # 2. Locate agent position and goal position
    agent_pos = None
    goal_pos = None
    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)

    # If agent or goal cannot be found, return 0.0
    if agent_pos is None or goal_pos is None:
        return 0.0

    # If the agent is already at the goal
    if agent_pos == goal_pos:
        return 1.0

    # 3. Use BFS to find the shortest path distance from agent to goal, avoiding holes 'H'
    # A well-reasoned value V(s) for a sparse reward is gamma^dist
    queue = collections.deque([(agent_pos, 0)])
    visited = {agent_pos}
    
    # We'll use a discount factor to reward shorter paths
    gamma = 0.9

    while queue:
        (r, c), dist = queue.popleft()

        # If we reached the goal
        if (r, c) == goal_pos:
            return gamma ** dist

        # Explore 4-way neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                # Can only move to cells that are NOT holes
                if grid[nr][nc] != 'H' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))

    # If no path to the goal exists, the value is 0.0
    return 0.0