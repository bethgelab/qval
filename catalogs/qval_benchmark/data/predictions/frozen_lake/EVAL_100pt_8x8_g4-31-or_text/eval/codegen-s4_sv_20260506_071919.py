def signal_function(state: str) -> float:
    import collections

    # Parse the ASCII grid from the input string
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        row = []
        for char in line:
            if char in ('@', 'G', 'H', 'S'):
                row.append(char)
            elif char in ('.', ' ', '\t'):
                row.append('.')
            else:
                # Treat any other character as a traversable cell
                row.append(char)
        if row:
            grid.append(row)

    if not grid:
        return 0.0

    # Locate the agent ('@') and the goal ('G')
    start_pos = None
    goal_pos = None
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == '@':
                start_pos = (r, c)
            elif grid[r][c] == 'G':
                goal_pos = (r, c)

    # If the agent or goal is not present in the grid, return 0.0
    if start_pos is None or goal_pos is None:
        return 0.0

    # Use Breadth-First Search (BFS) to find the shortest path from '@' to 'G'
    # this provides a direct analysis of the grid's navigability.
    queue = collections.deque([(start_pos[0], start_pos[1], 0)])
    visited = {start_pos}

    while queue:
        r, c, dist = queue.popleft()

        # If the goal is reached, estimate the value based on distance
        if (r, c) == goal_pos:
            # Step limit is 30. Efficient paths (shorter distance) are valued higher.
            if dist <= 30:
                # Use a discount factor (e.g., 0.95) to model the expected reward
                return float(0.95 ** dist)
            else:
                return 0.0

        # Explore neighbors (up, down, left, right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            # Check boundaries and ensure the cell is not a hole ('H')
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
                if grid[nr][nc] != 'H' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

    # If no path to 'G' exists, the value is 0.0
    return 0.0