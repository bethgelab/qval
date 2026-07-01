def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given frozen lake grid.
    The value is based on the Manhattan distance to the goal,
    the proximity to holes, and the step limit.
    """
    import math

    # Parse the grid into a 2D-like structure
    lines = [line.strip() for line in state.split('\n') if line.strip()]
    if not lines:
        return 0.0

    grid = lines
    rows = len(grid)
    cols = len(grid[0])

    agent_pos = None
    goal_pos = None
    holes = []

    # Find the locations of the agent, the goal, and all holes
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))

    # If agent position is not found, return 0.0
    if agent_pos is None:
        return 0.0

    ar, ac = agent_pos

    # Case 1: Agent is at the goal
    # If 'G' was replaced by '@', goal_pos will be None.
    if goal_pos is not None:
        if ar == goal_pos[0] and ac == goal_pos[1]:
            return 1.0
    else:
        # Heuristic: If G is not visible, assume the agent might be on it.
        # This is a risk, but often the only way to identify a terminal goal state.
        # We only assume this if no holes are present, to avoid misidentifying H as G.
        if not holes:
            return 1.0

    # Case 2: Agent is in a hole
    # Check if the agent's position matches any known hole position
    for hr, hc in holes:
        if ar == hr and ac == hc:
            return 0.0

    # Case 3: Standard navigation
    if goal_pos is None:
        # If the goal is not found and we haven't assumed the agent is on it,
        # the goal is unreachable in the current representation.
        return 0.0

    gr, gc = goal_pos
    dist = abs(ar - gr) + abs(ac - gc)

    # Step limit check: The episode limit is 30.
    if dist >= 30:
        return 0.0

    # Approximate the value using a discounted reward model: V(s) ≈ gamma^dist
    # Using gamma = 0.9 as a reasonable default for a sparse reward environment.
    value = 0.9 ** dist

    # Proximity Penalty: Reduce value based on the number of adjacent holes.
    # This models the risk of accidental slips into holes.
    hole_neighbors = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = ar + dr, ac + dc
        # Check if the neighbor is a hole
        for hr, hc in holes:
            if nr == hr and nc == hc:
                hole_neighbors += 1
                break

    # Apply a penalty factor based on how many holes are in the immediate vicinity.
    # 0.15 is a damping factor to prevent the value from dropping too drastically.
    value *= (1.0 - (0.15 * hole_neighbors))

    # Ensure the result is within the valid [0.0, 1.0] range.
    return max(0.0, min(1.0, value))