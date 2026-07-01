def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    The estimate is based on distance to the goal and whether the agent has reached the goal or fallen into a hole.
    """
    def parse_grid(grid_str: str):
        # Parses the grid string into a 2D list of characters
        lines = [line.strip() for line in grid_str.split('\n') if line.strip()]
        return [list(line) for line in lines]

    grid_s = parse_grid(state)
    grid_n = parse_grid(next_state)
    
    if not grid_s or not grid_n:
        return 0.0

    rows = len(grid_s)
    cols = len(grid_s[0])

    # 1. Locate the agent in the current state and the next state
    r_old, c_old = -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid_s[r][c] == '@':
                r_old, c_old = r, c
                break
        if r_old != -1:
            break

    r_new, c_new = -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid_n[r][c] == '@':
                r_new, c_new = r, c
                break
        if r_new != -1:
            break

    # 2. Locate the goal position (G)
    # The goal position is static, but it might be covered by the agent '@' in either grid.
    goal_pos = None
    for r in range(rows):
        for c in range(cols):
            if grid_s[r][c] == 'G' or grid_n[r][c] == 'G':
                goal_pos = (r, c)
                break
        if goal_pos:
            break
    
    # If 'G' is not found in either grid, the agent must be standing on 'G'.
    if not goal_pos:
        goal_pos = (r_new, c_new)

    # 3. Determine the immediate outcome of the move
    # If the agent's new position is the goal, the reward is 1.0
    if (r_new, c_new) == goal_pos:
        return 1.0

    # Determine the cell type of the new position
    # If the agent moved, the cell type is what was at that position in the original state.
    # If the agent did not move, we assume it remains in a safe (frozen) state.
    if r_new == r_old and c_new == c_old:
        cell_type = 'F'
    else:
        # Bounds check just in case
        if 0 <= r_new < rows and 0 <= c_new < cols:
            cell_type = grid_s[r_new][c_new]
        else:
            cell_type = 'F'

    # If the new position is a hole, the reward is 0.0
    if cell_type == 'H':
        return 0.0
    
    # 4. Heuristic for safe states (F or S)
    # Use a discounted distance-based heuristic: Q(s, a) is proportional to the inverse of distance to the goal.
    # A decay of 0.9 per Manhattan distance step approximates a discounted reward signal.
    if goal_pos:
        dist = abs(r_new - goal_pos[0]) + abs(c_new - goal_pos[1])
        # Return a value that decreases as distance increases
        return float(0.9 ** dist)
    
    return 0.0