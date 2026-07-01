def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and resulting next state
    in a frozen lake environment.
    """

    def get_grid(s: str):
        grid = []
        # Split into lines to handle the grid row by row
        lines = s.strip().split('\n')
        for line in lines:
            row = []
            for char in line:
                # Only extract relevant characters that define the map or agent position
                if char in ('@', 'G', 'H', 'S', 'F', 'L'):
                    row.append(char)
            if row:
                grid.append(row)
        return grid

    grid_s = get_grid(state)
    grid_ns = get_grid(next_state)

    holes = set()
    goal_pos = None
    agent_pos_old = None

    # Extract the map layout from the current state grid.
    # The state grid reliably contains the positions of holes and the goal,
    # as the agent is currently at agent_pos_old.
    for r in range(len(grid_s)):
        for c in range(len(grid_s[r])):
            char = grid_s[r][c]
            if char == 'H':
                holes.add((r, c))
            elif char == 'G':
                goal_pos = (r, c)
            elif char == '@':
                agent_pos_old = (r, c)

    # If the agent was already at the goal (terminal state), 
    # return 1.0. This is an edge case handling.
    if agent_pos_old is not None and goal_pos is not None and agent_pos_old == goal_pos:
        return 1.0

    # Identify the agent's position in the resulting next_state grid.
    agent_pos_new = None
    for r in range(len(grid_ns)):
        for c in range(len(grid_ns[r])):
            if grid_ns[r][c] == '@':
                agent_pos_new = (r, c)
                break
        if agent_pos_new:
            break

    # If the agent's position cannot be determined, return 0.0.
    if agent_pos_new is None:
        return 0.0

    # If the move resulted in the agent falling into a hole.
    if agent_pos_new in holes:
        return 0.0

    # If the move resulted in reaching the goal.
    if goal_pos is not None and agent_pos_new == goal_pos:
        return 1.0

    # If the agent is in a safe tile, estimate the value based on the
    # distance to the goal. A common approach is using a discount factor 
    # raised to the power of the distance.
    if goal_pos is not None:
        d = abs(agent_pos_new[0] - goal_pos[0]) + abs(agent_pos_new[1] - goal_pos[1])
        # A discount factor of 0.9 provides a reasonable heuristic for Q-value estimation.
        # As distance d increases, the expected return decreases.
        return 0.9 ** d

    return 0.0