def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given state in a Frozen Lake environment.
    The value is an approximation of the probability of reaching the goal
    before falling into a hole or exceeding the step limit.
    """
    # Parse the state string into a 2D grid
    lines = [line for line in state.strip().split('\n') if line.strip()]
    grid = []
    for line in lines:
        # Remove all whitespace to handle both "S . . ." and "S..." formats
        row = [char for char in line if not char.isspace()]
        if row:
            grid.append(row)

    if not grid:
        return 0.0

    rows = len(grid)
    cols = len(grid[0])
    agent_pos = None
    goal_pos = None

    # Locate agent, goal, and identify agent's current position
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif agent_pos is None and char == 'S':
                # In some representations, 'S' might be the initial position
                agent_pos = (r, c)
            
            if char == 'G':
                goal_pos = (r, c)

    # If the agent or goal is not found in the text representation
    if agent_pos is None or goal_pos is None:
        return 0.0

    ar, ac = agent_pos
    gr, gc = goal_pos

    # If the agent is currently on a hole (though episode would have ended)
    if grid[ar][ac] == 'H':
        return 0.0

    # If the agent has reached the goal
    if ar == gr and ac == gc:
        return 1.0

    # Manhattan distance as a proxy for steps needed
    dist = abs(ar - gr) + abs(ac - gc)

    # If the goal is beyond the step limit (30)
    if dist > 30:
        return 0.0

    # Baseline value: the probability of reaching the goal decays with distance.
    # A linear decay is used: as dist approaches 30, value approaches 0.
    val = (31.0 - dist) / 31.0

    # Risk factor: proximity to holes.
    # In a stochastic environment (Frozen Lake), being near a hole 
    # increases the risk of slipping into it.
    risk = 0.0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 'H':
                # Each adjacent hole reduces the expected value
                risk += 0.20

    # Apply the risk penalty to the baseline value
    val *= (1.0 - risk)

    # Ensure the value is within the valid [0, 1] range
    return float(max(0.0, min(1.0, val)))