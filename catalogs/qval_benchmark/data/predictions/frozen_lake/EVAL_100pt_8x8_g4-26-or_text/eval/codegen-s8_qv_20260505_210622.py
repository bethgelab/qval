def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a 8x8 Frozen Lake environment.
    The estimate is based on the proximity to the goal and the safety of the next state.
    """
    def get_grid(s: str):
        grid = []
        for line in s.strip().split('\n'):
            row = [c for c in line if c in 'SFHG@']
            if row:
                grid.append(row)
        return grid

    gs = get_grid(state)
    gns = get_grid(next_state)
    
    if not gs or not gns:
        return 0.0

    # Find current agent position in the previous state
    ar, ac = -1, -1
    for r in range(len(gs)):
        for c in range(len(gs[r])):
            if gs[r][c] == '@':
                ar, ac = r, c
                break
        if ar != -1:
            break

    # Find the resulting agent position in the next state
    nr, nc = -1, -1
    for r in range(len(gns)):
        for c in range(len(gns[r])):
            if gns[r][c] == '@':
                nr, nc = r, c
                break
        if nr != -1:
            break

    if ar == -1 or nr == -1:
        return 0.0

    # If the agent did not move (e.g., hit a boundary), the value is the value of the current position
    if ar == nr and ac == nc:
        gr, gc = -1, -1
        for r in range(len(gs)):
            for c in range(len(gs[r])):
                if gs[r][c] == 'G':
                    gr, gc = r, c
                    break
            if gr != -1:
                break
        if gr == -1:
            return 0.0
        dist = abs(ar - gr) + abs(ac - gc)
        return 0.9 ** dist

    # If movement occurred, check if the agent landed on a hole or the goal in the state grid
    if 0 <= nr < len(gs) and 0 <= nc < len(gs[nr]):
        target_char = gs[nr][nc]
        if target_char == 'H':
            return 0.0
        if target_char == 'G':
            return 1.0

    # If it's a safe cell, estimate Q-value based on the Manhattan distance to the goal
    gr, gc = -1, -1
    # Look for goal in the next_state grid (the goal position might be covered by the @)
    for r in range(len(gns)):
        for c in range(len(gns[r])):
            if gns[r][c] == 'G':
                gr, gc = r, c
                break
        if gr != -1:
            break
    
    if gr == -1:
        # If goal not found in next_state, search the original state
        for r in range(len(gs)):
            for c in range(len(gs[r])):
                if gs[r][c] == 'G':
                    gr, gc = r, c
                    break
            if gr != -1:
                break

    if gr == -1:
        return 0.0

    # Q(s, a) approximation: reward * discount^distance
    dist = abs(nr - gr) + abs(nc - gc)
    return 0.9 ** dist